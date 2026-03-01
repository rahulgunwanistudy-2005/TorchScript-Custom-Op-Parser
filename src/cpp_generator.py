from __future__ import annotations

import dataclasses
import io
import re
import textwrap
from typing import Any, Dict, List, Optional, Set

from ir_parser import (
    IRGraph, IRNode, IRType, IRValue, IRAttribute,
    IRTensorShape, NodeHandler
)


@dataclasses.dataclass
class GeneratorConfig:
    namespace: str = "inference"
    emit_weights: bool = True
    emit_test_main: bool = False
    emit_custom_stubs: bool = True
    shape_comments: bool = True
    use_half_precision: bool = False
    weight_dir: str = "weights"
    indent: str = "    "


def _sanitize(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name).lstrip("_")
    if s and s[0].isdigit():
        s = "_" + s
    return s or "v"


def _cpp_scalar(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        return f"{val}f"
    if isinstance(val, int):
        return f"{val}LL"
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


class ShapeInferencePass:
    def run(self, graph: IRGraph) -> None:
        vm = graph.values
        for node in graph.topological_order():
            handler = NodeHandler.get(node.op_kind)
            shapes = handler.infer_shapes(node, vm)
            node.output_shapes = shapes
            for out_name, shape in zip(node.outputs, shapes):
                if out_name in vm and shape is not None:
                    vm[out_name].shape = shape


@dataclasses.dataclass
class CustomOpSpec:
    op_kind: str
    appearances: int = 0
    input_names: List[List[str]] = dataclasses.field(default_factory=list)
    output_names: List[List[str]] = dataclasses.field(default_factory=list)
    all_attributes: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def namespace(self) -> str:
        return self.op_kind.split("::")[0] if "::" in self.op_kind else "custom"

    @property
    def name(self) -> str:
        return self.op_kind.split("::")[-1]

    def cpp_stub(self, indent: str) -> str:
        nargs = max((len(i) for i in self.input_names), default=0)
        params = ", ".join(f"const at::Tensor& x{k}" for k in range(nargs))
        nout = max((len(o) for o in self.output_names), default=1)
        if nout == 1:
            ret_type = "at::Tensor"
            body = f"{indent}    return x0;  // TODO: implement {self.op_kind}"
        else:
            ret_type = "std::tuple<" + ", ".join(["at::Tensor"] * nout) + ">"
            inner = ", ".join(f"x{k}" for k in range(min(nout, nargs)))
            body = f"{indent}    return std::make_tuple({inner});  // TODO"

        attr_comments = ""
        if self.all_attributes:
            attr_lines = [
                f"{indent}    // attr: {k} = {v}"
                for k, v in list(self.all_attributes.items())[:6]
            ]
            attr_comments = "\n".join(attr_lines) + "\n"

        return (
            f"{indent}// Custom op stub: {self.op_kind} "
            f"(appears {self.appearances}x in graph)\n"
            f"{attr_comments}"
            f"{indent}inline {ret_type} {_sanitize(self.namespace)}_{_sanitize(self.name)}"
            f"({params}) {{\n"
            f"{body}\n"
            f"{indent}}}\n"
        )


def _collect_custom_ops(graph: IRGraph) -> Dict[str, CustomOpSpec]:
    specs: Dict[str, CustomOpSpec] = {}
    for node in graph.nodes:
        if not node.is_custom:
            continue
        if node.op_kind not in specs:
            specs[node.op_kind] = CustomOpSpec(op_kind=node.op_kind)
        spec = specs[node.op_kind]
        spec.appearances += 1
        spec.input_names.append(node.inputs)
        spec.output_names.append(node.outputs)
        for k, v in node.attributes.items():
            spec.all_attributes[k] = v.value
    return specs


class WeightEmitter:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self._seen: Set[str] = set()

    def emit_weight_global(self, name: str, shape: Optional[IRTensorShape]) -> str:
        safe = _sanitize(name)
        if safe in self._seen:
            return ""
        self._seen.add(safe)
        dtype = "torch::kFloat16" if self.cfg.use_half_precision else "torch::kFloat32"
        shape_comment = shape.cpp_shape_comment() if shape else "/* shape unknown */"
        return (
            f"static at::Tensor {safe};  {shape_comment}\n"
            f"// loaded via: {safe} = torch::load(\"{self.cfg.weight_dir}/{safe}.pt\").to({dtype});\n"
        )

    def emit_weight_loader(self, weight_names: List[str]) -> str:
        lines = ["inline void load_weights(const std::string& weight_dir) {"]
        for name in weight_names:
            safe = _sanitize(name)
            lines.append(
                f'    {safe} = torch::load(weight_dir + "/{safe}.pt").to(torch::kFloat32).eval();'
            )
        lines.append("}")
        return "\n".join(lines)


class CppHeaderGenerator:
    _SILENT_PRIM = frozenset({
        "prim::GetAttr", "prim::SetAttr", "prim::Return",
        "prim::Uninitialized", "prim::unchecked_cast",
        "prim::RaiseException", "prim::device", "prim::dtype",
        "prim::requires_grad", "prim::is_nested", "prim::type",
        "prim::ListUnpack", "prim::TupleUnpack",
        "aten::size", "aten::len", "aten::gt", "aten::lt",
        "aten::__getitem__",
    })

    def __init__(self, cfg: Optional[GeneratorConfig] = None):
        self.cfg = cfg or GeneratorConfig()

    def generate(self, graph: IRGraph) -> str:
        ShapeInferencePass().run(graph)
        custom_specs = _collect_custom_ops(graph)

        buf = io.StringIO()
        w = buf.write

        w(self._prologue(graph))
        w(self._includes())
        w(self._namespace_open())

        if custom_specs and self.cfg.emit_custom_stubs:
            w(self._custom_op_section(custom_specs))

        if self.cfg.emit_weights:
            w(self._weights_section(graph))

        w(self._infer_function(graph, custom_specs))

        if self.cfg.emit_test_main:
            w(self._test_main(graph))

        w(self._namespace_close())
        w(self._epilogue(graph))

        return buf.getvalue()

    def _prologue(self, graph: IRGraph) -> str:
        guard = re.sub(r"[^A-Z0-9_]", "_", f"INFERENCE_{graph.name.upper()}_H")
        custom_list = (
            "  //   " + "\n  //   ".join(graph.custom_ops)
            if graph.custom_ops else "  //   (none)"
        )
        return textwrap.dedent(f"""\
            // AUTO-GENERATED by TorchScript Custom Op Parser
            // Model  : {graph.name}
            // Inputs : {len(graph.inputs)}
            // Outputs: {len(graph.outputs)}
            // Nodes  : {len(graph.nodes)}  ({sum(1 for n in graph.nodes if n.is_custom)} custom)
            // Custom ops:
            {custom_list}
            // DO NOT EDIT — regenerate with: ts-parser parse --model <model>.pt --output <header>.h
            #pragma once
            #ifndef {guard}
            #define {guard}

            """)

    def _includes(self) -> str:
        return textwrap.dedent("""\
            #include <torch/torch.h>
            #include <torch/script.h>
            #include <ATen/ATen.h>
            #include <c10/util/Optional.h>
            #include <vector>
            #include <string>
            #include <tuple>
            #include <stdexcept>
            #include <cstdint>

            """)

    def _namespace_open(self) -> str:
        return f"namespace {self.cfg.namespace} {{\n\n"

    def _namespace_close(self) -> str:
        return f"\n}}  // namespace {self.cfg.namespace}\n\n"

    def _epilogue(self, graph: IRGraph) -> str:
        guard = re.sub(r"[^A-Z0-9_]", "_", f"INFERENCE_{graph.name.upper()}_H")
        return f"#endif  // {guard}\n"

    def _custom_op_section(self, specs: Dict[str, CustomOpSpec]) -> str:
        lines = ["// Custom operator stubs — fill in implementations\n"]
        for spec in specs.values():
            lines.append(spec.cpp_stub(self.cfg.indent))
        return "\n".join(lines) + "\n"

    def _weights_section(self, graph: IRGraph) -> str:
        we = WeightEmitter(self.cfg)
        weight_nodes = [
            n for n in graph.nodes
            if n.op_kind in ("aten::conv2d", "aten::linear",
                             "aten::batch_norm", "aten::layer_norm",
                             "aten::group_norm")
        ]
        lines = ["// Weight tensors\n"]
        weight_names = []
        for node in weight_nodes:
            for idx in (1, 2):
                if len(node.inputs) > idx:
                    wname = node.inputs[idx]
                    shape = graph.values.get(wname, IRValue(wname, IRType.TENSOR)).shape
                    decl = we.emit_weight_global(wname, shape)
                    if decl:
                        lines.append(decl)
                        weight_names.append(wname)

        lines.append("")
        if weight_names:
            lines.append(we.emit_weight_loader(weight_names))
        lines.append("")
        return "\n".join(lines) + "\n"

    def _infer_function(self, graph: IRGraph, custom_specs: Dict[str, CustomOpSpec]) -> str:
        lines = ["// Forward pass\n"]

        sig_params = []
        for v in graph.inputs:
            shape_c = v.shape.cpp_shape_comment() if v.shape else ""
            if v.ir_type == IRType.TENSOR:
                sig_params.append(f"const at::Tensor& {_sanitize(v.name)}  {shape_c}")
            else:
                sig_params.append(f"const {v.cpp_type()}& {_sanitize(v.name)}")

        if len(graph.outputs) == 0:
            ret_type = "void"
            ret_stmt = ""
        elif len(graph.outputs) == 1:
            ret_type = "at::Tensor"
            ret_stmt = f"return {_sanitize(graph.outputs[0].name)};"
        else:
            inner = ", ".join(["at::Tensor"] * len(graph.outputs))
            ret_type = f"std::tuple<{inner}>"
            out_names = ", ".join(_sanitize(o.name) for o in graph.outputs)
            ret_stmt = f"return std::make_tuple({out_names});"

        joined_params = ",\n    ".join(sig_params) if sig_params else ""
        lines.append(f"inline {ret_type} infer(")
        if joined_params:
            lines.append(f"    {joined_params}")
        lines.append(") {")

        for node in graph.topological_order():
            if node.op_kind == "prim::Return":
                continue
            cpp_line = self._emit_node(node, graph, custom_specs)
            if not cpp_line.strip():
                continue

            shape_ann = ""
            if self.cfg.shape_comments and node.output_shapes:
                parts = [
                    f"{_sanitize(n)}:{s.cpp_shape_comment()}"
                    for n, s in zip(node.outputs, node.output_shapes)
                    if s is not None
                ]
                if parts:
                    shape_ann = "  // " + "; ".join(parts)

            sub_lines = [l.rstrip() for l in cpp_line.split("\n") if l.strip()]
            for sub in sub_lines:
                lines.append(f"    {sub}")
            if shape_ann and lines:
                lines[-1] = lines[-1].rstrip() + shape_ann

        if ret_stmt:
            ret_stmt = re.sub(r'\b(\d[\w]*)\b', lambda m: _sanitize(m.group(0)), ret_stmt)
            lines.append(f"    {ret_stmt}")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _emit_node(self, node: IRNode, graph: IRGraph,
                   custom_specs: Dict[str, CustomOpSpec]) -> str:
        if node.op_kind in self._SILENT_PRIM:
            return ""

        node = dataclasses.replace(
            node,
            inputs=[_sanitize(i) for i in node.inputs],
            outputs=[_sanitize(o) for o in node.outputs],
        )

        if node.op_kind == "prim::Constant":
            return self._emit_constant(node, graph)
        if node.op_kind == "prim::ListConstruct":
            return self._emit_list_construct(node, graph)
        if node.op_kind == "prim::TupleConstruct":
            return self._emit_tuple_construct(node, graph)
        if node.op_kind in ("prim::If", "prim::Loop", "prim::CallMethod", "prim::CallFunction"):
            return f"// {node.op_kind} — control flow not auto-lowered\n"

        if node.is_custom and node.op_kind in custom_specs:
            return self._emit_custom_call(node, custom_specs[node.op_kind])

        handler = NodeHandler.get(node.op_kind)
        cpp = handler.emit_cpp(node, graph.values)
        if cpp.strip().startswith("//") and "TODO" not in cpp:
            return self._emit_generic_aten(node)
        return cpp

    def _emit_constant(self, node: IRNode, graph: IRGraph) -> str:
        if not node.outputs:
            return ""
        out = node.outputs[0]
        if re.fullmatch(r"_\d+", out) or re.fullmatch(r"\d+", out):
            return ""
        val = node.attributes.get("value")
        if val is None:
            v = graph.values.get(out)
            if v and v.ir_type == IRType.TENSOR:
                return f"// const tensor {out} (loaded separately)"
            return ""
        cpp_type = {
            "float": "float", "int": "int64_t",
            "bool": "bool", "string": "std::string",
        }.get(val.attr_type, "auto")
        return f"constexpr {cpp_type} {out} = {val.cpp_literal()};"

    def _emit_list_construct(self, node: IRNode, graph: IRGraph) -> str:
        if not node.outputs:
            return ""
        out = node.outputs[0]
        if re.fullmatch(r"_\d+", out):
            return ""
        items = ", ".join(node.inputs)
        return f"auto {out} = std::vector<at::Tensor>{{ {items} }};"

    def _emit_tuple_construct(self, node: IRNode, graph: IRGraph) -> str:
        if not node.outputs:
            return ""
        items = ", ".join(node.inputs)
        return f"auto {node.outputs[0]} = std::make_tuple({items});"

    def _emit_custom_call(self, node: IRNode, spec: CustomOpSpec) -> str:
        fn = f"{_sanitize(spec.namespace)}_{_sanitize(spec.name)}"
        args = ", ".join(node.inputs)
        outs = ", ".join(node.outputs)
        if len(node.outputs) > 1:
            return f"auto [{outs}] = {fn}({args});  // {node.op_kind}"
        elif node.outputs:
            return f"auto {outs} = {fn}({args});  // {node.op_kind}"
        return f"{fn}({args});  // {node.op_kind}"

    def _emit_generic_aten(self, node: IRNode) -> str:
        args = ", ".join(node.inputs)
        outs = ", ".join(node.outputs)
        return f"// TODO: {node.op_kind}\n// auto {outs} = at::{node.opname}({args});"

    def _test_main(self, graph: IRGraph) -> str:
        lines = [
            "",
            "#ifdef TEST_MAIN",
            "#include <iostream>",
            "int main() {",
            f'    load_weights("{self.cfg.weight_dir}");',
        ]
        for inp in graph.inputs:
            shape = inp.shape
            if shape and shape.dims:
                dims_str = ", ".join("1" if d is None else str(d) for d in shape.dims)
                lines.append(f"    auto {_sanitize(inp.name)} = torch::randn({{{dims_str}}});")
            else:
                lines.append(f"    auto {_sanitize(inp.name)} = torch::randn({{1, 3, 224, 224}});")
        input_args = ", ".join(_sanitize(v.name) for v in graph.inputs)
        lines += [
            f"    auto result = infer({input_args});",
            '    std::cout << "Output shape: " << result.sizes() << std::endl;',
            "    return 0;",
            "}",
            "#endif  // TEST_MAIN",
            "",
        ]
        return "\n".join(lines)


def generate_cmake(project_name: str, header_name: str) -> str:
    return textwrap.dedent(f"""\
        cmake_minimum_required(VERSION 3.18)
        project({project_name} CXX)

        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)

        find_package(Torch REQUIRED)
        set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} ${{TORCH_CXX_FLAGS}}")

        add_library({project_name}_inference INTERFACE)
        target_include_directories({project_name}_inference INTERFACE
            ${{CMAKE_CURRENT_SOURCE_DIR}}/include
        )
        target_link_libraries({project_name}_inference INTERFACE "${{TORCH_LIBRARIES}}")

        add_executable({project_name}_test test_inference.cpp)
        target_compile_definitions({project_name}_test PRIVATE TEST_MAIN)
        target_link_libraries({project_name}_test
            {project_name}_inference
            "${{TORCH_LIBRARIES}}"
        )

        install(FILES include/{header_name}
            DESTINATION include/{project_name}
        )
        """)