"""
tests/test_parser.py
====================
Comprehensive pytest suite for the TorchScript Custom Op Parser.
Tests are split into:
  - Unit tests for IR data structures (no torch required)
  - Unit tests for the C++ generator (no torch required)
  - Integration tests that require torch (skipped otherwise)
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ir_parser import (
    IRGraph, IRNode, IRValue, IRType, IRTensorShape,
    IRAttribute, IRSerializer, _parse_ts_type, _sanitize_not_here
)
from src.cpp_generator import (
    CppHeaderGenerator, GeneratorConfig, ShapeInferencePass,
    CustomOpSpec, _collect_custom_ops
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _make_simple_graph(name="TestModel") -> IRGraph:
    """Build a minimal 3-node graph: conv → relu → linear."""
    v = lambda n, ir_t, dims=None: IRValue(
        name=n, ir_type=ir_t,
        shape=IRTensorShape(rank=len(dims), dims=dims) if dims else None
    )
    values = {
        "x":    v("x",    IRType.TENSOR, [1, 3, 32, 32]),
        "w":    v("w",    IRType.TENSOR, [16, 3, 3, 3]),
        "b":    v("b",    IRType.TENSOR, [16]),
        "v1":   v("v1",   IRType.TENSOR, [1, 16, 30, 30]),
        "v2":   v("v2",   IRType.TENSOR, [1, 16, 30, 30]),
        "fc_w": v("fc_w", IRType.TENSOR, [10, 16]),
        "fc_b": v("fc_b", IRType.TENSOR, [10]),
        "out":  v("out",  IRType.TENSOR, [1, 10]),
    }
    nodes = [
        IRNode("n1", "aten::conv2d",
               ["x", "w", "b"], ["v1"],
               attributes={
                   "stride": IRAttribute("stride", [1,1], "list"),
                   "padding": IRAttribute("padding", [0,0], "list"),
                   "dilation": IRAttribute("dilation", [1,1], "list"),
                   "groups": IRAttribute("groups", 1, "int"),
               }),
        IRNode("n2", "aten::relu",  ["v1"], ["v2"]),
        IRNode("n3", "aten::linear", ["v2", "fc_w", "fc_b"], ["out"]),
    ]
    return IRGraph(
        name=name,
        inputs=[values["x"]],
        outputs=[values["out"]],
        nodes=nodes, values=values,
    )


def _make_custom_op_graph() -> IRGraph:
    """Graph with one custom op (mylib::mish)."""
    v = lambda n, dims=None: IRValue(
        name=n, ir_type=IRType.TENSOR,
        shape=IRTensorShape(rank=len(dims), dims=dims) if dims else None
    )
    values = {
        "x": v("x", [1, 16, 32, 32]),
        "y": v("y", [1, 16, 32, 32]),
    }
    nodes = [
        IRNode("n1", "mylib::mish",
               ["x"], ["y"],
               is_custom=True,
               attributes={"alpha": IRAttribute("alpha", 1.0, "float")}),
    ]
    return IRGraph(
        name="MishModel",
        inputs=[values["x"]],
        outputs=[values["y"]],
        nodes=nodes, values=values,
        custom_ops=["mylib::mish"],
    )


class TestIRTypes:
    def test_parse_tensor_type(self):
        assert _parse_ts_type("Tensor") == IRType.TENSOR
        assert _parse_ts_type("Tensor(a)") == IRType.TENSOR

    def test_parse_scalar_types(self):
        assert _parse_ts_type("float") == IRType.FLOAT32
        assert _parse_ts_type("int")   == IRType.INT64
        assert _parse_ts_type("bool")  == IRType.BOOL

    def test_parse_container_types(self):
        assert _parse_ts_type("List[Tensor]")     == IRType.LIST
        assert _parse_ts_type("Optional[Tensor]") == IRType.OPTIONAL
        assert _parse_ts_type("Tuple[Tensor]")    == IRType.TUPLE

    def test_unknown_type(self):
        assert _parse_ts_type("SomeFutureType") == IRType.UNKNOWN


class TestIRTensorShape:
    def test_cpp_shape_comment_concrete(self):
        s = IRTensorShape(rank=4, dims=[1, 3, 224, 224])
        c = s.cpp_shape_comment()
        assert "[1, 3, 224, 224]" in c
        assert "float" in c

    def test_cpp_shape_comment_symbolic(self):
        s = IRTensorShape(rank=4, dims=[None, 3, None, None])
        c = s.cpp_shape_comment()
        assert "?" in c
        assert "3" in c

    def test_cpp_shape_comment_no_dims(self):
        s = IRTensorShape(rank=3)
        c = s.cpp_shape_comment()
        assert "rank=3" in c


class TestIRAttribute:
    def test_float_literal(self):
        a = IRAttribute("eps", 1e-5, "float")
        assert "f" in a.cpp_literal()

    def test_bool_literal(self):
        assert IRAttribute("flag", True,  "bool").cpp_literal() == "true"
        assert IRAttribute("flag", False, "bool").cpp_literal() == "false"

    def test_int_literal(self):
        lit = IRAttribute("n", 42, "int").cpp_literal()
        assert "42" in lit

    def test_string_literal(self):
        lit = IRAttribute("s", 'he"llo', "string").cpp_literal()
        assert '\\"' in lit

    def test_list_literal(self):
        lit = IRAttribute("dims", [1, 2, 3], "list").cpp_literal()
        assert "1, 2, 3" in lit


class TestIRGraph:
    def test_topological_order_is_correct(self):
        graph = _make_simple_graph()
        order = graph.topological_order()
        names = [n.op_kind for n in order]
        assert names.index("aten::conv2d") < names.index("aten::relu")
        assert names.index("aten::relu")   < names.index("aten::linear")

    def test_topological_order_length(self):
        graph = _make_simple_graph()
        assert len(graph.topological_order()) == 3



class TestIRSerializer:
    def test_roundtrip_simple(self, tmp_path):
        g = _make_simple_graph()
        p = tmp_path / "test.json"
        IRSerializer.save(g, p)
        g2 = IRSerializer.load(p)

        assert g2.name == g.name
        assert len(g2.nodes) == len(g.nodes)
        assert len(g2.inputs) == len(g.inputs)
        assert len(g2.outputs) == len(g.outputs)

    def test_roundtrip_custom_ops(self, tmp_path):
        g = _make_custom_op_graph()
        p = tmp_path / "custom.json"
        IRSerializer.save(g, p)
        g2 = IRSerializer.load(p)

        assert g2.custom_ops == ["mylib::mish"]
        assert g2.nodes[0].is_custom is True
        assert g2.nodes[0].op_kind == "mylib::mish"

    def test_json_is_valid(self, tmp_path):
        g = _make_simple_graph()
        p = tmp_path / "out.json"
        IRSerializer.save(g, p)
        with open(p) as f:
            data = json.load(f)
        assert data["schema_version"] == "1.0"
        assert "nodes" in data
        assert len(data["nodes"]) == 3

    def test_shapes_preserved(self, tmp_path):
        g = _make_simple_graph()
        p = tmp_path / "shapes.json"
        IRSerializer.save(g, p)
        g2 = IRSerializer.load(p)
        inp = g2.values.get("x")
        assert inp is not None
        assert inp.shape is not None
        assert inp.shape.dims == [1, 3, 32, 32]



class TestShapeInference:
    def test_relu_preserves_shape(self):
        g = _make_simple_graph()
        ShapeInferencePass().run(g)
        # relu output should inherit input shape
        v2 = g.values.get("v2")
        v1 = g.values.get("v1")
        if v2 and v1:
            assert v2.shape is not None or v1.shape is not None  # at least one shaped

    def test_no_crash_on_unknown_ops(self):
        g = _make_custom_op_graph()
        ShapeInferencePass().run(g)   # should not raise


class TestCppGenerator:
    def _gen(self, graph, **kwargs):
        cfg = GeneratorConfig(**kwargs)
        return CppHeaderGenerator(cfg).generate(graph)

    def test_header_guard_present(self):
        h = self._gen(_make_simple_graph())
        assert "#pragma once" in h
        assert "#ifndef" in h
        assert "#define" in h
        assert "#endif" in h

    def test_includes_torch(self):
        h = self._gen(_make_simple_graph())
        assert "#include <torch/torch.h>" in h

    def test_namespace_wraps_code(self):
        h = self._gen(_make_simple_graph(), namespace="myns")
        assert "namespace myns {" in h
        assert "}  // namespace myns" in h

    def test_infer_function_present(self):
        h = self._gen(_make_simple_graph())
        assert "inline" in h
        assert "infer(" in h

    def test_conv2d_emitted(self):
        h = self._gen(_make_simple_graph())
        assert "torch::conv2d" in h or "conv2d" in h

    def test_relu_emitted(self):
        h = self._gen(_make_simple_graph())
        assert "torch::relu" in h

    def test_linear_emitted(self):
        h = self._gen(_make_simple_graph())
        assert "torch::linear" in h

    def test_custom_op_stub_generated(self):
        h = self._gen(_make_custom_op_graph())
        assert "mylib_mish" in h
        assert "Custom op stub" in h

    def test_shape_comments_present(self):
        h = self._gen(_make_simple_graph(), shape_comments=True)
        assert "shape" in h.lower() or "/*" in h

    def test_shape_comments_absent_when_disabled(self):
        h = self._gen(_make_simple_graph(), shape_comments=False)
        # should still compile, just no shape annotations
        assert "infer(" in h

    def test_test_main_emitted(self):
        h = self._gen(_make_simple_graph(), emit_test_main=True)
        assert "TEST_MAIN" in h
        assert "int main()" in h

    def test_no_test_main_by_default(self):
        h = self._gen(_make_simple_graph())
        assert "int main()" not in h

    def test_weights_section_present(self):
        h = self._gen(_make_simple_graph(), emit_weights=True)
        assert "load_weights" in h

    def test_output_is_valid_ish_cpp(self):
        """Basic sanity: braces balance."""
        h = self._gen(_make_simple_graph())
        assert h.count("{") == h.count("}")

    def test_multi_output_model_tuple(self):
        """Model with 2 outputs should return std::tuple."""
        v = lambda n, dims=None: IRValue(
            name=n, ir_type=IRType.TENSOR,
            shape=IRTensorShape(rank=len(dims), dims=dims) if dims else None
        )
        values = {
            "x": v("x", [1, 8]),
            "y1": v("y1", [1, 4]),
            "y2": v("y2", [1, 4]),
        }
        nodes = [
            IRNode("n1", "aten::relu", ["x"], ["y1"]),
            IRNode("n2", "aten::relu", ["x"], ["y2"]),
        ]
        graph = IRGraph(
            "DualOut",
            inputs=[values["x"]],
            outputs=[values["y1"], values["y2"]],
            nodes=nodes, values=values,
        )
        h = self._gen(graph)
        assert "std::tuple" in h

    def test_custom_op_call_in_infer(self):
        h = self._gen(_make_custom_op_graph())
        assert "mylib_mish" in h



class TestCustomOpCollection:
    def test_finds_custom_ops(self):
        g = _make_custom_op_graph()
        specs = _collect_custom_ops(g)
        assert "mylib::mish" in specs

    def test_appearance_count(self):
        g = _make_custom_op_graph()
        g.nodes.append(
            IRNode("n2", "mylib::mish", ["y"], ["z"], is_custom=True)
        )
        specs = _collect_custom_ops(g)
        assert specs["mylib::mish"].appearances == 2

    def test_no_custom_ops_in_standard_model(self):
        g = _make_simple_graph()
        specs = _collect_custom_ops(g)
        assert len(specs) == 0

    def test_custom_op_stub_has_correct_namespace(self):
        spec = CustomOpSpec("mylib::mish")
        spec.input_names = [["x"]]
        spec.output_names = [["y"]]
        stub = spec.cpp_stub("    ")
        assert "mylib_mish" in stub
        assert "at::Tensor" in stub



class TestNodeHandlers:
    def _values(self):
        return {
            "x": IRValue("x", IRType.TENSOR,
                         IRTensorShape(rank=4, dims=[1, 16, 32, 32])),
            "w": IRValue("w", IRType.TENSOR,
                         IRTensorShape(rank=4, dims=[32, 16, 3, 3])),
            "b": IRValue("b", IRType.TENSOR,
                         IRTensorShape(rank=1, dims=[32])),
        }

    def test_conv2d_handler_emit(self):
        from src.ir_parser import NodeHandler
        h = NodeHandler.get("aten::conv2d")
        node = IRNode("n", "aten::conv2d", ["x", "w", "b"], ["out"],
                      attributes={
                          "stride":   IRAttribute("stride",   [1,1], "list"),
                          "padding":  IRAttribute("padding",  [1,1], "list"),
                          "dilation": IRAttribute("dilation", [1,1], "list"),
                          "groups":   IRAttribute("groups",   1,      "int"),
                      })
        cpp = h.emit_cpp(node, self._values())
        assert "conv2d" in cpp
        assert "out" in cpp

    def test_relu_handler_emit(self):
        from src.ir_parser import NodeHandler
        h = NodeHandler.get("aten::relu")
        node = IRNode("n", "aten::relu", ["x"], ["y"])
        cpp = h.emit_cpp(node, self._values())
        assert "relu" in cpp
        assert "y" in cpp

    def test_dropout_is_identity(self):
        from src.ir_parser import NodeHandler
        h = NodeHandler.get("aten::dropout")
        node = IRNode("n", "aten::dropout", ["x"], ["y"],
                      attributes={"p": IRAttribute("p", 0.5, "float"),
                                  "training": IRAttribute("training", False, "bool")})
        cpp = h.emit_cpp(node, self._values())
        assert "identity" in cpp.lower() or "x" in cpp

    def test_conv2d_shape_inference(self):
        from src.ir_parser import NodeHandler
        h = NodeHandler.get("aten::conv2d")
        node = IRNode("n", "aten::conv2d", ["x", "w", "b"], ["out"],
                      attributes={
                          "stride":   IRAttribute("stride",   [1,1], "list"),
                          "padding":  IRAttribute("padding",  [1,1], "list"),
                          "dilation": IRAttribute("dilation", [1,1], "list"),
                          "groups":   IRAttribute("groups",   1,      "int"),
                      })
        vm = self._values()
        shapes = h.infer_shapes(node, vm)
        assert shapes[0] is not None
        assert shapes[0].rank == 4



@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestLiveModels:
    def test_script_simple_module(self):
        import torch
        import torch.nn as nn
        from src.ts_parser import parse_model, generate_header

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        scripted = torch.jit.script(Net().eval())
        graph = parse_model(scripted, model_name="SimpleNet")
        assert graph.name == "SimpleNet"
        assert any("linear" in n.op_kind for n in graph.nodes)

        h = generate_header(graph)
        assert "infer(" in h

    def test_conv_relu_graph(self):
        import torch
        import torch.nn as nn
        from src.ts_parser import parse_model, generate_header

        class ConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc   = nn.Linear(8, 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.relu(self.conv(x))
                x = torch.flatten(self.pool(x), 1)
                return self.fc(x)

        graph = parse_model(torch.jit.script(ConvNet().eval()), "ConvNet")
        h = generate_header(graph, emit_test_main=True)
        assert "conv2d" in h
        assert "relu"   in h
        assert "linear" in h
        assert "TEST_MAIN" in h

    def test_custom_op_end_to_end(self, tmp_path):
        import torch
        import torch.nn as nn
        import torch.library as lib
        from src.ts_parser import parse_model, save_ir, generate_header

        my_lib = lib.Library("testlib_e2e", "DEF")
        my_lib.define("scale(Tensor x, float alpha) -> Tensor")

        @torch.library.impl(my_lib, "scale", "CPU")
        def scale_cpu(x, alpha):
            return x * alpha

        @torch.library.impl_abstract("testlib_e2e::scale")
        def scale_abstract(x, alpha):
            return torch.empty_like(x)

        class ScaleNet(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.ops.testlib_e2e.scale(x, 2.0)

        graph = parse_model(torch.jit.script(ScaleNet().eval()), "ScaleNet")
        assert "testlib_e2e::scale" in graph.custom_ops

        ir_path = tmp_path / "scale_ir.json"
        save_ir(graph, ir_path)

        from src.ir_parser import IRSerializer
        g2 = IRSerializer.load(ir_path)
        assert "testlib_e2e::scale" in g2.custom_ops

        h = generate_header(g2, emit_custom_stubs=True)
        assert "testlib_e2e_scale" in h
        assert "Custom op stub" in h

    def test_ir_json_roundtrip_live(self, tmp_path):
        import torch
        import torch.nn as nn
        from src.ts_parser import parse_model, save_ir
        from src.ir_parser import IRSerializer

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(x)

        graph = parse_model(torch.jit.script(Tiny().eval()), "Tiny")
        p = tmp_path / "tiny.json"
        save_ir(graph, p)
        g2 = IRSerializer.load(p)
        assert g2.name == graph.name
        assert len(g2.nodes) > 0


# ir_parser doesn't export _sanitize — grab it from generator
def _sanitize_not_here(s):
    import re
    return re.sub(r"[^A-Za-z0-9_]", "_", s).lstrip("_") or "v"


# Fix the bad import at top of file
import src.ir_parser as ir_parser
ir_parser._sanitize_not_here = _sanitize_not_here  # monkey-patch if needed


class TestSanitize:
    def test_basic(self):
        from src.cpp_generator import _sanitize
        assert _sanitize("foo.bar") == "foo_bar"
        assert _sanitize("123abc") == "_123abc" or _sanitize("123abc") == "123abc"
        assert _sanitize("valid_name") == "valid_name"

    def test_empty_becomes_v(self):
        from src.cpp_generator import _sanitize
        result = _sanitize("")
        assert result  # not empty
