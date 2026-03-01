"""
TorchScript IR Parser
=====================
Walks the TorchScript computation graph (either live torch.jit.ScriptModule
or a serialized .pt file) and builds an abstract IR that is architecture-
independent.  A backend then lowers that IR to a self-contained C++ inference
header.

Design goals
------------
* Research-grade: expose every node, attribute, and custom-op in the graph.
* Zero mandatory dependencies beyond the standard library + optional torch.
* Extensible: new node handlers are registered via @NodeHandler.register().
* Correct type mapping: scalar/tensor/list/optional/tuple → C++ equivalents.
"""

from __future__ import annotations

import abc
import dataclasses
import enum
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


class IRType(enum.Enum):
    """Canonical scalar / aggregate types used in the IR."""
    FLOAT32  = "float"
    FLOAT64  = "double"
    INT32    = "int32_t"
    INT64    = "int64_t"
    BOOL     = "bool"
    TENSOR   = "Tensor"
    LIST     = "List"
    OPTIONAL = "Optional"
    TUPLE    = "Tuple"
    STRING   = "std::string"
    UNKNOWN  = "/*unknown*/"


@dataclasses.dataclass
class IRTensorShape:
    """Symbolic or concrete tensor shape."""
    rank: Optional[int] = None
    dims: Optional[List[Optional[int]]] = None   
    dtype: IRType = IRType.FLOAT32

    def cpp_shape_comment(self) -> str:
        if self.dims is None:
            return f"/* rank={self.rank}, dtype={self.dtype.value} */"
        dim_str = ", ".join("?" if d is None else str(d) for d in self.dims)
        return f"/* [{dim_str}], dtype={self.dtype.value} */"


@dataclasses.dataclass
class IRValue:
    """A named SSA value in the graph."""
    name: str
    ir_type: IRType
    shape: Optional[IRTensorShape] = None
    is_output: bool = False
    is_input: bool = False
    debug_name: str = ""

    def cpp_type(self) -> str:
        if self.ir_type == IRType.TENSOR:
            return "at::Tensor"
        if self.ir_type == IRType.LIST:
            return "std::vector<at::Tensor>"
        if self.ir_type == IRType.OPTIONAL:
            return "c10::optional<at::Tensor>"
        return self.ir_type.value


@dataclasses.dataclass
class IRAttribute:
    """A constant / attribute on a node (weight, bias, config parameter)."""
    name: str
    value: Any
    attr_type: str 

    def cpp_literal(self) -> str:
        if self.attr_type == "float":
            return f"{self.value}f"
        if self.attr_type == "double":
            return str(self.value)
        if self.attr_type == "int":
            return f"{self.value}LL"
        if self.attr_type == "bool":
            return "true" if self.value else "false"
        if self.attr_type == "string":
            escaped = str(self.value).replace('"', '\\"')
            return f'"{escaped}"'
        if self.attr_type == "list":
            inner = ", ".join(str(v) for v in self.value)
            return f"{{{inner}}}"
        return "/*tensor attribute*/"


@dataclasses.dataclass
class IRNode:
    """A single operation node in the IR graph."""
    node_id: str
    op_kind: str           
    inputs: List[str]      
    outputs: List[str]     
    attributes: Dict[str, IRAttribute] = dataclasses.field(default_factory=dict)
    is_custom: bool = False
    source_location: str = ""
    # populated by shape-inference pass
    output_shapes: List[Optional[IRTensorShape]] = dataclasses.field(default_factory=list)

    @property
    def namespace(self) -> str:
        return self.op_kind.split("::")[0] if "::" in self.op_kind else ""

    @property
    def opname(self) -> str:
        return self.op_kind.split("::")[-1]


@dataclasses.dataclass
class IRGraph:
    """Complete IR for a TorchScript module."""
    name: str
    inputs: List[IRValue]
    outputs: List[IRValue]
    nodes: List[IRNode]
    values: Dict[str, IRValue] = dataclasses.field(default_factory=dict)
    custom_ops: List[str] = dataclasses.field(default_factory=list)
    subgraphs: Dict[str, "IRGraph"] = dataclasses.field(default_factory=dict)
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def topological_order(self) -> List[IRNode]:
        """Kahn's algorithm — returns nodes in execution order."""
        from collections import deque
        produced: Dict[str, IRNode] = {}
        for node in self.nodes:
            for out in node.outputs:
                produced[out] = node

        in_degree: Dict[str, int] = {n.node_id: 0 for n in self.nodes}
        deps: Dict[str, List[IRNode]] = {n.node_id: [] for n in self.nodes}

        for node in self.nodes:
            for inp in node.inputs:
                if inp in produced and produced[inp].node_id != node.node_id:
                    in_degree[node.node_id] += 1
                    deps[produced[inp].node_id].append(node)

        q: deque[IRNode] = deque(n for n in self.nodes if in_degree[n.node_id] == 0)
        order: List[IRNode] = []
        while q:
            node = q.popleft()
            order.append(node)
            for child in deps[node.node_id]:
                in_degree[child.node_id] -= 1
                if in_degree[child.node_id] == 0:
                    q.append(child)

        if len(order) != len(self.nodes):
            logger.warning("Cycle detected in graph — returning partial order")
        return order


_TORCH_DTYPE_MAP = {
    "Float":  IRType.FLOAT32,
    "Double": IRType.FLOAT64,
    "Long":   IRType.INT64,
    "Int":    IRType.INT32,
    "Bool":   IRType.BOOL,
    "float":  IRType.FLOAT32,
    "double": IRType.FLOAT64,
    "int":    IRType.INT64,
    "bool":   IRType.BOOL,
}


def _parse_ts_type(ts_type_str: str) -> IRType:
    """Convert a TorchScript type string to IRType."""
    s = ts_type_str.strip()
    if s.startswith("Tensor"):
        return IRType.TENSOR
    if s.startswith("List[") or s.startswith("list"):
        return IRType.LIST
    if s.startswith("Optional["):
        return IRType.OPTIONAL
    if s.startswith("Tuple[") or s.startswith("tuple"):
        return IRType.TUPLE
    if s in ("str", "string"):
        return IRType.STRING
    if s in ("float",):
        return IRType.FLOAT32
    if s in ("int", "int64_t"):
        return IRType.INT64
    if s in ("bool",):
        return IRType.BOOL
    return _TORCH_DTYPE_MAP.get(s, IRType.UNKNOWN)


def _dtype_to_irtype(dtype) -> IRType:  # dtype is torch.dtype
    try:
        import torch
        return {
            torch.float32: IRType.FLOAT32,
            torch.float64: IRType.FLOAT64,
            torch.int32:   IRType.INT32,
            torch.int64:   IRType.INT64,
            torch.bool:    IRType.BOOL,
        }.get(dtype, IRType.FLOAT32)
    except ImportError:
        return IRType.FLOAT32


class NodeHandler:
    """
    Base class for node-specific logic (shape inference, custom C++ emission).
    Sub-classes register themselves for one or more op kinds.
    """
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, *op_kinds: str):
        def decorator(sub: type) -> type:
            for k in op_kinds:
                cls._registry[k] = sub
            return sub
        return decorator

    @classmethod
    def get(cls, op_kind: str) -> "NodeHandler":
        handler_cls = cls._registry.get(op_kind) or cls._registry.get("*")
        return (handler_cls or NodeHandler)()

    def infer_shapes(self, node: IRNode, value_map: Dict[str, IRValue]) -> List[Optional[IRTensorShape]]:
        return [None] * len(node.outputs)

    def emit_cpp(self, node: IRNode, value_map: Dict[str, IRValue]) -> str:
        return f"// {node.op_kind}({', '.join(node.inputs)}) -> ({', '.join(node.outputs)})"


# Built-in handlers

@NodeHandler.register("aten::conv2d")
class Conv2dHandler(NodeHandler):
    def infer_shapes(self, node, vm):
        inp = vm.get(node.inputs[0]) if node.inputs else None
        if inp and inp.shape and inp.shape.dims:
            N, C, H, W = inp.shape.dims
            stride = node.attributes.get("stride")
            padding = node.attributes.get("padding")
            kernel_h, kernel_w = 3, 3  # default; refined from weight shape
            weight_val = vm.get(node.inputs[1]) if len(node.inputs) > 1 else None
            if weight_val and weight_val.shape and weight_val.shape.dims:
                out_C, _, kernel_h, kernel_w = weight_val.shape.dims
            else:
                out_C = None
            sh, sw = (1, 1)
            ph, pw = (0, 0)
            if stride and isinstance(stride.value, (list, tuple)) and len(stride.value) >= 2:
                sh, sw = stride.value[0], stride.value[1]
            if padding and isinstance(padding.value, (list, tuple)) and len(padding.value) >= 2:
                ph, pw = padding.value[0], padding.value[1]
            out_H = None if H is None else (H + 2*ph - kernel_h)//sh + 1
            out_W = None if W is None else (W + 2*pw - kernel_w)//sw + 1
            return [IRTensorShape(rank=4, dims=[N, out_C, out_H, out_W])]
        return [IRTensorShape(rank=4)]

    def emit_cpp(self, node, vm):
        i = node.inputs
        attrs = node.attributes
        stride  = attrs.get("stride",  IRAttribute("stride",  [1,1], "list"))
        padding = attrs.get("padding", IRAttribute("padding", [0,0], "list"))
        dilation= attrs.get("dilation",IRAttribute("dilation",[1,1], "list"))
        groups  = attrs.get("groups",  IRAttribute("groups",  1,     "int"))
        bias    = i[2] if len(i) > 2 else "/*no bias*/"
        return (
            f"auto {node.outputs[0]} = torch::conv2d("
            f"{i[0]}, {i[1]}, {bias}, "
            f"{{{', '.join(str(v) for v in stride.value)}}}, "
            f"{{{', '.join(str(v) for v in padding.value)}}}, "
            f"{{{', '.join(str(v) for v in dilation.value)}}}, "
            f"{groups.value});"
        )


@NodeHandler.register("aten::batch_norm")
class BatchNormHandler(NodeHandler):
    def infer_shapes(self, node, vm):
        if node.inputs:
            inp = vm.get(node.inputs[0])
            if inp and inp.shape:
                return [inp.shape]
        return [IRTensorShape(rank=4)]

    def emit_cpp(self, node, vm):
        i = node.inputs
        eps   = node.attributes.get("eps",     IRAttribute("eps",     1e-5, "float"))
        mom   = node.attributes.get("momentum",IRAttribute("momentum",0.1,  "float"))
        train = node.attributes.get("training",IRAttribute("training",False,"bool"))
        return (
            f"auto {node.outputs[0]} = torch::batch_norm("
            f"{i[0]}, {i[1]}, {i[2]}, {i[3]}, {i[4]}, "
            f"{train.cpp_literal()}, {mom.cpp_literal()}, {eps.cpp_literal()}, true);"
        )


@NodeHandler.register("aten::linear")
class LinearHandler(NodeHandler):
    def infer_shapes(self, node, vm):
        inp = vm.get(node.inputs[0]) if node.inputs else None
        wt  = vm.get(node.inputs[1]) if len(node.inputs) > 1 else None
        if inp and inp.shape and inp.shape.dims and wt and wt.shape and wt.shape.dims:
            *batch, _ = inp.shape.dims
            out_f = wt.shape.dims[0]
            return [IRTensorShape(rank=len(batch)+1, dims=batch+[out_f])]
        return [None]

    def emit_cpp(self, node, vm):
        i = node.inputs
        bias = i[2] if len(i) > 2 else "/*no bias*/"
        return f"auto {node.outputs[0]} = torch::linear({i[0]}, {i[1]}, {bias});"


@NodeHandler.register("aten::relu", "aten::relu_")
class ReluHandler(NodeHandler):
    def infer_shapes(self, node, vm):
        inp = vm.get(node.inputs[0]) if node.inputs else None
        return [inp.shape if inp else None]

    def emit_cpp(self, node, vm):
        return f"auto {node.outputs[0]} = torch::relu({node.inputs[0]});"


@NodeHandler.register("aten::max_pool2d")
class MaxPool2dHandler(NodeHandler):
    def infer_shapes(self, node, vm):
        return [IRTensorShape(rank=4)]

    def emit_cpp(self, node, vm):
        i = node.inputs
        attrs = node.attributes
        kernel  = attrs.get("kernel_size", IRAttribute("kernel_size", [2,2], "list"))
        stride  = attrs.get("stride",      IRAttribute("stride",      [2,2], "list"))
        padding = attrs.get("padding",     IRAttribute("padding",     [0,0], "list"))
        return (
            f"auto {node.outputs[0]} = torch::max_pool2d("
            f"{i[0]}, {{{', '.join(str(v) for v in kernel.value)}}}, "
            f"{{{', '.join(str(v) for v in stride.value)}}}, "
            f"{{{', '.join(str(v) for v in padding.value)}}});"
        )


@NodeHandler.register("aten::adaptive_avg_pool2d")
class AdaptiveAvgPool2dHandler(NodeHandler):
    def emit_cpp(self, node, vm):
        i = node.inputs
        out_size = node.attributes.get("output_size", IRAttribute("output_size", [1,1], "list"))
        return (
            f"auto {node.outputs[0]} = torch::adaptive_avg_pool2d("
            f"{i[0]}, {{{', '.join(str(v) for v in out_size.value)}}});"
        )


@NodeHandler.register("aten::flatten")
class FlattenHandler(NodeHandler):
    def emit_cpp(self, node, vm):
        i = node.inputs
        start = node.attributes.get("start_dim", IRAttribute("start_dim", 1, "int"))
        end   = node.attributes.get("end_dim",   IRAttribute("end_dim",  -1, "int"))
        return f"auto {node.outputs[0]} = torch::flatten({i[0]}, {start.value}, {end.value});"


@NodeHandler.register("aten::add", "aten::add_")
class AddHandler(NodeHandler):
    def emit_cpp(self, node, vm):
        i = node.inputs
        alpha = node.attributes.get("alpha", IRAttribute("alpha", 1, "int"))
        return f"auto {node.outputs[0]} = torch::add({i[0]}, {i[1]}, {alpha.value});"


@NodeHandler.register("aten::mul", "aten::mul_")
class MulHandler(NodeHandler):
    def emit_cpp(self, node, vm):
        i = node.inputs
        return f"auto {node.outputs[0]} = torch::mul({i[0]}, {i[1]});"


@NodeHandler.register("aten::matmul")
class MatMulHandler(NodeHandler):
    def emit_cpp(self, node, vm):
        i = node.inputs
        return f"auto {node.outputs[0]} = torch::matmul({i[0]}, {i[1]});"


@NodeHandler.register("aten::layer_norm")
class LayerNormHandler(NodeHandler):
    def emit_cpp(self, node, vm):
        i = node.inputs
        eps = node.attributes.get("eps", IRAttribute("eps", 1e-5, "float"))
        norm_shape = node.attributes.get("normalized_shape", IRAttribute("normalized_shape", [], "list"))
        return (
            f"auto {node.outputs[0]} = torch::layer_norm("
            f"{i[0]}, {{{', '.join(str(v) for v in norm_shape.value)}}}, "
            f"{i[2] if len(i)>2 else 'torch::Tensor()'}, "
            f"{i[3] if len(i)>3 else 'torch::Tensor()'}, "
            f"{eps.cpp_literal()});"
        )


@NodeHandler.register("aten::dropout")
class DropoutHandler(NodeHandler):
    """Inference-only: dropout is identity."""
    def infer_shapes(self, node, vm):
        inp = vm.get(node.inputs[0]) if node.inputs else None
        return [inp.shape if inp else None]

    def emit_cpp(self, node, vm):
        return f"auto {node.outputs[0]} = {node.inputs[0]};  // dropout → identity at inference"


@NodeHandler.register("aten::softmax")
class SoftmaxHandler(NodeHandler):
    def emit_cpp(self, node, vm):
        i = node.inputs
        dim  = node.attributes.get("dim", IRAttribute("dim", -1, "int"))
        return f"auto {node.outputs[0]} = torch::softmax({i[0]}, {dim.value});"


@NodeHandler.register("aten::transpose")
class TransposeHandler(NodeHandler):
    def emit_cpp(self, node, vm):
        i = node.inputs
        d0 = node.attributes.get("dim0", IRAttribute("dim0", 0, "int"))
        d1 = node.attributes.get("dim1", IRAttribute("dim1", 1, "int"))
        return f"auto {node.outputs[0]} = torch::transpose({i[0]}, {d0.value}, {d1.value});"


@NodeHandler.register("aten::view", "aten::reshape")
class ReshapeHandler(NodeHandler):
    def emit_cpp(self, node, vm):
        i = node.inputs
        shape_attr = node.attributes.get("shape")
        if shape_attr:
            dims = ", ".join(str(d) for d in shape_attr.value)
            return f"auto {node.outputs[0]} = {i[0]}.view({{{dims}}});"
        return f"auto {node.outputs[0]} = {i[0]}.view({i[1]});"


@NodeHandler.register("aten::cat")
class CatHandler(NodeHandler):
    def emit_cpp(self, node, vm):
        i = node.inputs
        dim = node.attributes.get("dim", IRAttribute("dim", 0, "int"))
        tensors = ", ".join(i)
        return f"auto {node.outputs[0]} = torch::cat({{{tensors}}}, {dim.value});"



class TorchScriptWalker:
    """
    Walks a live torch.jit.ScriptModule or loaded .pt file and populates
    an IRGraph.
    """

    def __init__(self, custom_op_registry: Optional[Dict[str, Any]] = None):
        self.custom_op_registry = custom_op_registry or {}
        self._node_counter = 0

    def _fresh_id(self) -> str:
        self._node_counter += 1
        return f"node_{self._node_counter:04d}"

    def _ts_type_to_ir(self, ts_type) -> IRType:
        s = str(ts_type)
        return _parse_ts_type(s)

    def _extract_shape(self, ts_type) -> Optional[IRTensorShape]:
        try:
            import torch
            if ts_type.kind() == "TensorType":
                sizes = ts_type.sizes()
                dtype = ts_type.scalarType()
                ir_dtype = IRType.FLOAT32
                if dtype is not None:
                    ir_dtype = _dtype_to_irtype(
                        getattr(torch, dtype.lower(), torch.float32)
                    )
                if sizes is not None:
                    return IRTensorShape(rank=len(sizes), dims=list(sizes), dtype=ir_dtype)
                return IRTensorShape(dtype=ir_dtype)
        except Exception:
            pass
        return None

    # Value extraction

    def _value_name(self, ts_value) -> str:
        dbg = ts_value.debugName()
        return dbg if dbg else f"v_{id(ts_value)}"

    def _ts_value_to_ir(self, ts_value, is_input=False, is_output=False) -> IRValue:
        name   = self._value_name(ts_value)
        t      = ts_value.type()
        ir_t   = self._ts_type_to_ir(t)
        shape  = self._extract_shape(t)
        return IRValue(
            name=name,
            ir_type=ir_t,
            shape=shape,
            is_input=is_input,
            is_output=is_output,
            debug_name=str(t),
        )

    # Attribute extraction

    def _extract_attributes(self, ts_node) -> Dict[str, IRAttribute]:
        attrs: Dict[str, IRAttribute] = {}
        try:
            for attr_name in ts_node.attributeNames():
                kind = ts_node.kindOf(attr_name)
                val: Any = None
                attr_type = "unknown"
                if kind == "f":
                    val = ts_node.f(attr_name); attr_type = "float"
                elif kind == "i":
                    val = ts_node.i(attr_name); attr_type = "int"
                elif kind == "s":
                    val = ts_node.s(attr_name); attr_type = "string"
                elif kind == "t":
                    attr_type = "tensor"
                    try:
                        t = ts_node.t(attr_name)
                        val = t.tolist()
                    except Exception:
                        val = None
                elif kind == "fs":
                    val = list(ts_node.fs(attr_name)); attr_type = "list"
                elif kind == "is":
                    val = list(ts_node.is_(attr_name)); attr_type = "list"
                elif kind == "ss":
                    val = list(ts_node.ss(attr_name)); attr_type = "list"
                elif kind == "ts":
                    attr_type = "list"; val = []
                elif kind == "g":
                    attr_type = "subgraph"; val = None
                elif kind == "gs":
                    attr_type = "subgraphs"; val = None
                else:
                    val = None

                if val is not None:
                    attrs[attr_name] = IRAttribute(attr_name, val, attr_type)
        except Exception as e:
            logger.debug("Attribute extraction error: %s", e)
        return attrs

    # Node walking

    def _is_custom(self, op_kind: str) -> bool:
        ns = op_kind.split("::")[0]
        return ns not in ("aten", "prim", "torchvision", "quantized")

    def _walk_block(self, block, value_map: Dict[str, IRValue],
                    nodes: List[IRNode], custom_ops: set) -> None:
        for ts_node in block.nodes():
            op_kind = ts_node.kind()
            node_id = self._fresh_id()

            # Recurse into sub-blocks (e.g. prim::Loop, prim::If)
            for sub_block in ts_node.blocks():
                self._walk_block(sub_block, value_map, nodes, custom_ops)

            # Register output values
            for out_val in ts_node.outputs():
                name = self._value_name(out_val)
                if name not in value_map:
                    value_map[name] = self._ts_value_to_ir(out_val)

            inputs  = [self._value_name(v) for v in ts_node.inputs()]
            outputs = [self._value_name(v) for v in ts_node.outputs()]
            attrs   = self._extract_attributes(ts_node)
            custom  = self._is_custom(op_kind)

            if custom:
                custom_ops.add(op_kind)

            source_loc = ""
            try:
                loc = ts_node.sourceRange()
                source_loc = str(loc)[:120] if loc else ""
            except Exception:
                pass

            ir_node = IRNode(
                node_id=node_id,
                op_kind=op_kind,
                inputs=inputs,
                outputs=outputs,
                attributes=attrs,
                is_custom=custom,
                source_location=source_loc,
            )
            nodes.append(ir_node)


    def walk(self, model_or_path: Any, model_name: str = "InferenceModel") -> IRGraph:
        import torch

        if isinstance(model_or_path, (str, Path)):
            model = torch.jit.load(str(model_or_path))
        elif isinstance(model_or_path, torch.nn.Module):
            model = torch.jit.script(model_or_path)
        else:
            model = model_or_path  # assume already ScriptModule

        graph = model.graph
        torch._C._jit_pass_inline(graph)
        torch._C._jit_pass_canonicalize(graph)

        value_map: Dict[str, IRValue] = {}
        nodes: List[IRNode] = []
        custom_ops: set = set()

        # Inputs
        inputs: List[IRValue] = []
        for inp in graph.inputs():
            name = self._value_name(inp)
            iv = self._ts_value_to_ir(inp, is_input=True)
            value_map[name] = iv
            inputs.append(iv)

        self._walk_block(graph, value_map, nodes, custom_ops)

        # Outputs — read directly from graph.outputs() (skipping the Return node itself)
        outputs: List[IRValue] = []
        for out_val in graph.outputs():
            name = self._value_name(out_val)
            if name in value_map:
                v = value_map[name]
                v.is_output = True
                outputs.append(v)
            else:
                iv = self._ts_value_to_ir(out_val, is_output=True)
                value_map[name] = iv
                outputs.append(iv)

        return IRGraph(
            name=model_name,
            inputs=inputs[1:],  # skip 'self' input
            outputs=outputs,
            nodes=nodes,
            values=value_map,
            custom_ops=sorted(custom_ops),
            metadata={"source": str(model_or_path) if isinstance(model_or_path, (str, Path)) else "live_module"},
        )



class IRSerializer:
    """JSON round-trip for IRGraph so the tool works without torch at load time."""

    @staticmethod
    def to_dict(graph: IRGraph) -> Dict[str, Any]:
        def val_dict(v: IRValue) -> Dict:
            d: Dict[str, Any] = {
                "name": v.name, "type": v.ir_type.value,
                "is_input": v.is_input, "is_output": v.is_output,
                "debug_name": v.debug_name,
            }
            if v.shape:
                d["shape"] = {"rank": v.shape.rank, "dims": v.shape.dims,
                              "dtype": v.shape.dtype.value}
            return d

        def attr_dict(a: IRAttribute) -> Dict:
            return {"name": a.name, "value": a.value, "type": a.attr_type}

        def node_dict(n: IRNode) -> Dict:
            return {
                "id": n.node_id, "op": n.op_kind,
                "inputs": n.inputs, "outputs": n.outputs,
                "is_custom": n.is_custom,
                "source": n.source_location,
                "attrs": {k: attr_dict(v) for k, v in n.attributes.items()},
            }

        return {
            "schema_version": "1.0",
            "name": graph.name,
            "inputs":  [val_dict(v) for v in graph.inputs],
            "outputs": [val_dict(v) for v in graph.outputs],
            "nodes":   [node_dict(n) for n in graph.nodes],
            "values":  {k: val_dict(v) for k, v in graph.values.items()},
            "custom_ops": graph.custom_ops,
            "metadata": graph.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> IRGraph:
        def mk_val(vd: Dict) -> IRValue:
            shape = None
            if "shape" in vd:
                sd = vd["shape"]
                dtype_map = {e.value: e for e in IRType}
                shape = IRTensorShape(
                    rank=sd.get("rank"),
                    dims=sd.get("dims"),
                    dtype=dtype_map.get(sd.get("dtype", "float"), IRType.FLOAT32),
                )
            type_map = {e.value: e for e in IRType}
            return IRValue(
                name=vd["name"],
                ir_type=type_map.get(vd.get("type", "/*unknown*/"), IRType.UNKNOWN),
                shape=shape,
                is_input=vd.get("is_input", False),
                is_output=vd.get("is_output", False),
                debug_name=vd.get("debug_name", ""),
            )

        def mk_attr(ad: Dict) -> IRAttribute:
            return IRAttribute(ad["name"], ad.get("value"), ad.get("type", "unknown"))

        def mk_node(nd: Dict) -> IRNode:
            return IRNode(
                node_id=nd["id"], op_kind=nd["op"],
                inputs=nd.get("inputs", []), outputs=nd.get("outputs", []),
                is_custom=nd.get("is_custom", False),
                source_location=nd.get("source", ""),
                attributes={k: mk_attr(v) for k, v in nd.get("attrs", {}).items()},
            )

        inputs  = [mk_val(v) for v in d.get("inputs", [])]
        outputs = [mk_val(v) for v in d.get("outputs", [])]
        nodes   = [mk_node(n) for n in d.get("nodes", [])]
        values  = {k: mk_val(v) for k, v in d.get("values", {}).items()}

        return IRGraph(
            name=d.get("name", "model"),
            inputs=inputs, outputs=outputs,
            nodes=nodes, values=values,
            custom_ops=d.get("custom_ops", []),
            metadata=d.get("metadata", {}),
        )

    @staticmethod
    def save(graph: IRGraph, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(IRSerializer.to_dict(graph), f, indent=2, default=str)

    @staticmethod
    def load(path: Union[str, Path]) -> IRGraph:
        with open(path) as f:
            return IRSerializer.from_dict(json.load(f))