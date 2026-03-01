#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ir_parser import (
    IRGraph, IRNode, IRValue, IRType, IRTensorShape,
    IRAttribute, IRSerializer, _parse_ts_type, NodeHandler
)
from cpp_generator import (
    CppHeaderGenerator, GeneratorConfig, ShapeInferencePass,
    _collect_custom_ops, CustomOpSpec, _sanitize
)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m⊙\033[0m"
results = {"pass": 0, "fail": 0, "skip": 0}


def test(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results["pass"] += 1
    except AssertionError as e:
        print(f"  {FAIL} {name}: {e}")
        results["fail"] += 1
    except Exception as e:
        print(f"  {FAIL} {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["fail"] += 1


def skip(name, reason):
    print(f"  {SKIP} {name}: skipped ({reason})")
    results["skip"] += 1


def eq(a, b): assert a == b, f"{a!r} != {b!r}"
def contains(needle, haystack): assert needle in haystack, f"{needle!r} not in {haystack!r}"


def _v(name, ir_t, dims=None):
    return IRValue(
        name=name, ir_type=ir_t,
        shape=IRTensorShape(rank=len(dims), dims=dims) if dims else None
    )


def make_simple_graph():
    values = {
        "x":    _v("x",    IRType.TENSOR, [1, 3, 32, 32]),
        "w":    _v("w",    IRType.TENSOR, [16, 3, 3, 3]),
        "b":    _v("b",    IRType.TENSOR, [16]),
        "v1":   _v("v1",   IRType.TENSOR, [1, 16, 30, 30]),
        "v2":   _v("v2",   IRType.TENSOR, [1, 16, 30, 30]),
        "fc_w": _v("fc_w", IRType.TENSOR, [10, 16]),
        "fc_b": _v("fc_b", IRType.TENSOR, [10]),
        "out":  _v("out",  IRType.TENSOR, [1, 10]),
    }
    nodes = [
        IRNode("n1", "aten::conv2d", ["x", "w", "b"], ["v1"],
               attributes={
                   "stride":   IRAttribute("stride",   [1, 1], "list"),
                   "padding":  IRAttribute("padding",  [0, 0], "list"),
                   "dilation": IRAttribute("dilation", [1, 1], "list"),
                   "groups":   IRAttribute("groups",   1,      "int"),
               }),
        IRNode("n2", "aten::relu",   ["v1"], ["v2"]),
        IRNode("n3", "aten::linear", ["v2", "fc_w", "fc_b"], ["out"]),
    ]
    return IRGraph("TestModel", inputs=[values["x"]], outputs=[values["out"]],
                   nodes=nodes, values=values)


def make_custom_graph():
    values = {
        "x": _v("x", IRType.TENSOR, [1, 16, 32, 32]),
        "y": _v("y", IRType.TENSOR, [1, 16, 32, 32]),
    }
    nodes = [IRNode("n1", "mylib::mish", ["x"], ["y"], is_custom=True)]
    return IRGraph("MishModel", inputs=[values["x"]], outputs=[values["y"]],
                   nodes=nodes, values=values, custom_ops=["mylib::mish"])


def _gen(graph, **kwargs):
    return CppHeaderGenerator(GeneratorConfig(**kwargs)).generate(graph)


def run_all(tmp_dir: Path):
    print("\n══ 1. IRType parsing ══════════════════════════")
    test("Tensor",   lambda: eq(_parse_ts_type("Tensor"),         IRType.TENSOR))
    test("float",    lambda: eq(_parse_ts_type("float"),          IRType.FLOAT32))
    test("int",      lambda: eq(_parse_ts_type("int"),            IRType.INT64))
    test("List",     lambda: eq(_parse_ts_type("List[Tensor]"),   IRType.LIST))
    test("Optional", lambda: eq(_parse_ts_type("Optional[Tensor]"), IRType.OPTIONAL))
    test("unknown",  lambda: eq(_parse_ts_type("FutureMagicType"), IRType.UNKNOWN))

    print("\n══ 2. IRTensorShape ═══════════════════════════")
    def shape_concrete():
        s = IRTensorShape(rank=4, dims=[1, 3, 224, 224])
        c = s.cpp_shape_comment()
        assert "[1, 3, 224, 224]" in c and "float" in c

    def shape_symbolic():
        s = IRTensorShape(rank=4, dims=[None, 3, None, None])
        c = s.cpp_shape_comment()
        assert "?" in c and "3" in c

    test("concrete shape comment", shape_concrete)
    test("symbolic shape comment", shape_symbolic)

    print("\n══ 3. IRAttribute literals ════════════════════")
    test("float literal", lambda: contains("f",    IRAttribute("e", 1e-5,   "float").cpp_literal()))
    test("int literal",   lambda: contains("42",   IRAttribute("n", 42,     "int").cpp_literal()))
    test("string escape", lambda: contains('\\"',   IRAttribute("s", 'a"b',  "string").cpp_literal()))
    test("list literal",  lambda: contains("1, 2", IRAttribute("d", [1,2,3], "list").cpp_literal()))

    print("\n══ 4. IRGraph topological order ═══════════════")
    def topo_order():
        order = [n.op_kind for n in make_simple_graph().topological_order()]
        assert order.index("aten::conv2d") < order.index("aten::relu")
        assert order.index("aten::relu")   < order.index("aten::linear")

    test("conv2d → relu → linear order", topo_order)

    print("\n══ 5. IR serialisation ════════════════════════")
    def roundtrip_simple():
        g = make_simple_graph()
        p = tmp_dir / "rt.json"
        IRSerializer.save(g, p)
        g2 = IRSerializer.load(p)
        assert g2.name == g.name
        assert len(g2.nodes) == 3
        assert len(g2.inputs) == 1

    def roundtrip_custom():
        g = make_custom_graph()
        p = tmp_dir / "custom.json"
        IRSerializer.save(g, p)
        g2 = IRSerializer.load(p)
        assert g2.custom_ops == ["mylib::mish"]
        assert g2.nodes[0].is_custom

    def shapes_preserved():
        g = make_simple_graph()
        p = tmp_dir / "shapes.json"
        IRSerializer.save(g, p)
        g2 = IRSerializer.load(p)
        assert g2.values["x"].shape.dims == [1, 3, 32, 32]

    test("roundtrip simple",  roundtrip_simple)
    test("roundtrip custom",  roundtrip_custom)
    test("shapes preserved",  shapes_preserved)

    print("\n══ 6. Shape inference ════════════════════════")
    def relu_preserves_shape():
        g = make_simple_graph()
        ShapeInferencePass().run(g)
        assert g.values["v2"].shape.dims == [1, 16, 30, 30]

    def custom_op_passthrough():
        g = make_custom_graph()
        ShapeInferencePass().run(g)
        assert g.values["y"].shape is not None

    test("relu preserves shape",      relu_preserves_shape)
    test("custom op shape passthrough", custom_op_passthrough)

    print("\n══ 7. Custom op collection ════════════════════")
    def finds_custom():
        specs = _collect_custom_ops(make_custom_graph())
        assert "mylib::mish" in specs

    def appearance_count():
        g = make_custom_graph()
        g.nodes.append(IRNode("n2", "mylib::mish", ["y"], ["z"], is_custom=True))
        assert _collect_custom_ops(g)["mylib::mish"].appearances == 2

    def no_false_positives():
        assert len(_collect_custom_ops(make_simple_graph())) == 0

    def stub_correct():
        spec = CustomOpSpec("mylib::mish")
        spec.input_names  = [["x"]]
        spec.output_names = [["y"]]
        stub = spec.cpp_stub("    ")
        assert "mylib_mish" in stub and "at::Tensor" in stub and "TODO" in stub

    test("finds custom ops",     finds_custom)
    test("appearance counting",  appearance_count)
    test("no false positives",   no_false_positives)
    test("stub is valid C++",    stub_correct)

    print("\n══ 8. C++ generator ═══════════════════════════")
    def header_guard():
        h = _gen(make_simple_graph())
        assert "#pragma once" in h and "#ifndef" in h and "#endif" in h

    def namespace_wraps():
        h = _gen(make_simple_graph(), namespace="myns")
        assert "namespace myns {" in h and "}  // namespace myns" in h

    def infer_signature():
        h = _gen(make_simple_graph())
        assert "inline at::Tensor infer(" in h

    def ops_emitted():
        h = _gen(make_simple_graph())
        assert "conv2d" in h and "relu" in h and "linear" in h

    def custom_stub_gen():
        h = _gen(make_custom_graph())
        assert "mylib_mish" in h and "Custom op stub" in h

    def test_main_opt_in():
        h = _gen(make_simple_graph(), emit_test_main=True)
        assert "TEST_MAIN" in h and "int main()" in h

    def no_main_by_default():
        assert "int main()" not in _gen(make_simple_graph())

    def weight_loader():
        assert "load_weights" in _gen(make_simple_graph(), emit_weights=True)

    def braces_balance():
        h = _gen(make_simple_graph())
        assert h.count("{") == h.count("}")

    def multi_output_tuple():
        values = {
            "x":  _v("x",  IRType.TENSOR, [1, 8]),
            "y1": _v("y1", IRType.TENSOR, [1, 4]),
            "y2": _v("y2", IRType.TENSOR, [1, 4]),
        }
        g = IRGraph("Dual",
                    inputs=[values["x"]],
                    outputs=[values["y1"], values["y2"]],
                    nodes=[
                        IRNode("n1", "aten::relu", ["x"], ["y1"]),
                        IRNode("n2", "aten::relu", ["x"], ["y2"]),
                    ],
                    values=values)
        assert "std::tuple" in _gen(g)

    test("header guard",        header_guard)
    test("namespace wrapping",  namespace_wraps)
    test("infer() signature",   infer_signature)
    test("conv2d+relu+linear",  ops_emitted)
    test("custom stub",         custom_stub_gen)
    test("test main opt-in",    test_main_opt_in)
    test("no main by default",  no_main_by_default)
    test("weight loader",       weight_loader)
    test("braces balance",      braces_balance)
    test("multi-output tuple",  multi_output_tuple)

    print("\n══ 9. Node handlers ═══════════════════════════")
    def conv2d_emit():
        node = IRNode("n", "aten::conv2d", ["x", "w", "b"], ["out"],
                      attributes={
                          "stride":   IRAttribute("stride",   [1, 1], "list"),
                          "padding":  IRAttribute("padding",  [1, 1], "list"),
                          "dilation": IRAttribute("dilation", [1, 1], "list"),
                          "groups":   IRAttribute("groups",   1,      "int"),
                      })
        vm = {"x": _v("x", IRType.TENSOR, [1, 16, 32, 32]),
              "w": _v("w", IRType.TENSOR, [32, 16, 3, 3]),
              "b": _v("b", IRType.TENSOR, [32])}
        cpp = NodeHandler.get("aten::conv2d").emit_cpp(node, vm)
        assert "conv2d" in cpp and "out" in cpp

    def relu_emit():
        node = IRNode("n", "aten::relu", ["x"], ["y"])
        cpp = NodeHandler.get("aten::relu").emit_cpp(node, {})
        assert "relu" in cpp and "y" in cpp

    def dropout_is_identity():
        node = IRNode("n", "aten::dropout", ["x"], ["y"])
        cpp = NodeHandler.get("aten::dropout").emit_cpp(node, {})
        assert "x" in cpp

    def conv2d_shape():
        node = IRNode("n", "aten::conv2d", ["x", "w", "b"], ["out"],
                      attributes={
                          "stride":   IRAttribute("stride",   [1, 1], "list"),
                          "padding":  IRAttribute("padding",  [1, 1], "list"),
                          "dilation": IRAttribute("dilation", [1, 1], "list"),
                          "groups":   IRAttribute("groups",   1,      "int"),
                      })
        vm = {"x": _v("x", IRType.TENSOR, [1, 16, 32, 32]),
              "w": _v("w", IRType.TENSOR, [32, 16, 3, 3])}
        shapes = NodeHandler.get("aten::conv2d").infer_shapes(node, vm)
        assert shapes[0] is not None and shapes[0].rank == 4

    test("conv2d emit",         conv2d_emit)
    test("relu emit",           relu_emit)
    test("dropout is identity", dropout_is_identity)
    test("conv2d shape",        conv2d_shape)

    print("\n══ 10. _sanitize ══════════════════════════════")
    test("dots to underscores",    lambda: eq("foo_bar", _sanitize("foo.bar")))
    test("digit-leading prefixed", lambda: eq("_1abc",   _sanitize("1abc")))
    test("empty becomes non-empty",lambda: assert_true(len(_sanitize("")) > 0))

    print("\n══ 11. Live torch (if available) ══════════════")
    try:
        import torch
        import torch.nn as nn
        from ts_parser import parse_model, generate_header, save_ir

        def live_simple():
            class Net(nn.Module):
                def __init__(self): super().__init__(); self.fc = nn.Linear(4, 2)
                def forward(self, x: torch.Tensor) -> torch.Tensor: return self.fc(x)
            g = parse_model(torch.jit.script(Net().eval()), "SimpleNet")
            h = generate_header(g)
            assert "infer(" in h and "linear" in h

        def live_conv():
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
            g = parse_model(torch.jit.script(ConvNet().eval()), "ConvNet")
            h = generate_header(g, emit_test_main=True)
            assert all(op in h for op in ["conv2d", "relu", "linear", "TEST_MAIN"])

        def live_roundtrip():
            class Tiny(nn.Module):
                def __init__(self): super().__init__(); self.fc = nn.Linear(8, 4)
                def forward(self, x: torch.Tensor) -> torch.Tensor: return self.fc(x)
            g = parse_model(torch.jit.script(Tiny().eval()), "Tiny")
            p = tmp_dir / "tiny.json"
            save_ir(g, p)
            g2 = IRSerializer.load(p)
            assert g2.name == g.name and len(g2.nodes) > 0

        test("simple module end-to-end",  live_simple)
        test("conv+relu+linear pipeline", live_conv)
        test("IR JSON roundtrip",         live_roundtrip)

    except ImportError:
        skip("live torch tests", "torch not installed")


def assert_true(cond): assert cond


if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    print("TorchScript Custom Op Parser — Test Suite")
    print("==========================================")
    run_all(tmp)
    total = sum(results.values())
    print(f"\n{'─' * 44}")
    print(f"Results: {results['pass']} passed, {results['fail']} failed, {results['skip']} skipped / {total} total")
    if results["fail"]:
        sys.exit(1)