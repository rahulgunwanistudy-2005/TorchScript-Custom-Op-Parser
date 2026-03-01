from __future__ import annotations

import argparse
import json
import logging
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from ir_parser import (
    IRGraph, IRSerializer, TorchScriptWalker, IRValue, IRType, IRTensorShape
)
from cpp_generator import CppHeaderGenerator, GeneratorConfig, generate_cmake

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("ts_parser")


def load_ops_file(ops_path: str | Path) -> None:
    """
    Execute a Python file that registers custom TorchScript operators.

    The file is executed in its own namespace before the model is loaded,
    so any torch.library registrations it performs are visible to torch.jit.load.

    Example ops file (mymodel.ops.py):

        import torch, torch.library as lib

        _lib = lib.Library("mylib", "DEF")
        _lib.define("mish(Tensor x) -> Tensor")

        @torch.library.impl(_lib, "mish", "CPU")
        def mish_cpu(x): return x * torch.tanh(torch.nn.functional.softplus(x))

        @torch.library.register_fake("mylib::mish")
        def mish_fake(x): return torch.empty_like(x)
    """
    path = Path(ops_path)
    if not path.exists():
        raise FileNotFoundError(f"Ops file not found: {path}")
    # runpy keeps the module-level objects alive in the returned dict,
    # preventing Library objects from being garbage collected mid-session.
    ns = runpy.run_path(str(path))
    # Stash the namespace so it stays alive for the duration of the process.
    load_ops_file._namespaces.append(ns)
    logger.info("Loaded ops file: %s", path)

load_ops_file._namespaces: List[Dict] = []


def parse_model(
    model_or_path: Any,
    model_name: Optional[str] = None,
    custom_op_registry: Optional[Dict[str, Any]] = None,
    ops_file: Optional[str | Path] = None,
) -> IRGraph:
    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "PyTorch is required for parsing live models. "
            "Install it or load from a pre-saved IR JSON with load_ir()."
        )

    if ops_file is not None:
        load_ops_file(ops_file)

    name = model_name
    if name is None:
        if isinstance(model_or_path, (str, Path)):
            name = Path(model_or_path).stem
        else:
            name = type(model_or_path).__name__

    walker = TorchScriptWalker(custom_op_registry=custom_op_registry)
    graph = walker.walk(model_or_path, model_name=name)
    logger.info("Parsed '%s': %d nodes, %d custom ops.", graph.name, len(graph.nodes), len(graph.custom_ops))
    return graph


def load_ir(path: str | Path) -> IRGraph:
    graph = IRSerializer.load(path)
    logger.info("Loaded IR '%s' from %s", graph.name, path)
    return graph


def save_ir(graph: IRGraph, path: str | Path) -> None:
    IRSerializer.save(graph, path)
    logger.info("Saved IR to %s", path)


def generate_header(
    graph: IRGraph,
    *,
    namespace: str = "inference",
    emit_weights: bool = True,
    emit_test_main: bool = False,
    emit_custom_stubs: bool = True,
    shape_comments: bool = True,
    use_half_precision: bool = False,
    weight_dir: str = "weights",
) -> str:
    cfg = GeneratorConfig(
        namespace=namespace,
        emit_weights=emit_weights,
        emit_test_main=emit_test_main,
        emit_custom_stubs=emit_custom_stubs,
        shape_comments=shape_comments,
        use_half_precision=use_half_precision,
        weight_dir=weight_dir,
    )
    return CppHeaderGenerator(cfg).generate(graph)


def inspect_model(graph: IRGraph) -> Dict[str, Any]:
    op_counts: Dict[str, int] = {}
    for node in graph.nodes:
        op_counts[node.op_kind] = op_counts.get(node.op_kind, 0) + 1

    return {
        "name": graph.name,
        "num_inputs": len(graph.inputs),
        "num_outputs": len(graph.outputs),
        "num_nodes": len(graph.nodes),
        "num_custom_ops": len(graph.custom_ops),
        "custom_ops": graph.custom_ops,
        "op_histogram": dict(sorted(op_counts.items(), key=lambda x: -x[1])),
        "inputs": [
            {"name": v.name, "type": v.ir_type.value, "shape": v.shape.dims if v.shape else None}
            for v in graph.inputs
        ],
        "outputs": [
            {"name": v.name, "type": v.ir_type.value, "shape": v.shape.dims if v.shape else None}
            for v in graph.outputs
        ],
        "metadata": graph.metadata,
    }


def _cmd_parse(args: argparse.Namespace) -> None:
    graph = parse_model(args.model, model_name=args.name, ops_file=args.ops)
    if args.ir:
        save_ir(graph, args.ir)
    if args.output:
        header = generate_header(
            graph,
            namespace=args.namespace,
            emit_weights=not args.no_weights,
            emit_test_main=args.test_main,
            emit_custom_stubs=not args.no_stubs,
            shape_comments=not args.no_shapes,
            weight_dir=args.weight_dir,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(header)
        logger.info("Header written to %s", args.output)
        if args.cmake:
            cmake_text = generate_cmake(Path(args.model).stem, Path(args.output).name)
            cmake_path = Path(args.output).parent / "CMakeLists.txt"
            cmake_path.write_text(cmake_text)
            logger.info("CMakeLists.txt written to %s", cmake_path)


def _cmd_dump(args: argparse.Namespace) -> None:
    graph = parse_model(args.model, model_name=args.name, ops_file=args.ops)
    save_ir(graph, args.ir)


def _cmd_inspect(args: argparse.Namespace) -> None:
    if args.model:
        graph = parse_model(args.model, model_name=args.name, ops_file=getattr(args, "ops", None))
    else:
        graph = load_ir(args.ir)
    print(json.dumps(inspect_model(graph), indent=2))


def _cmd_codegen(args: argparse.Namespace) -> None:
    graph = load_ir(args.ir)
    header = generate_header(
        graph,
        namespace=args.namespace,
        emit_weights=not args.no_weights,
        emit_test_main=args.test_main,
        emit_custom_stubs=not args.no_stubs,
        shape_comments=not args.no_shapes,
        weight_dir=args.weight_dir,
    )
    if args.output == "-":
        print(header)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(header)
        logger.info("Header written to %s", args.output)
    if args.cmake:
        cmake_text = generate_cmake(Path(args.ir).stem, Path(args.output).name)
        cmake_path = Path(args.output).parent / "CMakeLists.txt"
        cmake_path.write_text(cmake_text)
        logger.info("CMakeLists.txt written to %s", cmake_path)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--namespace",  default="inference", help="C++ namespace (default: inference)")
    p.add_argument("--no-weights", action="store_true",  help="Omit weight loading code")
    p.add_argument("--test-main",  action="store_true",  help="Emit TEST_MAIN harness")
    p.add_argument("--no-stubs",   action="store_true",  help="Omit custom op stubs")
    p.add_argument("--no-shapes",  action="store_true",  help="Omit shape annotations")
    p.add_argument("--weight-dir", default="weights",    help="Weight files directory (default: weights)")
    p.add_argument("--cmake",      action="store_true",  help="Also generate CMakeLists.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ts-parser",
        description="TorchScript Custom Op Parser — generates C++ inference headers",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="Parse a .pt file and emit C++ header")
    p_parse.add_argument("--model",  required=True, help="Path to .pt ScriptModule")
    p_parse.add_argument("--output", required=True, help="Output .h header path")
    p_parse.add_argument("--ops",    default=None,  help="Python file that registers custom ops")
    p_parse.add_argument("--ir",     default=None,  help="Also save IR JSON here")
    p_parse.add_argument("--name",   default=None,  help="Override model name")
    _add_common_args(p_parse)

    p_dump = sub.add_parser("dump", help="Dump IR to JSON (no code generation)")
    p_dump.add_argument("--model", required=True, help="Path to .pt file")
    p_dump.add_argument("--ir",    required=True, help="Output JSON path")
    p_dump.add_argument("--ops",   default=None,  help="Python file that registers custom ops")
    p_dump.add_argument("--name",  default=None,  help="Override model name")

    p_cg = sub.add_parser("codegen", help="Generate C++ from a saved IR JSON")
    p_cg.add_argument("--ir",     required=True, help="Input IR JSON path")
    p_cg.add_argument("--output", default="-",   help="Output .h path (- for stdout)")
    _add_common_args(p_cg)

    p_ins = sub.add_parser("inspect", help="Print graph summary as JSON")
    grp = p_ins.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model", default=None, help="Path to .pt file")
    grp.add_argument("--ir",    default=None, help="Path to IR JSON")
    p_ins.add_argument("--ops",  default=None, help="Python file that registers custom ops")
    p_ins.add_argument("--name", default=None)

    args = parser.parse_args()
    {"parse": _cmd_parse, "dump": _cmd_dump, "codegen": _cmd_codegen, "inspect": _cmd_inspect}[args.command](args)


if __name__ == "__main__":
    main()
