from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("example")

_LIBS = []


def register_custom_ops():
    try:
        import torch
        import torch.library as lib

        _LIBS.clear()
        my_lib = lib.Library("mylib", "DEF")
        _LIBS.append(my_lib)

        my_lib.define("mish(Tensor x) -> Tensor")

        @torch.library.impl(my_lib, "mish", "CPU")
        def mish_cpu(x: torch.Tensor) -> torch.Tensor:
            return x * torch.tanh(torch.nn.functional.softplus(x))

        @torch.library.impl(my_lib, "mish", "CUDA")
        def mish_cuda(x: torch.Tensor) -> torch.Tensor:
            return x * torch.tanh(torch.nn.functional.softplus(x))

        @torch.library.register_fake("mylib::mish")
        def mish_abstract(x: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(x)

        my_lib.define(
            "dw_sep_conv(Tensor x, Tensor dw_weight, Tensor pw_weight, int groups) -> Tensor"
        )

        @torch.library.impl(my_lib, "dw_sep_conv", "CPU")
        def dw_sep_conv_cpu(x, dw_weight, pw_weight, groups):
            x = torch.nn.functional.conv2d(x, dw_weight, groups=groups, padding=1)
            x = torch.nn.functional.conv2d(x, pw_weight)
            return x

        @torch.library.register_fake("mylib::dw_sep_conv")
        def dw_sep_conv_abstract(x, dw_weight, pw_weight, groups):
            N, _, H, W = x.shape
            out_C = pw_weight.shape[0]
            return x.new_empty(N, out_C, H, W)

        logger.info("Custom ops registered: mylib::mish, mylib::dw_sep_conv")
        return True
    except Exception as e:
        logger.warning("Could not register custom ops: %s", e)
        return False


def build_resnet_like():
    try:
        import torch
        import torch.nn as nn

        class ResBlock(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
                self.bn1   = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
                self.bn2   = nn.BatchNorm2d(channels)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                residual = x
                x = torch.relu(self.bn1(self.conv1(x)))
                x = self.bn2(self.conv2(x))
                return torch.relu(x + residual)

        class TinyResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem   = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                )
                self.block1 = ResBlock(32)
                self.block2 = ResBlock(32)
                self.pool   = nn.AdaptiveAvgPool2d(1)
                self.fc     = nn.Linear(32, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.relu(self.stem(x))
                x = self.block1(x)
                x = self.block2(x)
                x = self.pool(x)
                x = torch.flatten(x, 1)
                return self.fc(x)

        return torch.jit.script(TinyResNet().eval())
    except Exception as e:
        logger.warning("Could not build ResNet model: %s", e)
        return None


def build_custom_op_model():
    try:
        import torch
        import torch.nn as nn

        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
                self.bn   = nn.BatchNorm2d(16)
                self.dw_w = nn.Parameter(torch.randn(16, 1, 3, 3))
                self.pw_w = nn.Parameter(torch.randn(32, 16, 1, 1))
                self.fc   = nn.Linear(32, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv(x)
                x = self.bn(x)
                x = torch.ops.mylib.mish(x)
                x = torch.ops.mylib.dw_sep_conv(x, self.dw_w, self.pw_w, 16)
                x = torch.ops.mylib.mish(x)
                x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
                x = torch.flatten(x, 1)
                return self.fc(x)

        return torch.jit.script(CustomModel().eval())
    except Exception as e:
        logger.warning("Could not build CustomModel: %s", e)
        return None


def build_transformer_block():
    try:
        import torch
        import torch.nn as nn

        class TransformerBlock(nn.Module):
            def __init__(self, d_model: int = 64, nhead: int = 4):
                super().__init__()
                self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
                self.norm1 = nn.LayerNorm(d_model)
                self.ff1   = nn.Linear(d_model, d_model * 4)
                self.ff2   = nn.Linear(d_model * 4, d_model)
                self.norm2 = nn.LayerNorm(d_model)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                attn_out, _ = self.attn(x, x, x)
                x = self.norm1(x + attn_out)
                ff = self.ff2(torch.relu(self.ff1(x)))
                return self.norm2(x + ff)

        return torch.jit.script(TransformerBlock().eval())
    except Exception as e:
        logger.warning("Could not build TransformerBlock: %s", e)
        return None


def run_pipeline(scripted_model, name: str, output_dir: Path) -> bool:
    from ts_parser import parse_model, save_ir, generate_header, inspect_model

    try:
        graph = parse_model(scripted_model, model_name=name)
    except Exception as e:
        logger.error("parse_model failed for %s: %s", name, e)
        return False

    ir_path = output_dir / f"{name}_ir.json"
    save_ir(graph, ir_path)
    logger.info("  IR JSON -> %s", ir_path)

    summary = inspect_model(graph)
    logger.info(
        "  %s: %d nodes, %d custom ops, op-types: %s",
        name,
        summary["num_nodes"],
        summary["num_custom_ops"],
        list(summary["op_histogram"].keys())[:5],
    )
    if summary["custom_ops"]:
        logger.info("  Custom ops: %s", summary["custom_ops"])

    header = generate_header(
        graph,
        namespace=f"inference::{name}",
        emit_weights=True,
        emit_test_main=True,
        emit_custom_stubs=True,
        shape_comments=True,
    )
    h_path = output_dir / f"{name}_inference.h"
    h_path.write_text(header)
    logger.info("  C++ header -> %s  (%d lines)", h_path, header.count("\n"))
    return True


def run_synthetic_demo(output_dir: Path) -> None:
    from ir_parser import (
        IRGraph, IRNode, IRValue, IRType, IRTensorShape,
        IRAttribute, IRSerializer
    )
    from ts_parser import generate_header

    logger.info("Running synthetic IR demo (no torch required)...")

    def val(name, ir_t, dims=None):
        return IRValue(
            name=name, ir_type=ir_t,
            shape=IRTensorShape(rank=len(dims), dims=dims) if dims else None
        )

    values = {
        "x":        val("x",        IRType.TENSOR, [1, 3, 224, 224]),
        "w1":       val("w1",       IRType.TENSOR, [64, 3, 7, 7]),
        "b1":       val("b1",       IRType.TENSOR, [64]),
        "bn_w":     val("bn_w",     IRType.TENSOR, [64]),
        "bn_b":     val("bn_b",     IRType.TENSOR, [64]),
        "bn_m":     val("bn_m",     IRType.TENSOR, [64]),
        "bn_v":     val("bn_v",     IRType.TENSOR, [64]),
        "w2":       val("w2",       IRType.TENSOR, [10, 64]),
        "b2":       val("b2",       IRType.TENSOR, [10]),
        "v1":       val("v1",       IRType.TENSOR, [1, 64, 112, 112]),
        "v2":       val("v2",       IRType.TENSOR, [1, 64, 112, 112]),
        "v3":       val("v3",       IRType.TENSOR, [1, 64, 1, 1]),
        "v4":       val("v4",       IRType.TENSOR, [1, 64]),
        "v5":       val("v5",       IRType.TENSOR, [1, 10]),
        "v_custom": val("v_custom", IRType.TENSOR, [1, 64, 112, 112]),
    }

    nodes = [
        IRNode(
            node_id="node_0001", op_kind="aten::conv2d",
            inputs=["x", "w1", "b1"], outputs=["v1"],
            attributes={
                "stride":   IRAttribute("stride",   [2, 2], "list"),
                "padding":  IRAttribute("padding",  [3, 3], "list"),
                "dilation": IRAttribute("dilation", [1, 1], "list"),
                "groups":   IRAttribute("groups",   1,      "int"),
            },
        ),
        IRNode(
            node_id="node_0002", op_kind="aten::batch_norm",
            inputs=["v1", "bn_w", "bn_b", "bn_m", "bn_v"], outputs=["v2"],
            attributes={
                "training": IRAttribute("training", False, "bool"),
                "momentum": IRAttribute("momentum", 0.1,   "float"),
                "eps":      IRAttribute("eps",      1e-5,  "float"),
            },
        ),
        IRNode(
            node_id="node_0003", op_kind="mylib::mish",
            inputs=["v2"], outputs=["v_custom"],
            is_custom=True,
        ),
        IRNode(
            node_id="node_0004", op_kind="aten::adaptive_avg_pool2d",
            inputs=["v_custom"], outputs=["v3"],
            attributes={"output_size": IRAttribute("output_size", [1, 1], "list")},
        ),
        IRNode(
            node_id="node_0005", op_kind="aten::flatten",
            inputs=["v3"], outputs=["v4"],
            attributes={
                "start_dim": IRAttribute("start_dim", 1,  "int"),
                "end_dim":   IRAttribute("end_dim",  -1,  "int"),
            },
        ),
        IRNode(
            node_id="node_0006", op_kind="aten::linear",
            inputs=["v4", "w2", "b2"], outputs=["v5"],
        ),
    ]

    graph = IRGraph(
        name="SyntheticCNN",
        inputs=[values["x"]],
        outputs=[values["v5"]],
        nodes=nodes,
        values=values,
        custom_ops=["mylib::mish"],
        metadata={"source": "synthetic", "description": "hand-built IR demo"},
    )

    ir_path = output_dir / "synthetic_ir.json"
    IRSerializer.save(graph, ir_path)
    logger.info("  Synthetic IR -> %s", ir_path)

    header = generate_header(
        graph,
        namespace="inference::synthetic",
        emit_test_main=True,
        emit_custom_stubs=True,
        shape_comments=True,
    )
    h_path = output_dir / "synthetic_inference.h"
    h_path.write_text(header)
    logger.info("  Synthetic C++ header -> %s  (%d lines)", h_path, header.count("\n"))


def main():
    output_dir = Path(__file__).parent.parent / "generated"
    output_dir.mkdir(exist_ok=True)

    run_synthetic_demo(output_dir)

    try:
        import torch
        logger.info("PyTorch %s detected -- running live model demos", torch.__version__)
        register_custom_ops()

        models = {
            "tiny_resnet": build_resnet_like,
            "custom_ops":  build_custom_op_model,
            "transformer": build_transformer_block,
        }
        for name, builder in models.items():
            logger.info("Processing %s ...", name)
            m = builder()
            if m is not None:
                run_pipeline(m, name, output_dir)
    except ImportError:
        logger.info("PyTorch not installed -- skipping live model demos")

    logger.info("\nDone. Generated files in: %s", output_dir)
    for f in sorted(output_dir.iterdir()):
        logger.info("  %s  (%d bytes)", f.name, f.stat().st_size)


if __name__ == "__main__":
    main()