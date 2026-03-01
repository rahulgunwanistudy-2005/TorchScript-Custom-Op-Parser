import torch
import torch.library as lib

_lib = lib.Library("mylib", "DEF")
_lib.define("mish(Tensor x) -> Tensor")

@torch.library.impl(_lib, "mish", "CPU")
def mish_cpu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(torch.nn.functional.softplus(x))

@torch.library.impl(_lib, "mish", "CUDA")
def mish_cuda(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(torch.nn.functional.softplus(x))

@torch.library.register_fake("mylib::mish")
def mish_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
