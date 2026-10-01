"""Experiment: shared"""

import math

import torch
import torch.nn as nn

# The single LoRA config warm-start checkpoints are written and read at.
# experiments/dreams/cli.py's --init-adapter treats a rank/alpha mismatch against the
# checkpoint's lora_config.json as fatal, so the warm start (training/loop.py's
# --lora-rank/--lora-alpha, pinned in the Makefile's warm-start target) has to
# use the same numbers or no checkpoint loads.
DEFAULT_RANK = 16
DEFAULT_ALPHA = 32.0
DEFAULT_DROPOUT = 0.0


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.linear = linear
        self.scale = alpha / rank
        dtype = linear.weight.dtype
        device = linear.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(rank, linear.in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(linear.out_features, rank, device=device, dtype=dtype)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        # mamba2's fused kernel accesses .weight directly; return the merged weight
        # so LoRA is applied even through the fused path and gradients flow correctly.
        return self.linear.weight + (self.lora_B @ self.lora_A) * self.scale

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scale


def apply_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> nn.Module:
    named = dict(model.named_modules())
    attached = 0
    for name, module in list(named.items()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in target_modules):
            continue
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if not parent_name else named[parent_name]
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
        attached += 1
    if attached == 0:
        raise RuntimeError(
            f"apply_lora matched 0 modules for targets {target_modules}. "
            "Mamba's projections may not be nn.Linear — inspect model.named_modules()."
        )
    print(f"attached {attached} LoRA adapters")
    return model


__all__ = [
    "DEFAULT_RANK",
    "DEFAULT_ALPHA",
    "DEFAULT_DROPOUT",
    "LoRALinear",
    "apply_lora",
]
