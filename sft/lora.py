import math

import torch
import torch.nn as nn


# The single LoRA config warm-start checkpoints are written and read at.
# dream_sleep.py's --init-adapter treats a rank/alpha mismatch against the
# checkpoint's lora_config.json as fatal, so the warm start (train.py's
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
        # linear.weight.dtype is the packed storage dtype (e.g. uint8) when
        # `linear` is a quantized bitsandbytes Linear4bit (see
        # models.common.quantize_lora_targets for QLoRA) -- compute_dtype is
        # the dtype it actually computes/dequantizes in, and is what the
        # adapters need to match. Plain nn.Linear has no compute_dtype, so
        # this falls back to its (correct) weight dtype.
        dtype = getattr(linear, "compute_dtype", None) or linear.weight.dtype
        device = linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank, device=device, dtype=dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        # mamba2's fused kernel accesses .weight directly; return the merged weight
        # so LoRA is applied even through the fused path and gradients flow correctly.
        # When `linear` is a quantized Linear4bit (detected the same way __init__
        # does, via compute_dtype), its .weight is packed 4-bit storage -- not
        # addable to lora_B @ lora_A -- so skip the merge. Harmless for QLoRA
        # models: they never reach the fused-kernel path (see this class's
        # historical note), the only other caller is mamba_ssm's
        # allocate_inference_cache, which just reads .weight.device.
        if getattr(self.linear, "compute_dtype", None) is not None:
            return self.linear.weight
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
