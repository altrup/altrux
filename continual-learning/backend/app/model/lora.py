import math
from pathlib import Path

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight + (self.lora_B @ self.lora_A) * self.scale

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + x @ self.lora_A.T @ self.lora_B.T * self.scale


def apply_lora(model: nn.Module, target_modules: list[str], rank: int, alpha: float) -> nn.Module:
    named = dict(model.named_modules())
    attached = 0
    for name, module in list(named.items()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(t) for t in target_modules):
            continue
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if not parent_name else named[parent_name]
        setattr(parent, child_name, LoRALinear(module, rank, alpha))
        attached += 1
    if attached == 0:
        raise RuntimeError(f"apply_lora matched 0 modules for targets {target_modules}")
    print(f"attached {attached} LoRA adapters (rank={rank}, alpha={alpha})")
    return model


def read_lora_config(checkpoint_path: str | Path) -> tuple[int, float]:
    """Read rank and alpha saved by the SFT trainer. Falls back to defaults if absent."""
    import json
    cfg_path = Path(checkpoint_path) / "lora_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        return int(cfg["rank"]), float(cfg["alpha"])
    return 16, 32.0  # match sft/train.py defaults


def load_lora(model: nn.Module, checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path)
    state = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.unexpected_keys)
    print(f"loaded {loaded}/{len(state)} LoRA adapter tensors from {path}")
    if loaded == 0:
        raise RuntimeError("load_lora loaded 0 tensors — checkpoint keys don't match model structure")
