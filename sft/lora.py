import math
from pathlib import Path

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.linear = linear
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        # mamba2's fused kernel accesses .weight directly; return the merged weight
        # so LoRA is applied even through the fused path and gradients flow correctly
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


def save_lora(model: nn.Module, path: str | Path, rank: int, alpha: float) -> None:
    import json
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}
    torch.save(state, path / "adapter.pt")
    (path / "lora_config.json").write_text(json.dumps({"rank": rank, "alpha": alpha}))


def load_lora(model: nn.Module, path: str | Path) -> nn.Module:
    state = torch.load(Path(path) / "adapter.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.unexpected_keys)
    print(f"loaded {loaded}/{len(state)} adapter tensors")
    if loaded == 0:
        raise RuntimeError("load_lora loaded 0 tensors — checkpoint keys don't match model structure")
    return model
