from pathlib import Path

import torch

from .model import (
    ASST_OPEN,
    MODEL_ID,
    SPECIAL_TOKENS,
    TARGET_LORA_MODULES,
    TOKENIZER_ID,
    USER_OPEN,
    Model,
    load_base,
    load_inference,
)


def post_load(model: Model, checkpoint_path: str | Path) -> None:
    state_path = Path(checkpoint_path) / "state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        total_tokens = state.get("total_tokens", 0.0)
        model.set_beta_anneal(total_tokens)
        print(f"beta anneal set for {total_tokens:.0f} training tokens")


__all__ = [
    "MODEL_ID",
    "TOKENIZER_ID",
    "TARGET_LORA_MODULES",
    "USER_OPEN",
    "ASST_OPEN",
    "SPECIAL_TOKENS",
    "Model",
    "load_base",
    "load_inference",
    "post_load",
]
