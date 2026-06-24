"""Training hooks for mamba2_780m, called by sft/train.py's generic training
loop.

Model.forward loops over tokens and threads a MixerState across calls (see
model.py for why -- mamba_ssm's own fused kernels are broken on this
hardware), so training here chunks long examples the same way
models/mamba2_780m_memory/train_hooks.py does: state is carried and detached
across chunks of the *same* example, never across different examples, so
training RAM is bounded by chunk length rather than example length.

sft/train.py owns everything that's the same across models: shuffling,
gradient-accumulation counting, checkpoint cadence/rotation (including
mid-example resume), non-finite checks, chunk iteration itself, evaluation,
and preflight. What's irreducibly model-specific -- how to load/wrap the
model for training, and how to compute loss for one chunk -- lives here.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sft"))
from lora import apply_lora

from . import model as _model_mod

EOS_ID = 0  # <|endoftext|> for EleutherAI/gpt-neox-20b
DEFAULT_CHUNK_LEN = 512


def setup_training(device, lora_rank: int, lora_alpha: float, lora_dropout: float):
    """Loads the base backbone with LoRA attached, then wraps it in the
    inference Model (needed now since training goes through Model.forward's
    chunked/state-threaded path, not load_base()'s raw output directly).
    Returns (model, trainable_params)."""
    base = _model_mod.load_base(str(device))
    base = base.to(torch.float32)
    base = apply_lora(base, _model_mod.TARGET_LORA_MODULES, lora_rank, lora_alpha, lora_dropout)
    model = _model_mod.Model(base).to(device)

    for param in model.parameters():
        param.requires_grad_(False)
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
            trainable_params.append(param)
    print(f"trainable params: {sum(p.numel() for p in trainable_params):,}")
    return model, trainable_params


def chunk_loss(
    model,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mask_slice: torch.Tensor,
    state,
    eos_weight: float,
):
    """input_ids/target_ids: shape (1, T). mask_slice: shape (T,), bool/0-1,
    1 for assistant-turn positions, 0 otherwise -- unlike mamba2_780m_memory,
    this model has no reason to train on user turns too. Returns (loss_sum,
    weight_sum, state); the generic loop in sft/train.py owns chunking,
    accumulation, and checkpointing across calls."""
    logits, state = model(input_ids, state=state)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="none")
    loss_mask = mask_slice.float().clone()
    if eos_weight != 1.0:
        eos_positions = (target_ids.reshape(-1) == EOS_ID) & (loss_mask > 0)
        loss_mask[eos_positions] = eos_weight
    return (loss * loss_mask).sum(), loss_mask.sum(), state
