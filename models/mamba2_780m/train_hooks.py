"""Training hooks for mamba2_780m, called by sft/train.py's generic training
loop.

Model.forward loops over tokens and threads a MixerState across calls (see
model.py for why -- mamba_ssm's own fused kernels are broken on this
hardware), so training here chunks long examples: state is carried and
detached across chunks of the *same* example, never across different
examples, so training RAM is bounded by chunk length rather than example
length.

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
# This model's manual, unfused, per-token mixer step holds a live backward
# graph whose VRAM cost scales with chunk_len -- on this project's dev GPU
# (8GB, bf16 -- see load_base), chunk_len 52 is the largest value confirmed
# (via `make smoke-test --chunk-len N`, which runs 5+ consecutive chunks
# rather than one isolated chunk -- a single chunk's peak VRAM is NOT
# representative of real multi-chunk training, since the held-over
# gradient/cache floor from earlier chunks eats into the next chunk's
# headroom) to run clean; chunk_len 54 OOMs by the second chunk. 48 leaves a
# small margin below that for real-data variance. Override with --chunk-len
# on a GPU with more VRAM.
DEFAULT_CHUNK_LEN = 48


def setup_training(device, lora_rank: int, lora_alpha: float, lora_dropout: float):
    """Loads the base backbone with LoRA attached, then wraps it in the
    inference Model (needed now since training goes through Model.forward's
    chunked/state-threaded path, not load_base()'s raw output directly).
    Returns (model, trainable_params)."""
    base = _model_mod.load_base(str(device))
    base = apply_lora(base, _model_mod.TARGET_LORA_MODULES, lora_rank, lora_alpha, lora_dropout)
    model = _model_mod.Model(base).to(device)

    for param in model.parameters():
        param.requires_grad_(False)
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "marker_delta" in name:
            param.requires_grad_(True)
            trainable_params.append(param)
    print(f"trainable params: {sum(p.numel() for p in trainable_params):,}")
    return model, trainable_params


def set_grad_checkpoint(model, enabled: bool, block: int | None = None) -> None:
    """Optional hook -- train.py calls this (if defined) at the start of every
    config-group segment with that slice's `grad_checkpoint` setting, so a
    long-chunk slice can pay the recompute tax while the short-chunk slices in
    the same run don't. See Model.set_grad_checkpoint."""
    if block is None:
        model.set_grad_checkpoint(enabled)
    else:
        model.set_grad_checkpoint(enabled, block)


def chunk_loss(
    model,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mask_slice: torch.Tensor,
    state,
    eos_weight: float,
):
    """input_ids/target_ids: shape (B, T). mask_slice: shape (B, T) (or (T,) for
    a single sequence), the per-position weight -- nonzero for assistant-turn
    positions, 0 otherwise -- unlike mamba2_2_7b_memory, this model has no
    reason to train on user turns too. Returns (loss_sum,
    weight_sum, state); the generic loop in sft/train.py owns chunking,
    accumulation, and checkpointing across calls.

    A chunk with no assistant-turn tokens (mask_slice all zero -- the common
    case for a conversation's opening chunk(s), which are pure user-turn
    content) gets weight_sum == 0, so the caller never calls .backward() on
    it -- but without that, the chunk's forward graph (every token's
    activations across all 48 layers) was never freed at all, just left for
    Python's cyclic GC to eventually catch -- which doesn't happen before the
    *next* chunk's forward needs that VRAM (confirmed: this OOMs real
    conversations well before chunk_len's own VRAM ceiling is ever reached,
    since every conversation opens with user-turn content). Running the
    no-trainable-tokens case under no_grad avoids building that graph at all
    -- state still advances correctly, it's detached again by the caller
    regardless of which branch produced it."""
    if not mask_slice.any():
        with torch.no_grad():
            _, state = model(input_ids, state=state)
        zero = torch.zeros((), device=input_ids.device)
        return zero, zero, state
    logits, state = model(input_ids, state=state)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="none")
    loss_mask = mask_slice.reshape(-1).float().clone()
    if eos_weight != 1.0:
        eos_positions = (target_ids.reshape(-1) == EOS_ID) & (loss_mask > 0)
        loss_mask[eos_positions] = eos_weight
    return (loss * loss_mask).sum(), loss_mask.sum(), state
