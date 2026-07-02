"""Training hooks for mamba2_2_7b_memory, called by sft/train.py's generic
training loop. Contrast with models/mamba2_780m/train_hooks.py: this model's
Model.forward(input_ids, state) is stateful, so sft/train.py processes
examples in --chunk-len chunks with `state` carried (and detached) across
chunks of the SAME example -- never across different examples -- bounding
training RAM by chunk length rather than example length. See
models/mamba2_2_7b_memory/README.md for why long examples aren't truncated at
data-prep time instead.

Unlike standard SFT (and unlike the other model's hooks), loss is computed
over EVERY non-padded token: prepare_data.py's mask marks user turns as
non-trainable, which is correct for short Q&A-style chat, but for this model
most of the content that's supposed to exercise long-range recall is *in* the
long user turns (a document, a long context) -- masking that out would throw
away most of the signal this model exists to learn from. mask_slice is used
only as a padding mask (True = real token, False = padded position) -- it
does NOT restrict loss to assistant turns.
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
# graph whose VRAM cost scales with chunk_len -- the cost per token is larger
# than the 780m variant (MEM_HIDDEN scales from 6144 to 10240, so the
# fast-weight snapshot per token is ~210 MB vs ~75 MB). DEFAULT_CHUNK_LEN=12
# is tuned for batch_size=4 on a cloud H100 (80 GB). Benchmark with
# `make smoke-test --chunk-len N` across 10+ consecutive chunks before
# raising it -- a single isolated chunk's peak VRAM is NOT representative of
# real multi-chunk training, since the held-over gradient/cache floor from
# earlier chunks eats into the next chunk's headroom.
DEFAULT_CHUNK_LEN = 7


def setup_training(device, lora_rank: int, lora_alpha: float, lora_dropout: float):
    """Loads the backbone, attaches LoRA, then wraps in Model -- training
    operates on the full memory-augmented wrapper, not the raw backbone, since
    the memory subsystem (front_end, injections) only exists on Model.
    Model.__init__ already freezes everything except lora_A/lora_B and the
    memory subsystem -- nothing extra to freeze here. Returns (model,
    trainable_params)."""
    base = _model_mod.load_base(str(device))
    base = apply_lora(base, _model_mod.TARGET_LORA_MODULES, lora_rank, lora_alpha, lora_dropout)
    model = _model_mod.Model(base).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable_params):,}")
    return model, trainable_params


def init_state(model, batch_size: int, device):
    """Create a fresh batched MemoryState for batch_size parallel slots."""
    dtype = model.embedding.weight.dtype
    return model._init_state(batch_size, device, dtype)


def chunk_loss(
    model,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mask_slice: torch.Tensor | None,
    state,
    eos_weight: float,
):
    """mask_slice is a (B, T) bool tensor: True = real token, False = padding.
    Trains on every real token (no user/assistant distinction -- see module
    docstring). Returns (loss_sum, weight_sum, state); the generic loop in
    sft/train.py owns chunking, accumulation, checkpointing, and live
    progress display across calls."""
    logits, state = model(input_ids, state=state)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="none")
    weight = torch.ones_like(loss)
    if eos_weight != 1.0:
        weight[target_ids.reshape(-1) == EOS_ID] = eos_weight
    if mask_slice is not None:
        weight = weight * mask_slice.reshape(-1).float()
    return (loss * weight).sum(), weight.sum(), state


def reset_slot(model, state, slot_idx: int) -> None:
    """Reset slot slot_idx's MemoryState to fresh random init, leaving all
    other slots unchanged. Call only on a detached state."""
    model.reset_slot(state, slot_idx)


def on_step(model, total_tokens: float) -> None:
    """Optional hook -- train.py calls this (if defined) once before training
    starts and again after every optimizer step, passing cumulative training
    tokens. Used here to anneal beta's startup suppression away over the
    first BETA_BIAS_ANNEAL_TOKENS tokens of training (see model.py's
    BETA_BIAS_ANNEAL_START/_TOKENS and Model.set_beta_anneal) -- keyed on
    total_tokens rather than optimizer steps so it lines up with this
    project's other token-denominated knobs (e.g. --ckpt-every-tokens) and
    resumes correctly without extra checkpoint state."""
    model.set_beta_anneal(total_tokens)


def extra_log(model) -> str | None:
    """Optional hook -- train.py calls this (if defined) after each optimizer
    step and prints whatever string it returns (None to print nothing this
    step). Used here to report whether the memory subsystem is actually
    being used, not just receiving gradient -- see Model.pop_memory_stats
    for what beta/retain/surprise/o_t_norm/grad_norm mean. grad_norm is the
    raw (pre-soft-clip) per-token memory-write gradient norm -- see
    model.py's GRAD_SCALE for why it's worth watching: a healthy value
    should sit well below GRAD_SCALE most of the time, not against it."""
    stats = model.pop_memory_stats()
    if stats is None:
        return None
    return (
        f"memory  beta {stats['beta']:.4f}  retain {stats['retain']:.4f}"
        f"  surprise {stats['surprise']:.4f}  o_t_norm {stats['o_t_norm']:.4f}"
        f"  grad_norm {stats['grad_norm']:.4g}"
    )


def chunk_extra_log(model) -> list[str] | None:
    """Optional hook -- train.py's generic per-chunk live progress display
    calls this (if defined) for extra per-slot status lines. Returns one
    string per batch slot (last_token_log is a live snapshot of each slot's
    last token). None if forward() hasn't run yet."""
    logs = model.last_token_log()
    if logs is None:
        return None
    return [
        f"beta {log['beta']:.4f}  retain {log['retain']:.4f}"
        f"  active {log['active_layers']:>2}/{log['n_layers']}  min_cos_sim {log['min_cos_sim']:.4f}"
        f"  surprise {log['surprise']:.4f}  o_t_norm {log['o_t_norm']:.4f}"
        f"  grad_norm {log['grad_norm']:.4g}"
        for log in logs
    ]
