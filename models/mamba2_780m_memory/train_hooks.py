"""Training hooks for mamba2_780m_memory, called by sft/train.py's generic
training loop. Contrast with models/mamba2_780m/train_hooks.py: this model's
Model.forward(input_ids, state) is stateful, so sft/train.py processes
examples in --chunk-len chunks with `state` carried (and detached) across
chunks of the SAME example -- never across different examples -- bounding
training RAM by chunk length rather than example length. See
models/mamba2_780m_memory/README.md for why long examples aren't truncated at
data-prep time instead.

Unlike standard SFT (and unlike the other model's hooks), loss is computed
over EVERY token, not just assistant turns: prepare_data.py's mask marks user
turns as non-trainable, which is correct for short Q&A-style chat, but for
this model most of the content that's supposed to exercise long-range recall
is *in* the long user turns (a document, a long context) -- masking that out
would throw away most of the signal this model exists to learn from.
`mask_slice` is accepted by chunk_loss (for interface parity with the other
model's hook) but ignored.
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
# (8GB), chunk_len 8 is the largest value confirmed (via `make smoke-test
# --chunk-len N`, which now runs 5+ consecutive chunks rather than one
# isolated chunk -- a single chunk's peak VRAM is NOT representative of real
# multi-chunk training, since the held-over gradient/cache floor from
# earlier chunks eats into the next chunk's headroom) to run clean across 10
# consecutive chunks; chunk_len 9 OOMs by the second chunk. Gradient
# checkpointing on the per-token mixer step is the way to raise this further
# within a fixed VRAM budget; override with --chunk-len on a GPU with more
# VRAM.
DEFAULT_CHUNK_LEN = 8


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


def chunk_loss(
    model,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mask_slice: torch.Tensor | None,
    state,
    eos_weight: float,
):
    """mask_slice ignored (see module docstring -- trains on every token).
    Returns (loss_sum, weight_sum, state); the generic loop in
    sft/train.py owns chunking, accumulation, checkpointing, and live
    progress display across calls."""
    logits, state = model(input_ids, state=state)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="none")
    weight = torch.ones_like(loss)
    if eos_weight != 1.0:
        weight[target_ids.reshape(-1) == EOS_ID] = eos_weight
    return (loss * weight).sum(), weight.sum(), state


def on_step(model, total_tokens: float) -> None:
    """Optional hook -- train.py calls this (if defined) once before training
    starts and again after every optimizer step, passing cumulative training
    tokens. Used here to anneal beta's startup suppression away over the
    first BETA_BIAS_ANNEAL_TOKENS tokens (see model.py's
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
    for what beta/clear/surprise/o_t_norm/grad_norm mean. grad_norm is the
    raw (pre-soft-clip) per-token memory-write gradient norm -- see
    model.py's GRAD_SCALE for why it's worth watching: a healthy value
    should sit well below GRAD_SCALE most of the time, not against it."""
    stats = model.pop_memory_stats()
    if stats is None:
        return None
    return (
        f"memory  beta {stats['beta']:.4f}  clear {stats['clear']:.4f}"
        f"  surprise {stats['surprise']:.4f}  o_t_norm {stats['o_t_norm']:.4f}"
        f"  grad_norm {stats['grad_norm']:.1f}"
    )


def chunk_extra_log(model) -> str | None:
    """Optional hook -- train.py's generic per-chunk live progress display
    calls this (if defined) for an extra status line beyond the generic
    token/loss line. last_token_log is a live snapshot of exactly this
    chunk's last token (unlike extra_log's per-optimizer-step average from
    pop_memory_stats)."""
    log = model.last_token_log()
    if log is None:
        return None
    return (
        f"beta {log['beta']:.4f}  clear {log['clear']:.4f}"
        f"  active {log['active_layers']:>2}/{log['n_layers']}"
        f"  surprise {log['surprise']:.4f}  o_t_norm {log['o_t_norm']:.4f}"
        f"  grad_norm {log['grad_norm']:.1f}"
    )
