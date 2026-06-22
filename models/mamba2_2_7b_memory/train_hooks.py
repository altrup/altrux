"""Training hooks for mamba2_2_7b_memory, called by sft/train.py's generic
training loop. Contrast with models/mamba2_780m/train_hooks.py: this model's
Model.forward(input_ids, state) is stateful, so examples are processed in
--chunk-len chunks with `state` carried (and detached) across chunks of the
SAME example -- never across different examples -- bounding training RAM by
chunk length rather than example length. See
models/mamba2_2_7b_memory/README.md for why long examples aren't truncated at
data-prep time instead.

Unlike standard SFT (and unlike the other models' hooks), loss is computed
over EVERY token, not just assistant turns: prepare_data.py's mask marks user
turns as non-trainable, which is correct for short Q&A-style chat, but for
this model most of the content that's supposed to exercise long-range recall
is *in* the long user turns (a document, a long context) -- masking that out
would throw away most of the signal this model exists to learn from. `mask`
is accepted (for interface parity with the other models' hooks) but ignored.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sft"))
from lora import apply_lora

from . import model as _model_mod

EOS_ID = 0  # <|endoftext|> for EleutherAI/gpt-neox-20b
# This model's manual, unfused, per-token mixer step costs ~1GB of VRAM per
# token while a backward graph is live -- on this project's dev GPU (8GB),
# chunk_len 512 (or even 32) OOMs before finishing a single chunk's forward
# pass; chunk_len 4 gets through forward but OOMs in .backward(); 3 is the
# largest value confirmed to get through forward without OOMing. NOTE: this
# is not a confirmed-safe value for backward specifically -- chunk_len 1-3
# all hit a separate non-finite-loss bug (see process_example) before ever
# reaching .backward(), so backward's memory cost at this size is still
# unverified. Override with --chunk-len once that's resolved.
DEFAULT_CHUNK_LEN = 3


def setup_training(device, lora_rank: int, lora_alpha: float, lora_dropout: float):
    """Loads the backbone (quantizing TARGET_LORA_MODULES if
    QUANTIZE_LORA_BASE), attaches LoRA, then wraps in Model -- training
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


def _chunk_loss(model, input_ids: torch.Tensor, target_ids: torch.Tensor, state, eos_weight: float):
    logits, state = model(input_ids, state=state)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="none")
    weight = torch.ones_like(loss)
    if eos_weight != 1.0:
        weight[target_ids.reshape(-1) == EOS_ID] = eos_weight
    return (loss * weight).sum(), weight.sum(), state


def process_example(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    device,
    eos_weight: float,
    backward_scale: float,
    chunk_len: int | None = None,
) -> tuple[float, float, int]:
    """ids already on `device`; mask ignored (see module docstring). Calls
    .backward() once per chunk (state carried within this example, detached
    at each boundary -- truncated BPTT). Returns (loss_sum, weight_sum,
    n_chunks) -- n_chunks lets the caller count gradient-accumulation steps
    per chunk rather than per example, matching this model's actual backward
    granularity."""
    chunk_len = chunk_len or DEFAULT_CHUNK_LEN
    state = None
    total_loss = total_weight = 0.0
    n_chunks = 0
    seqlen = ids.numel()
    for start in range(0, seqlen - 1, chunk_len):
        end = min(start + chunk_len, seqlen - 1)
        input_ids = ids[start:end].unsqueeze(0)
        target_ids = ids[start + 1 : end + 1].unsqueeze(0)
        weighted_loss, weight, state = _chunk_loss(model, input_ids, target_ids, state, eos_weight)
        if not torch.isfinite(weighted_loss):
            raise FloatingPointError(f"non-finite loss at chunk [{start}:{end}]")
        (weighted_loss / weight * backward_scale).backward()
        total_loss += weighted_loss.item()
        total_weight += weight.item()
        state = state.detach()
        n_chunks += 1
    return total_loss, total_weight, n_chunks


def eval_loss(
    model,
    eval_ids: list[torch.Tensor],
    eval_masks: list[torch.Tensor],
    device,
    max_len: int,
    chunk_len: int | None = None,
) -> float:
    chunk_len = chunk_len or DEFAULT_CHUNK_LEN
    model.eval()
    total_loss = total_tokens = 0.0
    with torch.no_grad():
        for ids in eval_ids:
            if ids.numel() > max_len or ids.numel() < 2:
                continue
            ids = ids.to(device)
            state = None
            seqlen = ids.numel()
            for start in range(0, seqlen - 1, chunk_len):
                end = min(start + chunk_len, seqlen - 1)
                input_ids = ids[start:end].unsqueeze(0)
                target_ids = ids[start + 1 : end + 1].unsqueeze(0)
                weighted_loss, weight, state = _chunk_loss(model, input_ids, target_ids, state, eos_weight=1.0)
                total_loss += weighted_loss.item()
                total_tokens += weight.item()
                state = state.detach()
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float("nan")


def preflight(
    model,
    trainable_params: list[torch.nn.Parameter],
    all_ids: list[torch.Tensor],
    all_masks: list[torch.Tensor],
    device,
    max_len: int,
    eos_weight: float = 1.0,
    chunk_len: int | None = None,
) -> None:
    sample = None
    for ids in all_ids:
        if 2 <= ids.numel() <= max_len:
            sample = ids
            break
    assert sample is not None, "no valid examples found in dataset"

    model.train()
    process_example(model, sample.to(device), None, device, eos_weight, backward_scale=1.0, chunk_len=chunk_len)

    lora_grads = memory_grads = 0
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None or p.grad.abs().max() == 0:
            continue
        if "lora_A" in name or "lora_B" in name:
            lora_grads += 1
        else:
            memory_grads += 1
    model.zero_grad()

    assert lora_grads > 0, "preflight: 0 LoRA params received gradients"
    assert memory_grads > 0, "preflight: 0 memory-subsystem params received gradients"
    print(f"preflight OK -- {lora_grads} LoRA params and {memory_grads} memory-subsystem params have gradients")
