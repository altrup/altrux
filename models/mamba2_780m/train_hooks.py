"""Training hooks for mamba2_780m, called by sft/train.py's generic training
loop.

Model.forward loops over tokens and threads a MixerState across calls (see
model.py for why -- mamba_ssm's own fused kernels are broken on this
hardware), so training here chunks long examples the same way
models/mamba2_2_7b_memory/train_hooks.py does: state is carried and detached
across chunks of the *same* example, never across different examples, so
training RAM is bounded by chunk length rather than example length.

sft/train.py owns everything that's the same across models: shuffling,
gradient-accumulation counting, checkpoint cadence/rotation, resume, and
non-finite checks. What's irreducibly model-specific -- how to load/wrap the
model for training, and how to run forward+backward for one example -- lives
here.
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


def _chunk_loss(model, input_ids: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor, state):
    """loss_mask: per-target-position weight (0 for non-assistant tokens,
    1 or eos_weight for assistant tokens -- computed by the caller, which
    already has both the conversation mask and eos_weight available)."""
    logits, state = model(input_ids, state=state)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="none")
    return (loss * loss_mask).sum(), loss_mask.sum(), state


def process_example(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    device,
    eos_weight: float,
    backward_scale: float,
    chunk_len: int | None = None,
) -> tuple[float, float, int]:
    """ids, mask already on `device`. Loss is masked to assistant turns via
    `mask`, same as before chunking -- unlike mamba2_2_7b_memory, this model
    has no reason to train on user turns too. Returns (loss_sum, weight_sum,
    n_chunks) -- n_chunks is what the generic loop counts against
    --accum-steps."""
    chunk_len = chunk_len or DEFAULT_CHUNK_LEN
    state = None
    total_loss = total_weight = 0.0
    seqlen = ids.numel()
    n_chunks = 0
    for start in range(0, seqlen - 1, chunk_len):
        end = min(start + chunk_len, seqlen - 1)
        input_ids = ids[start:end].unsqueeze(0)
        target_ids = ids[start + 1 : end + 1].unsqueeze(0)
        loss_mask = mask[start + 1 : end + 1].float().clone()
        if eos_weight != 1.0:
            eos_positions = (target_ids.view(-1) == EOS_ID) & (loss_mask > 0)
            loss_mask[eos_positions] = eos_weight

        weighted_loss, weight, state = _chunk_loss(model, input_ids, target_ids, loss_mask, state)
        if weight > 0:
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
        for ids, mask in zip(eval_ids, eval_masks):
            if ids.numel() > max_len or mask.sum() == 0:
                continue
            ids, mask = ids.to(device), mask.to(device)
            state = None
            seqlen = ids.numel()
            for start in range(0, seqlen - 1, chunk_len):
                end = min(start + chunk_len, seqlen - 1)
                input_ids = ids[start:end].unsqueeze(0)
                target_ids = ids[start + 1 : end + 1].unsqueeze(0)
                loss_mask = mask[start + 1 : end + 1].float()
                weighted_loss, weight, state = _chunk_loss(model, input_ids, target_ids, loss_mask, state)
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
    sample_ids = sample_mask = None
    for ids, mask in zip(all_ids, all_masks):
        if mask.any() and ids.numel() <= max_len:
            sample_ids, sample_mask = ids, mask
            break
    assert sample_ids is not None, "no valid examples found in dataset"

    model.train()
    process_example(model, sample_ids.to(device), sample_mask.to(device), device, eos_weight, backward_scale=1.0, chunk_len=chunk_len)

    grads_nonzero = sum(1 for p in trainable_params if p.grad is not None and p.grad.abs().max() > 0)
    model.zero_grad()

    assert grads_nonzero > 0, f"preflight: 0/{len(trainable_params)} params received gradients"
    print(f"preflight OK -- {grads_nonzero}/{len(trainable_params)} params have gradients")
