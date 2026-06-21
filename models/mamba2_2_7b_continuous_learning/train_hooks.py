"""Training hooks for mamba2_2_7b_continuous_learning, called by
sft/train.py's generic training loop. This model's forward is a single
stateless call regardless of sequence length, so there's no chunking here --
contrast with models/mamba2_2_7b_memory/train_hooks.py, which needs state
threaded and detached across chunks.

sft/train.py owns everything that's the same across models: shuffling,
gradient-accumulation counting, checkpoint cadence/rotation, resume, and
non-finite checks. What's irreducibly model-specific -- how to load/wrap the
model for training, and how to run forward+backward for one example -- lives
here. (Training-wise this is currently identical to mamba2_780m's hooks --
duplicated rather than shared, matching how the two models' model.py files
are already independent rather than sharing a base class.)
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sft"))
from lora import apply_lora

from . import model as _model_mod

EOS_ID = 0  # <|endoftext|> for EleutherAI/gpt-neox-20b


def setup_training(device, lora_rank: int, lora_alpha: float, lora_dropout: float):
    """Loads the raw backbone (not the inference Model wrapper -- this
    model's forward is already stateless, so training operates directly on
    load_base()'s output) with LoRA attached. Returns (model, trainable_params)."""
    model = _model_mod.load_base(str(device))
    model = model.to(torch.float32)
    model = apply_lora(model, _model_mod.TARGET_LORA_MODULES, lora_rank, lora_alpha, lora_dropout)

    for param in model.parameters():
        param.requires_grad_(False)
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
            trainable_params.append(param)
    print(f"trainable params: {sum(p.numel() for p in trainable_params):,}")
    return model, trainable_params


def process_example(
    model,
    ids: torch.Tensor,
    mask: torch.Tensor,
    device,
    eos_weight: float,
    backward_scale: float,
    chunk_len: int | None = None,
) -> tuple[float, float, int]:
    """ids, mask already on `device`. chunk_len is accepted for interface
    parity with chunked models but unused here. Returns (loss_sum, weight_sum,
    n_backward_calls) -- n_backward_calls is always 1 for this model."""
    input_ids = ids[:-1].unsqueeze(0)
    target_ids = ids[1:].unsqueeze(0)
    loss_mask = mask[1:].float().clone()
    if eos_weight != 1.0:
        eos_positions = (target_ids.view(-1) == EOS_ID) & (loss_mask > 0)
        loss_mask[eos_positions] = eos_weight
    weight = loss_mask.sum()
    logits = model(input_ids).logits
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction="none")
    weighted_loss = (loss * loss_mask).sum()
    (weighted_loss / weight * backward_scale).backward()
    return weighted_loss.item(), weight.item(), 1


def eval_loss(
    model,
    eval_ids: list[torch.Tensor],
    eval_masks: list[torch.Tensor],
    device,
    max_len: int,
    chunk_len: int | None = None,
) -> float:
    model.eval()
    total_loss = total_tokens = 0.0
    with torch.no_grad():
        for ids, mask in zip(eval_ids, eval_masks):
            if ids.numel() > max_len or mask.sum() == 0:
                continue
            ids, mask = ids.to(device), mask.to(device)
            input_ids = ids[:-1].unsqueeze(0)
            target_ids = ids[1:].unsqueeze(0)
            loss_mask = mask[1:].float()
            weight = loss_mask.sum()
            logits = model(input_ids).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction="none")
            total_loss += (loss * loss_mask).sum().item()
            total_tokens += weight.item()
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
    process_example(model, sample_ids.to(device), sample_mask.to(device), device, eos_weight, backward_scale=1.0)

    grads_nonzero = sum(1 for p in trainable_params if p.grad is not None and p.grad.abs().max() > 0)
    model.zero_grad()

    assert grads_nonzero > 0, f"preflight: 0/{len(trainable_params)} params received gradients"
    print(f"preflight OK -- {grads_nonzero}/{len(trainable_params)} params have gradients")
