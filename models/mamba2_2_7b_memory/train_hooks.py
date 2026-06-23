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
# pass; chunk_len 4 gets through forward but OOMs in .backward(). chunk_len 3
# was confirmed to OOM in .backward() too once the non-finite-loss bug (see
# model.py's MAX_WRITE_GRAD_NORM/_rms_normalize/eta cap) was fixed and
# backward could actually be reached -- 2 is the largest value confirmed to
# get all the way through both forward and backward without OOMing. Gradient
# checkpointing on the per-token mixer step would be the way to raise this
# again within the same 8GB budget (at the cost of ~2x forward compute);
# override with --chunk-len if running on a GPU with more VRAM.
DEFAULT_CHUNK_LEN = 2


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


def _live_progress_lines(model, end: int, seqlen: int, chunk_loss: float) -> list[str]:
    """Builds the per-chunk live-status lines for process_example's \r
    update -- kept separate from _print_live's cursor-movement bookkeeping
    so formatting can change without touching the printing mechanics.
    last_token_log is a live snapshot (not the per-optimizer-step average
    pop_memory_stats reports), so this reflects exactly this chunk's last
    token. Two lines because cramming both onto one and truncating to
    terminal width would lose content; see _print_live for how a known,
    fixed number of lines gets overwritten in place regardless of width."""
    lines = [f"  token {end:>6}/{seqlen:<6}  loss {chunk_loss:.4f}"]
    log = model.last_token_log()
    if log is not None:
        lines.append(
            f"  beta {log['beta']:.4f}  clear {log['clear']:.4f}"
            f"  active {log['active_layers']:>2}/{log['n_layers']}"
            f"  surprise {log['surprise']:.4f}  o_t_norm {log['o_t_norm']:.4f}"
        )
    return lines


def _print_live(lines: list[str], prev_n_lines: int) -> int:
    """Overwrites whatever this function last printed (prev_n_lines lines)
    with `lines`, in place -- moves the cursor up to the start of that
    block first (ESC[<n>A), then clears to end of each line as it's
    rewritten (ESC[K) so a shorter new line doesn't leave old characters
    trailing past its end. Unlike clamping to terminal width, this doesn't
    truncate any content -- it works for any number of lines, as long as
    the caller passes back the count it returns each time so the next call
    knows how far to rewind. Only does anything useful on a real terminal;
    when piped (e.g. `make train`'s `tee`), the escape codes land in the
    log file as literal bytes, same as the plain \r this replaced."""
    out = (f"\033[{prev_n_lines - 1}A" if prev_n_lines > 1 else "") + "\r"
    out += "\n".join(f"{line}\033[K" for line in lines)
    sys.stdout.write(out)
    sys.stdout.flush()
    return len(lines)


def _clear_live(prev_n_lines: int) -> None:
    """Clears whatever _print_live last left on screen (ESC[J clears from
    the cursor to end of screen, removing every line of the block at once)."""
    if prev_n_lines == 0:
        return
    out = (f"\033[{prev_n_lines - 1}A" if prev_n_lines > 1 else "") + "\r\033[J"
    sys.stdout.write(out)
    sys.stdout.flush()


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
    prev_n_lines = 0
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
        # This model's manual mixer step is slow enough per-token (see
        # DEFAULT_CHUNK_LEN's comment) that a single long example can take
        # minutes -- live in-place update so it's visible without flooding
        # the log with one line per chunk.
        chunk_loss = weighted_loss.item() / weight.item()
        prev_n_lines = _print_live(_live_progress_lines(model, end, seqlen, chunk_loss), prev_n_lines)
    _clear_live(prev_n_lines)
    return total_loss, total_weight, n_chunks


def extra_log(model) -> str | None:
    """Optional hook -- train.py calls this (if defined) after each optimizer
    step and prints whatever string it returns (None to print nothing this
    step). Used here to report whether the memory subsystem is actually
    being used, not just receiving gradient -- see Model.pop_memory_stats
    for what beta/clear/surprise/o_t_norm mean."""
    stats = model.pop_memory_stats()
    if stats is None:
        return None
    return (
        f"memory  beta {stats['beta']:.4f}  clear {stats['clear']:.4f}"
        f"  surprise {stats['surprise']:.4f}  o_t_norm {stats['o_t_norm']:.4f}"
    )


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
