"""Fast real-model sanity check: loads the actual model (paying the real
load/quantize cost) and runs the same chunked forward+backward path the
training loop uses on a short synthetic sequence, then asserts gradients
actually reached the trainable parameters -- this is what catches a model
whose forward silently fails to connect some part of itself to the loss (see
models/mamba2_780m_memory/train_hooks.py's history for why this check
matters). LoRA params and any other trainable params (e.g. a model's own
full-gradient subsystem) are checked separately so one silently disconnected
branch can't hide behind the other's gradient.

Using a synthetic sequence instead of a real dataset example avoids paying
for dataset example length -- for mamba2_780m_memory, real examples can be
tens of thousands of tokens, which at this model's --chunk-len would be
thousands of slow chunks before the check tells you anything. This script
does NOT confirm the real dataset is actually usable end-to-end (tokenization,
example lengths, etc.) -- only that gradients wire through correctly.

`--length` defaults to 5x the resolved chunk_len (not a single chunk's
worth) so this also doubles as the tool for tuning DEFAULT_CHUNK_LEN: a
single isolated chunk's peak VRAM is NOT representative of real multi-chunk
training -- the held-over gradient/cache floor from earlier chunks eats into
the next chunk's headroom (confirmed: chunk_len values that ran clean at
--length == --chunk-len OOM'd on the second chunk of a real run). Always
tune chunk_len against several consecutive chunks, never one.
"""

import argparse
from datetime import datetime

import torch

import train


def _run_chunks(hooks, model, ids: torch.Tensor, mask: torch.Tensor, device, eos_weight: float, chunk_len: int) -> None:
    """Runs one synthetic example through the same chunked forward+backward
    path the main training loop uses. Used to catch a model whose forward
    silently fails to connect some part of itself to the loss (see
    models/mamba2_780m_memory/train_hooks.py's history for why this check
    matters)."""
    model.train()
    ids = ids.to(device)
    mask = mask.to(device)
    seqlen = ids.numel()
    chunk_extra_log = getattr(hooks, "chunk_extra_log", None)
    state = None
    prev_n_lines = 0
    for start, end, input_ids, target_ids, mask_slice in train._chunks(ids, mask, chunk_len):
        loss_sum, weight_sum, state = hooks.chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight)
        if weight_sum > 0:
            chunk_loss_val = loss_sum.item() / weight_sum.item()
            (loss_sum / weight_sum).backward()
            ts = datetime.now().strftime("%H:%M:%S")
            lines = [f"[{ts}]  token {end:>6}/{seqlen:<6}  loss {chunk_loss_val:.4f}"]
            if chunk_extra_log is not None:
                extra = chunk_extra_log(model)
                if isinstance(extra, list):
                    extra = extra[0] if extra else None
                if extra:
                    lines.append(f"  {extra}")
            prev_n_lines = train._print_live(lines, prev_n_lines)
        state = state.detach() if state is not None else None
    train._clear_live(prev_n_lines)

    lora_total = other_total = 0
    lora_grads = other_grads = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        has_grad = int(p.grad is not None and p.grad.abs().max() > 0)
        if "lora_A" in name or "lora_B" in name:
            lora_total += 1
            lora_grads += has_grad
        else:
            other_total += 1
            other_grads += has_grad
    model.zero_grad()

    assert lora_grads + other_grads > 0, "smoke test: 0 trainable params received gradients"
    assert lora_total == 0 or lora_grads > 0, "smoke test: 0 LoRA params received gradients"
    assert other_total == 0 or other_grads > 0, "smoke test: 0 non-LoRA trainable params received gradients"
    print(f"smoke test OK -- {lora_grads} LoRA params and {other_grads} other trainable params have gradients")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Fast real-model gradient-wiring smoke test for {train.MODEL_NAME}")
    parser.add_argument("--length", type=int, default=None, help="Synthetic sequence length in tokens -- defaults to 5x the resolved chunk_len, to exercise several consecutive chunks rather than one isolated chunk")
    parser.add_argument("--chunk-len", type=int, default=None, help="Defaults to the model's own DEFAULT_CHUNK_LEN")
    parser.add_argument("--eos-weight", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading {train.MODEL_ID} ...")
    model, _ = train.hooks.setup_training(device, args.lora_rank, args.lora_alpha, args.lora_dropout)

    chunk_len = args.chunk_len or train.hooks.DEFAULT_CHUNK_LEN
    length = args.length if args.length is not None else chunk_len * 5
    print(f"chunk_len: {chunk_len}  length: {length} ({length / chunk_len:.1f} chunks)")

    # Small ids are safe for any of this project's tokenizers (vocab sizes
    # are all in the tens of thousands) -- no need to know the real vocab size.
    ids = torch.randint(0, 100, (length,))
    mask = torch.ones(length, dtype=torch.bool)

    _run_chunks(train.hooks, model, ids, mask, device, args.eos_weight, chunk_len)


if __name__ == "__main__":
    main()
