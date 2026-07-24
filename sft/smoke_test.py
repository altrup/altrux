"""Fast real-model sanity check: loads the actual model (paying the real
load/quantize cost) and drives it through train.py's real run_training loop
-- same slot-based batching, same gradient-accumulation counting, same
--batch-size/--accum-tokens defaults as `make train` -- on a synthetic dataset
sized to complete exactly one optimizer step, then asserts gradients actually
reached the trainable parameters. This is what catches a model whose forward
silently fails to connect some part of itself to the loss (see
models/mamba2_2_7b_memory/train_hooks.py's history for why this check
matters), including bugs that only show up under real batching/accumulation
(e.g. a per-slot state reset that only breaks slot > 0). LoRA params and any
other trainable params (e.g. a model's own full-gradient subsystem) are
checked separately so one silently disconnected branch can't hide behind the
other's gradient.

Checkpoints written during the run are redirected to a throwaway temp
directory (train.CKPT_DIR is monkeypatched for the duration) so this never
touches the real models/{name}/checkpoints/ tree.

Using a synthetic dataset instead of the real one avoids paying for real
example length -- for mamba2_2_7b_memory, real examples can be tens of
thousands of tokens, which at this model's --chunk-len would be thousands of
slow chunks before the check tells you anything. This script does NOT
confirm the real dataset is actually usable end-to-end (tokenization,
example lengths, etc.) -- only that gradients wire through correctly under
the real batching/accumulation path.

Each synthetic example defaults to exactly `accum_steps * chunk_len + 1`
tokens (accum_steps derived from --accum-tokens the same way run_training
derives it -- see train.py's run_training) so all `batch_size` slots finish
together after precisely one optimizer step -- override with --length to
also use this as a tool for tuning DEFAULT_CHUNK_LEN: a single isolated
chunk's peak VRAM is NOT representative of real multi-chunk training -- the
held-over gradient/cache floor from earlier chunks eats into the next
chunk's headroom (confirmed: chunk_len values that ran clean on one chunk
OOM'd on the second chunk of a real run). Always tune chunk_len against
several consecutive chunks, never one.
"""

import argparse
import math
import tempfile
from pathlib import Path

import torch

import train


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Fast real-model gradient-wiring smoke test for {train.MODEL_NAME}")
    parser.add_argument("--length", type=int, default=None, help="Per-example synthetic sequence length in tokens -- defaults to accum_steps * chunk_len + 1 (accum_steps derived from --accum-tokens), so batch_size slots finish together after exactly one optimizer step")
    parser.add_argument("--chunk-len", type=int, default=None, help="Defaults to the model's own DEFAULT_CHUNK_LEN")
    parser.add_argument("--memory-window", type=int, default=None, help="Same meaning as train.py's --memory-window -- no-op for models without set_memory_window")
    parser.add_argument("--batch-size", type=int, default=6, help="Same meaning as train.py's --batch-size")
    parser.add_argument("--accum-tokens", type=int, default=256, help="Same meaning as train.py's --accum-tokens")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eos-weight", type=float, default=5.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading {train.MODEL_ID} ...")
    model, trainable_params = train.hooks.setup_training(device, args.lora_rank, args.lora_alpha, args.lora_dropout)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    chunk_len = args.chunk_len or train.hooks.DEFAULT_CHUNK_LEN
    # Mirrors run_training's own accum_tokens -> accum_steps derivation so
    # `length` actually produces one optimizer step's worth of chunks.
    accum_steps = max(1, round(args.accum_tokens / chunk_len))
    length = args.length if args.length is not None else accum_steps * chunk_len + 1
    print(f"chunk_len: {chunk_len}  batch_size: {args.batch_size}  accum_tokens: {args.accum_tokens} (accum_steps: {accum_steps})  length: {length} ({length / chunk_len:.1f} chunks/example)")

    # Small ids are safe for any of this project's tokenizers (vocab sizes
    # are all in the tens of thousands) -- no need to know the real vocab size.
    train_ids = [torch.randint(0, 100, (length,)) for _ in range(args.batch_size)]
    train_masks = [torch.ones(length, dtype=torch.bool) for _ in range(args.batch_size)]

    # Gradients get zeroed by run_training's own optimizer.step()/zero_grad(),
    # so inspecting p.grad afterward can't tell us whether a param ever
    # received one. register_hook fires with the freshly computed gradient on
    # every backward() call, before it's added into .grad -- recording that
    # here survives the later zeroing.
    grad_seen = {"lora": False, "other": False}
    handles = []
    lora_total = other_total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind = "lora" if ("lora_A" in name or "lora_B" in name) else "other"
        if kind == "lora":
            lora_total += 1
        else:
            other_total += 1

        def _record(grad, kind=kind):
            if grad is not None and torch.isfinite(grad).any() and grad.abs().max() > 0:
                grad_seen[kind] = True
            return grad

        handles.append(p.register_hook(_record))

    run_args = argparse.Namespace(
        data="<synthetic>",
        epochs=1,
        eos_weight=args.eos_weight,
        recall_weight=1.0,
        head_weight=1.0,
        head_tokens=1024,
        accum_tokens=args.accum_tokens,
        chunk_len=chunk_len,
        batch_size=args.batch_size,
        ckpt_every_tokens=math.inf,
        keep_ckpts=1,
        keep_full_state=0,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        warmup_steps=0,
        max_len=math.inf,
        memory_window=args.memory_window,
    )

    original_ckpt_dir = train.CKPT_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            train.CKPT_DIR = Path(tmp_dir)
            train.run_training(
                train.hooks, model, optimizer, trainable_params, train_ids, train_masks,
                [None] * len(train_ids), [None] * len(train_ids), device, run_args,
                start_epoch=0, start_slot_states=None, start_next_ptr=0, start_step=0,
                start_total_tokens=0.0, start_last_ckpt_tokens=0.0, start_full_state=None,
            )
    finally:
        train.CKPT_DIR = original_ckpt_dir
        for h in handles:
            h.remove()

    lora_ok = grad_seen["lora"]
    other_ok = grad_seen["other"]
    assert lora_ok or other_ok, "smoke test: 0 trainable params received gradients"
    assert lora_total == 0 or lora_ok, "smoke test: 0 LoRA params received gradients"
    assert other_total == 0 or other_ok, "smoke test: 0 non-LoRA trainable params received gradients"
    print(f"smoke test OK -- LoRA params received gradients: {lora_ok} ({lora_total} params), other trainable params received gradients: {other_ok} ({other_total} params)")


if __name__ == "__main__":
    main()
