"""Warm-start SFT: teach the model the repo's chat format before any dream.

Registered in notes/DISCUSSION-20260807-g2-results-erase-geometry-and-warmstart-run.md
sec 3.1. The `[USER]`/`[ASSISTANT]` markers are freshly-registered special
tokens whose embeddings the base backbone has never seen, so a dream generated
after the assistant marker runs off-distribution (mojibake, bracket mimicry).
This trains those embeddings (the marker delta) plus a LoRA on ordinary chat
data rendered through prepare_data.py's exact turn format, and writes one
adapter checkpoint that every dream_sleep cell then loads with
`--init-adapter`.

The checkpoint format, the LoRA rank/alpha and the dropout come from lora.py --
load_adapter is fatal on a mismatch, so there is nothing here to tune
independently of dream_sleep.

Box tool: trains a LoRA on the real backbone, so it runs on the rented CUDA
hardware, not the local ROCm box.

Usage (from sft/, env vars as in the Makefile):
    make warm-start ARGS="--out checkpoints/warm_start.pt"
    make warm-start ARGS="--steps 800 --out checkpoints/warm_start_800.pt"
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from consolidation_null import fmt_duration, ts
from lora import DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_RANK, save_adapter
from prepare_data import format_conversation

# Chunk length for the box's GPU, not this repo's 8GB dev box (whose
# train_hooks.DEFAULT_CHUNK_LEN of 48 would make 400 steps ~20k tokens -- far
# too little text to move the marker embeddings). Lower it with --chunk-len if
# the run OOMs.
CHUNK_LEN = 512


def build_corpus(records, tokenizer, user_open: str, asst_open: str, max_len: int, turns: int):
    """Render conversations through prepare_data's turn format until `turns`
    marker turns are collected. Returns (corpus, turns_rendered), where corpus
    is [(ids, loss_mask)]."""
    marker_ids = {tokenizer.convert_tokens_to_ids(user_open), tokenizer.convert_tokens_to_ids(asst_open)}
    corpus: list[tuple[list[int], list[bool]]] = []
    rendered = 0
    for record in records:
        ids, mask, _ = format_conversation(record["messages"], tokenizer, max_len, user_open, asst_open)
        if not any(mask):
            continue
        corpus.append((ids, mask))
        rendered += sum(tok in marker_ids for tok in ids)
        print(f"\r[{ts()}] corpus: {len(corpus)} conversations, {rendered}/{turns} turns", end="", flush=True)
        if rendered >= turns:
            break
    print()
    return corpus, rendered


def corpus_invariants(corpus, tokenizer, user_open: str, asst_open: str) -> dict[str, int]:
    """Counts whose correct value is zero: a user turn nobody answered, a
    trained token outside an assistant turn, an example with nothing to train
    on."""
    user_id = tokenizer.convert_tokens_to_ids(user_open)
    asst_id = tokenizer.convert_tokens_to_ids(asst_open)
    counts = {
        "user_turns_with_no_answer": 0,
        "trained_tokens_outside_assistant_turns": 0,
        "examples_with_no_trained_tokens": 0,
    }
    for ids, mask in corpus:
        if not any(mask):
            counts["examples_with_no_trained_tokens"] += 1
        role = None
        for tok, trained in zip(ids, mask, strict=True):
            if tok == user_id:
                counts["user_turns_with_no_answer"] += role == "user"
                role = "user"
            elif tok == asst_id:
                role = "assistant"
            counts["trained_tokens_outside_assistant_turns"] += trained and role != "assistant"
        counts["user_turns_with_no_answer"] += role == "user"
    return counts


def sample_text(ids: list[int], tokenizer, asst_open: str, width: int = 200) -> str:
    """Decoded text either side of the first user->assistant join -- the one
    structural event this corpus exists to teach."""
    asst_id = tokenizer.convert_tokens_to_ids(asst_open)
    join = ids.index(asst_id) if asst_id in ids else 0
    return tokenizer.decode(ids[max(0, join - width):join + width])


def iter_chunks(corpus, chunk_len: int, rng: random.Random):
    """Endless stream of (starts_example, input_ids, target_ids, weights) over
    a reshuffled corpus. Weights index target positions, as in train.py."""
    import torch

    while True:
        for i in rng.sample(range(len(corpus)), len(corpus)):
            ids, mask = corpus[i]
            toks = torch.tensor(ids, dtype=torch.long)
            weights = torch.tensor(mask, dtype=torch.float32)
            for lo in range(0, len(ids) - 1, chunk_len):
                hi = min(lo + chunk_len, len(ids) - 1)
                yield lo == 0, toks[lo:hi][None], toks[lo + 1:hi + 1][None], weights[lo + 1:hi + 1][None]


def train(model, hooks, trainable, corpus, steps: int, lr: float, chunk_len: int,
          seed: int, device, log_every: int = 10) -> int:
    """Cross-entropy on the assistant turns, one optimizer step per chunk that
    carries trained tokens. Returns the number of steps taken."""
    import torch

    opt = torch.optim.AdamW(trainable, lr=lr)
    chunks = iter_chunks(corpus, chunk_len, random.Random(seed))
    state = None
    step = 0
    tokens = 0
    window: list[float] = []
    start = time.time()
    while step < steps:
        fresh, inp, tgt, weights = next(chunks)
        if fresh:
            state = None
        loss_sum, weight_sum, state = hooks.chunk_loss(
            model, inp.to(device), tgt.to(device), weights.to(device), state, 1.0
        )
        if weight_sum > 0 and torch.isfinite(loss_sum):
            (loss_sum / weight_sum).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tokens += int(weight_sum.item())
            window.append(loss_sum.item() / weight_sum.item())
            if step % log_every == 0 or step == steps:
                rate = (time.time() - start) / step
                print(f"[{ts()}] step {step}/{steps} loss {sum(window) / len(window):.4f} "
                      f"({tokens:,} trained tokens, {rate:.2f}s/step, ETA {fmt_duration(rate * (steps - step))})")
                window.clear()
        state = state.detach() if state is not None and hasattr(state, "detach") else state
    return step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hf-dataset", default="HuggingFaceH4/ultrachat_200k", help="Chat corpus (default: %(default)s)")
    parser.add_argument("--hf-split", default="train_sft", help="Dataset split (default: %(default)s)")
    parser.add_argument("--turns", type=int, default=4000, help="Rendered turns to collect (default: %(default)s)")
    parser.add_argument("--max-len", type=int, default=4096, help="Max tokens per conversation (default: %(default)s)")
    parser.add_argument("--steps", type=int, default=400, help="Optimizer steps; the registered fallback is 800 (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default: %(default)s)")
    parser.add_argument("--chunk-len", type=int, default=CHUNK_LEN, help="Tokens per forward chunk (default: %(default)s)")
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--lora-alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--log-every", type=int, default=10, help="Steps between loss lines (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="checkpoints/warm_start.pt", help="Adapter checkpoint (default: %(default)s)")
    return parser


def main() -> None:
    import torch
    from datasets import load_dataset

    from models.common import build_tokenizer

    args = build_parser().parse_args()
    model_name = os.getenv("MODEL_NAME", "mamba2_780m")
    model_mod = importlib.import_module(f"models.{model_name}")
    train_hooks = importlib.import_module(f"models.{model_name}.train_hooks")
    tokenizer = build_tokenizer(model_mod)

    print(f"[{ts()}] warm start for {model_name}: {args.turns} turns of {args.hf_dataset}, "
          f"{args.steps} steps at lr {args.lr}, LoRA rank {args.lora_rank} alpha {args.lora_alpha} "
          f"dropout {DEFAULT_DROPOUT}, seed {args.seed}")

    ds = load_dataset(args.hf_dataset, split=args.hf_split)
    # Seeded index sample, not the file order: ultrachat_200k is grouped by
    # source subset, so the head of the split is not a sample of the corpus.
    order = random.Random(args.seed).sample(range(len(ds)), len(ds))
    corpus, rendered = build_corpus(
        (ds[i] for i in order), tokenizer, model_mod.USER_OPEN, model_mod.ASST_OPEN, args.max_len, args.turns
    )
    counts = corpus_invariants(corpus, tokenizer, model_mod.USER_OPEN, model_mod.ASST_OPEN)
    total_tokens = sum(len(ids) for ids, _ in corpus)
    trained_tokens = sum(sum(mask) for _, mask in corpus)
    print(f"[{ts()}] {len(corpus)} conversations, {rendered} turns, {total_tokens:,} tokens "
          f"({trained_tokens:,} trained)")
    for name, count in counts.items():
        print(f"[{ts()}] invariant {name}: {count} (must be 0)")
    if any(counts.values()):
        raise SystemExit("corpus invariants failed -- not training on this")
    print(f"[{ts()}] sample at the first user->assistant join:\n---\n"
          f"{sample_text(corpus[0][0], tokenizer, model_mod.ASST_OPEN)}\n---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, trainable = train_hooks.setup_training(device, args.lora_rank, args.lora_alpha, DEFAULT_DROPOUT)
    model.train()
    print(f"[{ts()}] training on {device}, chunk_len {args.chunk_len}")
    train(model, train_hooks, trainable, corpus, args.steps, args.lr, args.chunk_len,
          args.seed, device, args.log_every)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_adapter(model, out, args.lora_rank, args.lora_alpha)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"[{ts()}] wrote {out} (sha256 {sha}) -- load it with "
          f"`dream_sleep.py --init-adapter {out}`")


if __name__ == "__main__":
    main()
