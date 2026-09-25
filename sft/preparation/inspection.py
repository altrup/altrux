"""Experiment: shared

Quick cross-slice sanity read before training: a few decoded windows and
headline counts from each data artifact -- NOT a substitute for the full
generation-time validation, just the ten-second read that catches a slice
whose contents don't match its name (root CLAUDE.md: no dataset goes to a
training run until someone has read a sample).

  make sanity-sample ARGS="--data data/train_chains.pt --data data/train_cram.pt"

Windows prefer recall-credited spans (cram/needles: shows a cue->answer with
its credit), then sleep positions (chains: shows a wipe boundary), then random
text. Output is meant to be handed to a reviewer (a cheap subagent on the
box) with the question: does each slice look like what it claims to be?
"""

import argparse
import random
import time

import torch


def _ts() -> str:
    return time.strftime("[%H:%M:%S]")


def pick_windows(n_tokens: int, recall, sleeps, n: int, width: int, seed: int):
    """Up to n (lo, hi, kind) windows, kind in recall|sleep|random, each
    centered on the most informative positions the artifact offers."""
    rng = random.Random(seed)
    half = width // 2

    def clamp(center: int, kind: str):
        lo = max(0, center - half)
        hi = min(n_tokens, center + half)
        return (lo, hi, kind)

    centers: list[tuple[int, str]] = []
    if recall is not None and bool(recall.any()):
        starts = (recall[1:] & ~recall[:-1]).nonzero().flatten() + 1
        if recall[0]:
            starts = torch.cat([torch.tensor([0]), starts])
        centers += [(int(s), "recall") for s in rng.sample(starts.tolist(), min(n, len(starts)))]
    if len(centers) < n and sleeps is not None and len(sleeps):
        picks = rng.sample(sleeps.tolist(), min(n - len(centers), len(sleeps)))
        centers += [(int(s), "sleep") for s in picks]
    while len(centers) < n:
        centers.append((rng.randrange(max(1, n_tokens)), "random"))
    return [clamp(c, k) for c, k in centers]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data", action="append", required=True, help="Artifact .pt; repeat per slice"
    )
    parser.add_argument("--samples", type=int, default=2, help="Decoded windows per artifact")
    parser.add_argument("--width", type=int, default=400, help="Window width in tokens")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import importlib
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.common import build_tokenizer

    model_mod = importlib.import_module(f"models.{os.getenv('MODEL_NAME', 'mamba2_780m')}")
    tokenizer = build_tokenizer(model_mod)

    for path in args.data:
        d = torch.load(path, map_location="cpu", weights_only=False)
        ids_list = d["ids"]
        recalls = d.get("recall_masks") or [None] * len(ids_list)
        sleeps_list = d.get("sleep_positions") or [None] * len(ids_list)
        n_tok = sum(len(x) for x in ids_list)
        n_recall = sum(int(r.sum()) for r in recalls if r is not None)
        n_sleeps = sum(len(s) for s in sleeps_list if s is not None)
        print(
            f"\n{_ts()} == {path}: {len(ids_list)} examples, {n_tok / 1e6:.1f}M tokens, "
            f"{n_recall} recall-credited ({100 * n_recall / max(n_tok, 1):.2f}%), {n_sleeps} sleeps =="
        )

        rng = random.Random(args.seed)
        for si, ei in enumerate(rng.sample(range(len(ids_list)), min(args.samples, len(ids_list)))):
            ids, recall, sleeps = ids_list[ei], recalls[ei], sleeps_list[ei]
            for lo, hi, kind in pick_windows(
                len(ids), recall, sleeps, 1, args.width, args.seed + si
            ):
                text = tokenizer.decode(ids[lo:hi].tolist())
                print(f"{_ts()}  [example {ei}, tokens {lo}:{hi}, around {kind}]\n    {text!r}")


if __name__ == "__main__":
    main()
