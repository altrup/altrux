"""Experiment: shared

Count gist-eval-eligible conversations in a preparation/conversations.py .pt file.

A conversation is eligible for (prefix P, cont C) when some turn-boundary
token (USER_OPEN/ASST_OPEN) has >= P tokens before it and >= C after —
build_gist_rows' criterion in diagnostics/recall.py. Prints a P x C grid so the
probe config can be chosen per corpus before sweeping.

Usage: python eligibility_check.py data/eval_ultrachat.pt [--prefixes 1024,2048,...] [--conts 256,512]
"""

import argparse

import torch
from transformers import AutoTokenizer

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--prefixes", default="1024,2048,3072,4096,6144")
    parser.add_argument("--conts", default="256,512")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[USER]", "[ASSISTANT]"]})
    user_id = tokenizer.convert_tokens_to_ids("[USER]")
    asst_id = tokenizer.convert_tokens_to_ids("[ASSISTANT]")

    data = torch.load(args.data, map_location="cpu", weights_only=True)
    prefixes = [int(x) for x in args.prefixes.split(",")]
    conts = [int(x) for x in args.conts.split(",")]

    lens = sorted(len(ids) for ids in data["ids"])
    n = len(lens)
    print(
        f"{args.data}: {n} conversations; len median {lens[n // 2]}, p90 {lens[int(n * 0.9)]}, max {lens[-1]}"
    )

    counts = {(p, c): 0 for p in prefixes for c in conts}
    for i, ids in enumerate(data["ids"]):
        if i % 2000 == 0:
            print(f"\r{i}/{n}", end="", flush=True)
        bounds = ((ids == user_id) | (ids == asst_id)).nonzero().flatten().tolist()
        for p in prefixes:
            if len(ids) < p:
                continue
            for c in conts:
                if any(b >= p and len(ids) - b >= c for b in bounds):
                    counts[(p, c)] += 1
    print(f"\r{n}/{n}")

    header = "prefix\\cont " + " ".join(f"{c:>6}" for c in conts)
    print(header)
    for p in prefixes:
        print(f"{p:>11} " + " ".join(f"{counts[(p, c)]:>6}" for c in conts))


if __name__ == "__main__":
    main()
