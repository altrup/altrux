"""Injects interference-recall structure into an already-tokenized dataset.

Motivation (see notes/WATCH_NOTES.md, 2026-07-17): probe_recall.py showed the
backbone's SSM state alone handles the recall load the current training data
ever presents (a handful of facts per example), so the gradient has no reason
to use the Titans neural memory and learns to suppress it instead. The
memory's real niche starts where SSM capacity collapses (~64+ interfering
facts). This script manufactures that regime: it takes a fraction of the
existing examples and splices in a block of labeled random codes ("The code
for river is 4 8 2 1 3.") near the start, plus several query/answer turn
pairs at random later turn boundaries. Everything operates on token ids —
the carrier conversations are never re-tokenized, only the short injected
turns are (batched), so a full dataset regenerates in minutes on CPU.

Masks follow the dataset's convention: True on assistant-turn content. Facts
are user turns (False); each query's assistant answer is True — that answer
IS the recall training signal for mask-respecting models (mamba2_2_7b_memory
itself trains on all tokens regardless).

Label words come from the same vocab scan as probe_recall.py but from a
disjoint slice (skip=1024), so the probe remains an honest held-out eval.

  make prepare-interference           # data/train_memory.pt -> data/train_memory_v2.pt
  uv run --no-sync python prepare_interference.py --fraction 0.3 --max-facts 256
"""

import argparse
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from probe_recall import single_token_labels

LABEL_POOL = 1024
LABEL_SKIP = 1024  # probe_recall uses labels [0, 1024); we train on [1024, 2048)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default="data/train_memory.pt")
    parser.add_argument("--output", default="data/train_memory_v2.pt")
    parser.add_argument("--fraction", type=float, default=0.3, help="Fraction of examples to inject into")
    parser.add_argument("--min-facts", type=int, default=8)
    parser.add_argument("--max-facts", type=int, default=256)
    parser.add_argument("--min-queries", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import models.mamba2_2_7b_memory as model_mod
    from models.common import build_tokenizer

    tokenizer = build_tokenizer(model_mod)
    user_open, asst_open = model_mod.USER_OPEN, model_mod.ASST_OPEN
    user_id = tokenizer.convert_tokens_to_ids(user_open)
    asst_id = tokenizer.convert_tokens_to_ids(asst_open)
    labels = single_token_labels(tokenizer, LABEL_POOL, skip=LABEL_SKIP)

    print(f"loading {args.source} ...")
    data = torch.load(args.source, map_location="cpu", weights_only=True)
    ids_list, masks_list = data["ids"], data["masks"]
    n = len(ids_list)

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(range(n), int(n * args.fraction)))

    # Pass 1: plan every injection and collect all injected-turn strings for
    # one batched tokenizer call.
    plans = []
    strings: list[str] = []
    for ex in selected:
        # log-uniform fact count, so small and large loads are both represented
        n_facts = int(math.exp(rng.uniform(math.log(args.min_facts), math.log(args.max_facts))))
        ex_labels = rng.sample(labels, n_facts)
        codes = [" " + " ".join(str(rng.randrange(10)) for _ in range(5)) for _ in range(n_facts)]
        fact_idx = []
        for label, code in zip(ex_labels, codes):
            fact_idx.append(len(strings))
            strings.append(f"{user_open} The code for {label} is{code}.")
        n_queries = rng.randint(args.min_queries, min(args.max_queries, n_facts))
        query_idx = []
        for j in rng.sample(range(n_facts), n_queries):
            query_idx.append((len(strings), len(strings) + 1))
            strings.append(f"{user_open} What was the code for {ex_labels[j]}?")
            strings.append(f"{asst_open} The code for {ex_labels[j]} is{codes[j]}.")
        plans.append((ex, fact_idx, query_idx))

    print(f"tokenizing {len(strings)} injected turns ...")
    encoded = tokenizer(strings, add_special_tokens=False)["input_ids"]

    def turn_tensors(idx: int, trainable: bool) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.tensor(encoded[idx], dtype=torch.long)
        m = torch.zeros(len(t), dtype=torch.bool)
        if trainable:
            m[1:] = True  # True on assistant content, False on the role marker itself
        return t, m

    # Pass 2: splice. Facts go in one block at the second turn boundary (the
    # conversation still opens naturally); each query/answer pair goes at a
    # random later turn boundary, ordered so earlier text always states the
    # facts a query asks about.
    for done, (ex, fact_idx, query_idx) in enumerate(plans):
        ids, masks = ids_list[ex], masks_list[ex]
        boundaries = ((ids == user_id) | (ids == asst_id)).nonzero().flatten().tolist()
        fact_pos = boundaries[1] if len(boundaries) > 1 else len(ids)
        later = [b for b in boundaries if b > fact_pos]
        inserts: list[tuple[int, list[int]]] = [(fact_pos, fact_idx)]
        for pair in query_idx:
            pos = rng.choice(later) if later else len(ids)
            inserts.append((pos, list(pair)))
        inserts.sort(key=lambda x: x[0])

        id_parts, mask_parts, cursor = [], [], 0
        for pos, turn_ids in inserts:
            id_parts.append(ids[cursor:pos])
            mask_parts.append(masks[cursor:pos])
            for ti in turn_ids:
                t, m = turn_tensors(ti, trainable=(ti in {q[1] for q in query_idx}))
                id_parts.append(t)
                mask_parts.append(m)
            cursor = pos
        id_parts.append(ids[cursor:])
        mask_parts.append(masks[cursor:])
        ids_list[ex] = torch.cat(id_parts)
        masks_list[ex] = torch.cat(mask_parts)
        if done % 200 == 0:
            print(f"\r  spliced {done + 1}/{len(plans)} examples", end="", flush=True)
    print(f"\r  spliced {len(plans)}/{len(plans)} examples")

    torch.save({"ids": ids_list, "masks": masks_list}, args.output)
    total = sum(len(t) for t in ids_list)
    print(f"wrote {args.output}: {n} examples ({len(plans)} injected), {total / 1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
