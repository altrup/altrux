"""Injects interference-recall structure into an already-tokenized dataset.

Motivation (see notes/WATCH_NOTES.md, 2026-07-17): probe_recall.py showed the
backbone's SSM state alone handles the recall load the current training data
ever presents (a handful of facts per example), so the gradient has no reason
to use the Titans neural memory and learns to suppress it instead. The
memory's real niche starts where SSM capacity collapses (~64+ interfering
facts). This script manufactures that regime: it takes a fraction of the
existing examples and splices in a block of key/value facts near the start,
plus several query/answer turn pairs at random later turn boundaries.

Facts are deliberately heterogeneous -- digit codes, keywords, colors,
weekdays, counts, owners -- each with several statement/question/answer
phrasings, so the only invariant the memory can exploit is key->value
binding itself, not one template's surface form. A fraction of facts are
later REVISED ("Actually, the code for river has changed to ...") with
queries placed after the revision expecting the newest value -- training
the delta-rule write's overwrite behavior directly.

Everything operates on token ids -- the carrier conversations are never
re-tokenized, only the short injected turns are (batched), so a full
dataset regenerates in minutes on CPU.

Masks follow the dataset's convention: True on assistant-turn content. Facts
are user turns (False); each query's assistant answer is True -- that answer
IS the recall training signal for mask-respecting models (mamba2_2_7b_memory
itself trains on all tokens regardless).

The output also carries a `recall_masks` list (True exactly on the spliced
answer-content tokens, None for untouched examples) so train.py's
--recall-weight can amplify the recall signal, which is otherwise ~0.1% of
all tokens.

Fact keys come from the same vocab scan as probe_recall.py but from a
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


LABEL_POOL = 1024
LABEL_SKIP = 1024  # probe_recall uses labels [0, 1024); we train on [1024, 2048)

COLORS = ["red", "blue", "green", "teal", "amber", "violet", "crimson", "olive", "navy", "coral"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NAMES = ["Maria", "Ethan", "Priya", "Jonas", "Amara", "Felix", "Nadia", "Oscar", "Lena", "Tariq", "Ivy", "Marcus"]


def _digits(rng: random.Random) -> str:
    return " ".join(str(rng.randrange(10)) for _ in range(5))


# Each kind: value sampler + (statement, question, answer, revision) template
# quadruples. Statement/question/answer are picked as a matched set so the
# semantics always line up; within a set the phrasing still varies per fact.
FACT_KINDS = [
    {
        "value": _digits,
        "sets": [
            ("The code for {k} is {v}.", "What was the code for {k}?", "The code for {k} is {v}.",
             "Actually, the code for {k} has changed to {v}."),
            ("{k}'s passcode is {v}.", "Do you remember {k}'s passcode?", "{k}'s passcode is {v}.",
             "Correction: {k}'s passcode is now {v}."),
            ("Please note down the access code {v} for {k}.", "Which access code goes with {k}?", "It's {v}.",
             "Scratch that -- the access code for {k} is now {v}."),
        ],
    },
    {
        "value": "vocab_word",
        "sets": [
            ("The keyword for {k} is {v}.", "What was the keyword for {k}?", "The keyword for {k} is {v}.",
             "Update: the keyword for {k} is now {v}."),
            ("{k} is filed under {v}.", "What is {k} filed under?", "{k} is filed under {v}.",
             "We moved {k}; it is now filed under {v}."),
        ],
    },
    {
        "value": lambda rng: rng.choice(COLORS),
        "sets": [
            ("The {k} folder is marked {v}.", "What color is the {k} folder marked?", "The {k} folder is marked {v}.",
             "The {k} folder was re-marked {v}."),
            ("{k}'s team wears {v}.", "What color does {k}'s team wear?", "{k}'s team wears {v}.",
             "{k}'s team switched to wearing {v}."),
        ],
    },
    {
        "value": lambda rng: rng.choice(WEEKDAYS),
        "sets": [
            ("The meeting about {k} is on {v}.", "When is the meeting about {k}?", "The {k} meeting is on {v}.",
             "The meeting about {k} was moved to {v}."),
            ("The {k} shipment arrives on {v}.", "Which day does the {k} shipment arrive?", "It arrives on {v}.",
             "The {k} shipment was rescheduled to {v}."),
        ],
    },
    {
        "value": lambda rng: str(rng.randrange(2, 99)),
        "sets": [
            ("There are {v} boxes in the {k} room.", "How many boxes are in the {k} room?",
             "There are {v} boxes in the {k} room.", "Recount: the {k} room now holds {v} boxes."),
            ("The {k} order is for {v} units.", "How many units is the {k} order for?", "The {k} order is for {v} units.",
             "The {k} order was amended to {v} units."),
        ],
    },
    {
        "value": lambda rng: rng.choice(NAMES),
        "sets": [
            ("The {k} ledger belongs to {v}.", "Who does the {k} ledger belong to?", "The {k} ledger belongs to {v}.",
             "The {k} ledger was handed over to {v}."),
            ("{v} is in charge of {k}.", "Who is in charge of {k}?", "{v} is in charge of {k}.",
             "{v} has taken over {k}."),
        ],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default="data/train_memory.pt")
    parser.add_argument("--output", default="data/train_memory_v2.pt")
    parser.add_argument("--fraction", type=float, default=0.3, help="Fraction of examples to inject into")
    parser.add_argument("--min-facts", type=int, default=8)
    parser.add_argument("--max-facts", type=int, default=256)
    parser.add_argument("--min-queries", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--revise-rate", type=float, default=0.12, help="Fraction of facts later revised to a new value")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from diagnostics.recall import single_token_labels
    import models.mamba2_2_7b_memory as model_mod
    from models.common import build_tokenizer

    tokenizer = build_tokenizer(model_mod)
    user_open, asst_open = model_mod.USER_OPEN, model_mod.ASST_OPEN
    user_id = tokenizer.convert_tokens_to_ids(user_open)
    asst_id = tokenizer.convert_tokens_to_ids(asst_open)
    labels = single_token_labels(tokenizer, LABEL_POOL, skip=LABEL_SKIP)

    def sample_value(kind, rng: random.Random) -> str:
        if kind["value"] == "vocab_word":
            return rng.choice(labels)
        return kind["value"](rng)

    print(f"loading {args.source} ...")
    data = torch.load(args.source, map_location="cpu", weights_only=True)
    ids_list, masks_list = data["ids"], data["masks"]
    n = len(ids_list)

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(range(n), int(n * args.fraction)))

    # Pass 1: plan every injection and collect all injected-turn strings for
    # one batched tokenizer call. Each planned turn is a string index; the
    # plan records where it goes and what it is.
    plans = []
    strings: list[str] = []

    def add_string(s: str) -> int:
        strings.append(s)
        return len(strings) - 1

    n_revised_total = 0
    for ex in selected:
        # log-uniform fact count, so small and large loads are both represented
        n_facts = int(math.exp(rng.uniform(math.log(args.min_facts), math.log(args.max_facts))))
        ex_keys = rng.sample(labels, n_facts)
        facts = []  # per fact: dict(kind_set, key, value, revised_value|None, string idxs)
        fact_turn_idx = []
        for k in ex_keys:
            kind = rng.choice(FACT_KINDS)
            stmt, q, a, rev = rng.choice(kind["sets"])
            v = sample_value(kind, rng)
            fact = {"templates": (q, a, rev), "k": k, "v": v, "v2": None}
            fact_turn_idx.append(add_string(f"{user_open} {stmt.format(k=k, v=v)}"))
            if rng.random() < args.revise_rate:
                fact["v2"] = sample_value(kind, rng)
                fact["rev_idx"] = add_string(f"{user_open} {rev.format(k=k, v=fact['v2'])}")
                n_revised_total += 1
            facts.append(fact)

        n_queries = rng.randint(args.min_queries, min(args.max_queries, n_facts))
        query_plan = []  # (fact_idx, question_string_idx, answer_string_idx)
        for j in rng.sample(range(n_facts), n_queries):
            f = facts[j]
            q, a, _ = f["templates"]
            final_v = f["v2"] if f["v2"] is not None else f["v"]
            query_plan.append(
                (j, add_string(f"{user_open} {q.format(k=f['k'])}"), add_string(f"{asst_open} {a.format(k=f['k'], v=final_v)}"))
            )
        plans.append((ex, fact_turn_idx, facts, query_plan))

    print(f"tokenizing {len(strings)} injected turns ({n_revised_total} revisions) ...")
    encoded = tokenizer(strings, add_special_tokens=False)["input_ids"]

    def turn_tensors(idx: int, trainable: bool) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.tensor(encoded[idx], dtype=torch.long)
        m = torch.zeros(len(t), dtype=torch.bool)
        if trainable:
            m[1:] = True  # True on assistant content, False on the role marker itself
        return t, m

    # Pass 2: splice. Facts go in one block at the second turn boundary (the
    # conversation still opens naturally). Revisions land at random later
    # boundaries; queries about a revised fact only land after its revision,
    # so the expected answer is always the newest value.
    recall_list: list[torch.Tensor | None] = [None] * n
    for done, (ex, fact_turn_idx, facts, query_plan) in enumerate(plans):
        ids, masks = ids_list[ex], masks_list[ex]
        boundaries = ((ids == user_id) | (ids == asst_id)).nonzero().flatten().tolist()
        fact_pos = boundaries[1] if len(boundaries) > 1 else len(ids)
        later = [b for b in boundaries if b > fact_pos] or [len(ids)]

        rev_pos: dict[int, int] = {}
        inserts: list[tuple[int, list[int], bool]] = [(fact_pos, fact_turn_idx, False)]
        for j, f in enumerate(facts):
            if f["v2"] is not None:
                # revisions go in the earlier half of the conversation so
                # post-revision queries still have room after them
                pos = rng.choice(later[: max(1, len(later) // 2)])
                rev_pos[j] = pos
                inserts.append((pos, [f["rev_idx"]], False))
        for j, q_idx, a_idx in query_plan:
            floor = rev_pos.get(j, fact_pos)
            candidates = [b for b in later if b > floor] or [len(ids)]
            inserts.append((rng.choice(candidates), [q_idx, a_idx], True))
        inserts.sort(key=lambda x: x[0])

        id_parts, mask_parts, recall_parts, cursor = [], [], [], 0
        for pos, turn_ids, is_query in inserts:
            id_parts.append(ids[cursor:pos])
            mask_parts.append(masks[cursor:pos])
            recall_parts.append(torch.zeros(pos - cursor, dtype=torch.bool))
            for i, ti in enumerate(turn_ids):
                t, m = turn_tensors(ti, trainable=(is_query and i == 1))
                id_parts.append(t)
                mask_parts.append(m)
                recall_parts.append(m)  # True exactly on spliced answer content
            cursor = pos
        id_parts.append(ids[cursor:])
        mask_parts.append(masks[cursor:])
        recall_parts.append(torch.zeros(len(ids) - cursor, dtype=torch.bool))
        ids_list[ex] = torch.cat(id_parts)
        masks_list[ex] = torch.cat(mask_parts)
        recall_list[ex] = torch.cat(recall_parts)
        if done % 200 == 0:
            print(f"\r  spliced {done + 1}/{len(plans)} examples", end="", flush=True)
    print(f"\r  spliced {len(plans)}/{len(plans)} examples")

    torch.save({"ids": ids_list, "masks": masks_list, "recall_masks": recall_list}, args.output)
    total = sum(len(t) for t in ids_list)
    n_recall = sum(int(r.sum()) for r in recall_list if r is not None)
    print(
        f"wrote {args.output}: {n} examples ({len(plans)} injected, {n_revised_total} revised facts), "
        f"{total / 1e6:.1f}M tokens ({n_recall / 1e3:.1f}k recall-answer tokens for --recall-weight)"
    )

__all__ = [
    "LABEL_POOL",
    "LABEL_SKIP",
    "COLORS",
    "WEEKDAYS",
    "NAMES",
    "FACT_KINDS",
    "main",
]


if __name__ == "__main__":
    main()
