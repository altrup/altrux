"""Converts databricks/databricks-dolly-15k's context-bearing categories (a
short document plus an instruction about it, the shape of a LAMA evidence
document) into the {messages: [{role, content}]} JSONL shape
preparation/conversations.py reads from --input. A seeded partition holds out
probe contexts that never reach the output; --acknowledge writes framed
contexts with a templated acknowledgement instead of instruction rows."""

import argparse
import json
import random
from pathlib import Path

from progress import ts

CATEGORIES = {"summarization", "closed_qa", "information_extraction"}
# Equal to experiments.lama_ckl.protocol.WAKE_FRAME (a test pins it); not
# imported so preparation stays free of experiment code.
ACK_FRAME = "Remember this for later: "
ACK_TEMPLATES = (
    "Got it. I'll remember that.",
    "Noted. I will keep that in mind.",
    "Understood. I'll remember this.",
    "Okay, I've got it.",
    "Thanks, I'll keep that for later.",
)


def dolly_messages(row: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": row["context"] + "\n\n" + row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ]
    }


def acknowledgement_messages(row: dict, rng: random.Random) -> dict:
    return {
        "messages": [
            {"role": "user", "content": ACK_FRAME + row["context"]},
            {"role": "assistant", "content": rng.choice(ACK_TEMPLATES)},
        ]
    }


HELD_OUT_SEED = 0


def partition(rows: list[dict], held_out: int = 50) -> tuple[list[dict], list[dict]]:
    """Fixed seed: the probe set must not move with --seed, or a top-up drawn
    under another seed could carry probe contexts."""
    shuffled = list(rows)
    random.Random(HELD_OUT_SEED).shuffle(shuffled)
    return shuffled[:held_out], shuffled[held_out:]


def write_rows(rows: list[dict], args: argparse.Namespace) -> None:
    probe, pool = partition(rows)
    rng = random.Random(args.seed)

    held_out = Path(args.held_out_output)
    held_out.parent.mkdir(parents=True, exist_ok=True)
    held_out.write_text(json.dumps([row["context"] for row in probe]))

    if args.acknowledge > 0:
        if args.acknowledge > len(pool):
            raise ValueError(f"--acknowledge {args.acknowledge} exceeds {len(pool)} pool rows")
        mode = "acknowledge"
        written = [acknowledgement_messages(row, rng) for row in rng.sample(pool, args.acknowledge)]
    else:
        mode = "instruction"
        drawn = (
            pool
            if args.max_examples is None
            else rng.sample(pool, min(args.max_examples, len(pool)))
        )
        written = [dolly_messages(row) for row in drawn]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(messages) + "\n" for messages in written)

    print(f"[{ts()}] {len(rows)} context rows: {len(probe)} probe -> {held_out}, {len(pool)} pool")
    print(f"[{ts()}] mode {mode}: wrote {len(written)} rows to {out}")
    sample = written[0]["messages"]
    print(f"[{ts()}] sample user: {sample[0]['content'][:400]!r}")
    print(f"[{ts()}] sample assistant: {sample[1]['content'][:200]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Dolly context rows to messages JSONL")
    parser.add_argument(
        "--max-examples", type=int, default=None, help="Pool rows drawn (default: all)"
    )
    parser.add_argument(
        "--acknowledge",
        type=int,
        default=0,
        help="Write this many acknowledgement rows instead of instruction rows (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Draw seed (default: %(default)s)")
    parser.add_argument("--output", default="data/dolly_context.jsonl", help="Output JSONL path")
    parser.add_argument(
        "--held-out-output",
        default="data/dolly_heldout.json",
        help="Probe contexts JSON path (default: %(default)s)",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"[{ts()}] loading databricks/databricks-dolly-15k")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    write_rows([row for row in ds if row["category"] in CATEGORIES], args)


if __name__ == "__main__":
    main()
