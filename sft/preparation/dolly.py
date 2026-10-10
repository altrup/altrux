"""Converts databricks/databricks-dolly-15k's context-bearing categories (a
short document plus an instruction about it, the shape of a LAMA evidence
document) into the {messages: [{role, content}]} JSONL shape
preparation/conversations.py reads from --input."""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from progress import ts

CATEGORIES = {"summarization", "closed_qa", "information_extraction"}


def dolly_messages(row: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": row["context"] + "\n\n" + row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Dolly context rows to messages JSONL")
    parser.add_argument(
        "--max-examples", type=int, default=1000, help="Rows drawn (default: %(default)s)"
    )
    parser.add_argument("--seed", type=int, default=0, help="Draw seed (default: %(default)s)")
    parser.add_argument("--output", default="data/dolly_context.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = [row for row in ds if row["category"] in CATEGORIES]
    drawn = random.Random(args.seed).sample(rows, min(args.max_examples, len(rows)))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in drawn:
            f.write(json.dumps(dolly_messages(row)) + "\n")

    print(f"[{ts()}] wrote {len(drawn)} of {len(rows)} context rows to {out}")
    sample = dolly_messages(drawn[0])["messages"]
    print(f"[{ts()}] sample user: {sample[0]['content'][:400]!r}")
    print(f"[{ts()}] sample assistant: {sample[1]['content'][:200]!r}")


if __name__ == "__main__":
    main()
