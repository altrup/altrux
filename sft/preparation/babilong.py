"""Converts RMT-team/babilong (a long-context recall QA benchmark, schema
{input, question, target}) into the {messages: [{role, content}]} JSONL shape
preparation/conversations.py expects from --input. Kept separate from preparation/conversations.py
because babilong's schema is benchmark-specific, not a chat dataset --
running this once produces a JSONL that preparation/conversations.py then tokenizes
exactly like any other conversation dataset.

babilong embeds the answer-bearing fact at a random position inside long,
mostly-irrelevant filler text, specifically to force genuine long-range
recall rather than reward attending to nearby context -- see the README for
why this is mixed in alongside THUDM/LongAlign-10k.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert RMT-team/babilong to messages-shaped JSONL"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["4k", "8k"],
        help="babilong context-length configs to pull (see RMT-team/babilong on the Hub for the full list)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[f"qa{i}" for i in range(1, 11)],
        help="bAbI task splits within each config (qa1..qa10)",
    )
    parser.add_argument(
        "--max-per-split", type=int, default=100, help="Cap examples per (config, task) pair"
    )
    parser.add_argument("--output", default="data/babilong_raw.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out, "w") as f:
        for config in args.configs:
            for task in args.tasks:
                ds = load_dataset("RMT-team/babilong", config, split=task)
                if args.max_per_split:
                    ds = ds.select(range(min(args.max_per_split, len(ds))))
                for ex in ds:
                    # The question rides in its own field, not joined into
                    # content: preparation/conversations.py appends it to the same user turn
                    # (token-identical to the joined form) but records its
                    # token offset, so preparation/chains.py's split-QA can move
                    # the dataset's own question verbatim to the resumed tail.
                    record = {
                        "messages": [
                            {"role": "user", "content": ex["input"], "question": ex["question"]},
                            {"role": "assistant", "content": ex["target"]},
                        ]
                    }
                    f.write(json.dumps(record) + "\n")
                    written += 1

    print(f"Wrote {written} examples to {out}")


if __name__ == "__main__":
    main()
