"""Experiment: shared

babilong-style needle blocks: the 15% retention slice of the next run
(notes/discussion/DISCUSSION-20260724-next-run-plan.md 1.3).

The RMT-proven needle form under the *identical* curriculum, block assembly
and recall weighting as the Wikipedia cram slice -- everything structural
comes from preparation.cram.build_blocks; only the items differ:

  source = a bAbI story (RMT-team/babilong's `0k` config: the task text with
           no filler, so the gap is ours to control, not the benchmark's)
  cue    = the dataset's own question
  answer = the dataset's own target, credited in full

Wikipedia passages (the same loader the cram slice uses, no NER needed) are
the interference between a story and its question.

Kept as a minority slice because its failure modes are disjoint from IMR's:
synthetic facts cannot leak from pretraining, while bAbI's tiny template
space trains format-matching as much as memory.

  make prepare-needles
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from preparation.cram import (
    add_block_args,
    build_blocks,
    emit,
    load_wikipedia_passages,
    split_articles,
)


def babilong_items(records: list[dict], task: str) -> list[dict]:
    """{input, question, target} -> cram-shaped items. Drops records whose
    target already appears in the question (nothing to recall) or whose
    target is missing from the story (unanswerable)."""
    items = []
    for i, r in enumerate(records):
        story, question, target = r["input"].strip(), r["question"].strip(), r["target"].strip()
        if not target or target in question or target not in story:
            continue
        items.append(
            {
                "source": story,
                "cue": question,
                "answer": target,
                "span": (0, len(target)),
                "meta": {"article": f"babilong-{task}-{i}", "entity": target, "entity_type": task},
            }
        )
    return items


def load_babilong(args) -> list[dict]:
    from datasets import load_dataset

    items: list[dict] = []
    for task in args.tasks:
        ds = load_dataset("RMT-team/babilong", args.config, split=task)
        if args.max_per_task:
            ds = ds.select(range(min(args.max_per_task, len(ds))))
        items += babilong_items(list(ds), task)
        print(f"\r  {task}: {len(items)} items so far", end="", flush=True)
    print(f"\r  {len(args.tasks)} tasks, {len(items)} items")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="data/train_needles.pt")
    parser.add_argument("--heldout-output", default="data/eval_needles.pt")
    parser.add_argument(
        "--config", default="0k", help="babilong context config; 0k is the bare task text"
    )
    parser.add_argument("--tasks", nargs="+", default=[f"qa{i}" for i in range(1, 11)])
    parser.add_argument("--max-per-task", type=int, default=400)
    parser.add_argument(
        "--articles", type=int, default=3000, help="Wikipedia articles streamed for filler"
    )
    parser.add_argument("--wiki-dataset", default="wikimedia/wikipedia")
    parser.add_argument("--wiki-config", default="20231101.en")
    parser.add_argument("--passages-per-article", type=int, default=3)
    parser.add_argument("--min-words", type=int, default=45)
    parser.add_argument("--max-words", type=int, default=140)
    parser.add_argument("--heldout-frac", type=float, default=0.05)
    add_block_args(parser)
    # One story per block, Wikipedia in the gap -- BABILong's own construction.
    # A second bAbI story would re-state where the apple is and invalidate the
    # first question's target. And a bAbI target ("kitchen") recurs across
    # stories by construction while the binding the question asks about does
    # not, so test B of preparation/filtering.py, not a string check, decides
    # solvability for this slice.
    parser.set_defaults(max_items_per_block=1, allow_repeated_credit=True)
    args = parser.parse_args()

    import importlib
    import os

    from models.common import build_tokenizer

    model_mod = importlib.import_module(f"models.{os.getenv('MODEL_NAME', 'mamba2_780m')}")
    tokenizer = build_tokenizer(model_mod)
    user_id = tokenizer.convert_tokens_to_ids(model_mod.USER_OPEN)
    asst_id = tokenizer.convert_tokens_to_ids(model_mod.ASST_OPEN)
    sep = tokenizer.encode(" ", add_special_tokens=False)
    nl = tokenizer.encode("\n", add_special_tokens=False)
    assert len(sep) == 1 and len(nl) == 1, (
        f'expected " " and "\\n" to be single tokens, got {sep} {nl}'
    )

    items = load_babilong(args)
    passages = load_wikipedia_passages(args)
    train_titles, heldout_titles = split_articles(
        [p["article"] for p in passages], args.heldout_frac, args.seed
    )
    train_ids, heldout_ids = split_articles(
        [it["meta"]["article"] for it in items], args.heldout_frac, args.seed
    )

    def encode(strings: list[str]) -> list[list[int]]:
        return tokenizer(strings, add_special_tokens=False)["input_ids"]

    for name, keep_items, keep_titles, path in (
        ("needles", train_ids, train_titles, args.output),
        ("needles-heldout", heldout_ids, heldout_titles, args.heldout_output),
    ):
        sel = [it for it in items if it["meta"]["article"] in keep_items]
        fillers = [p["text"] for p in passages if p["article"] in keep_titles]
        print(f"\n== {name}: {len(sel)} needles, {len(fillers)} filler passages ==")
        # One bAbI story answers one question, so every group is a singleton and
        # the items-per-source density knob is inert on this slice.
        dataset, stats = build_blocks(
            [[it] for it in sel],
            fillers,
            encode,
            user_id=user_id,
            asst_id=asst_id,
            sep_id=sep[0],
            nl_id=nl[0],
            args=args,
        )
        # Both halves of the reservation: held-out needles and the held-out
        # articles their filler comes from.
        emit(
            dataset,
            stats,
            path,
            tokenizer,
            user_id,
            asst_id,
            name,
            sorted(heldout_titles | heldout_ids),
            args,
        )


__all__ = [
    "add_block_args",
    "babilong_items",
    "build_blocks",
    "emit",
    "load_babilong",
    "load_wikipedia_passages",
    "split_articles",
    "main",
]


if __name__ == "__main__":
    main()
