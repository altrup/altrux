import argparse
import importlib
import json
import multiprocessing
import os
import random
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

# Add the repo root to sys.path so the models/ package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.common import build_tokenizer

from progress import ts

_worker_tokenizer = None
_worker_max_len = None
_worker_markers: tuple[str, str] | None = None
_worker_eoc: str | None = None


def _worker_init(model_name: str, max_len: int) -> None:
    global _worker_tokenizer, _worker_max_len, _worker_markers, _worker_eoc
    # Imported here, not at module top: importing a model package pulls in
    # mamba_ssm, which needs a working GPU even to import -- keeping it lazy
    # keeps format_conversation importable (and testable) without one.
    mod = importlib.import_module(f"models.{model_name}")
    _worker_tokenizer = build_tokenizer(mod)
    _worker_max_len = max_len
    _worker_markers = (mod.USER_OPEN, mod.ASST_OPEN)
    _worker_eoc = getattr(mod, "EOC", None)


def _worker_format(
    group: list[tuple[str, list[dict]]],
) -> tuple[list[int], list[bool], int | None, int]:
    return format_pack(group, _worker_tokenizer, _worker_max_len, *_worker_markers, _worker_eoc)


def format_conversation(
    messages: list[dict],
    tokenizer,
    max_len: int,
    user_open: str,
    asst_open: str,
    eoc: str | None = None,
) -> tuple[list[int], list[bool], int | None]:
    """Returns (ids, mask, question_offset).

    A user message may carry a separate "question" field (preparation/babilong.py
    emits this): the question is appended to the turn after a newline --
    token-identical to it having been part of `content` -- and its token
    offset within `ids` is returned, so preparation/chains.py's split-QA can later
    cut the episode at the question start without synthesizing any text.
    Offset is None when no question survives (no question field, a tokenizer
    seam mismatch, or the turn dropped by max_len) -- consumers fail closed
    and never split such episodes.

    `eoc` closes the conversation with the model's boundary marker, so the
    corpus teaches it as conversation-end (DISCUSSION-20260808 sec 2.9.3). It
    is trainable: emitting it is the behaviour being taught.
    """
    ids: list[int] = []
    mask: list[bool] = []
    question_offset: int | None = None

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        qoff_in_turn: int | None = None

        if role == "user":
            text = user_open + " " + content + "\n"
            question = msg.get("question")
            if question:
                q_text = question + "\n"
                head_ids = tokenizer.encode(text, add_special_tokens=False)
                q_ids = tokenizer.encode(q_text, add_special_tokens=False)
                # The two-part encoding must reproduce the joint encoding
                # exactly, or the recorded offset would misalign -- fail
                # closed (no metadata) rather than record a wrong offset.
                if tokenizer.encode(text + q_text, add_special_tokens=False) == head_ids + q_ids:
                    turn_ids = head_ids + q_ids
                    qoff_in_turn = len(head_ids)
                else:
                    turn_ids = tokenizer.encode(text + q_text, add_special_tokens=False)
            else:
                turn_ids = tokenizer.encode(text, add_special_tokens=False)
            turn_mask = [False] * len(turn_ids)
        elif role == "assistant":
            text = asst_open + " " + content
            toks = tokenizer.encode(text, add_special_tokens=False)
            turn_ids = toks + [tokenizer.eos_token_id]
            # "train": false keeps the turn as context but excludes it from the
            # loss (e.g. an earlier answer that a later turn revises)
            trainable = msg.get("train", True)
            turn_mask = [trainable] * len(turn_ids)
        else:
            continue

        if len(ids) + len(turn_ids) > max_len:
            break  # drop this and all remaining turns; keeps only complete turns

        if qoff_in_turn is not None and question_offset is None:
            question_offset = len(ids) + qoff_in_turn
        ids.extend(turn_ids)
        mask.extend(turn_mask)

    if eoc and ids:
        ids.append(tokenizer.convert_tokens_to_ids(eoc))
        mask.append(True)

    return ids, mask, question_offset


def pack_records(
    records: list[dict], rng: random.Random, repeat_rate: float
) -> list[list[tuple[str, list[dict]]]]:
    """Group conversations into examples of 2-5, each element tagged `fresh`
    (the next pool record) or `repeat` (the same messages list as an earlier
    fresh conversation of the pack, never repeated twice). A repeat does not
    consume a pool record."""
    groups: list[list[tuple[str, list[dict]]]] = []
    pending = [r.get("messages", []) for r in records]
    at = 0
    while at < len(pending):
        size = rng.randint(2, 5)
        group: list[tuple[str, list[dict]]] = [("fresh", pending[at])]
        at += 1
        unrepeated = [pending[at - 1]]
        while len(group) < size:
            if unrepeated and rng.random() < repeat_rate:
                group.append(("repeat", unrepeated.pop(rng.randrange(len(unrepeated)))))
            elif at < len(pending):
                group.append(("fresh", pending[at]))
                unrepeated.append(pending[at])
                at += 1
            else:
                break
        groups.append(group)
    return groups


def format_pack(
    group: list[tuple[str, list[dict]]],
    tokenizer,
    max_len: int,
    user_open: str,
    asst_open: str,
    eoc: str | None,
) -> tuple[list[int], list[bool], int | None, int]:
    """One packed example: every conversation of `group` in order, each closed
    by `eoc`. `max_len` bounds the whole example, not each conversation, so a
    conversation with no budget left is dropped whole; the returned count is
    how many of `group` actually landed, which is what the boundary invariant
    is checked against."""
    ids: list[int] = []
    mask: list[bool] = []
    question_offset: int | None = None
    packed = 0
    for _kind, messages in group:
        part_ids, part_mask, qoff = format_conversation(
            messages, tokenizer, max_len - len(ids), user_open, asst_open, eoc
        )
        if not part_ids:
            break
        if qoff is not None and question_offset is None:
            question_offset = len(ids) + qoff
        ids.extend(part_ids)
        mask.extend(part_mask)
        packed += 1
    return ids, mask, question_offset, packed


def iter_records(args) -> list[dict]:
    records = []
    if args.hf_dataset:
        from datasets import load_dataset

        ds = load_dataset(args.hf_dataset, split=args.hf_split)
        if args.max_examples:
            ds = ds.select(range(min(args.max_examples, len(ds))))
        records.extend(ds)
    if args.input:
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    if args.hf_dataset and args.input:
        random.Random(args.seed).shuffle(records)
    return records


def report_packing(
    groups, packed_counts, all_ids, tokenizer, eoc: str | None, user_open: str
) -> None:
    """Structural invariants whose correct value is zero, plus decoded text
    either side of one boundary of each kind (root CLAUDE.md: counts confirm
    the generator did what it was told, never that what it was told was right)."""
    eoc_id = tokenizer.convert_tokens_to_ids(eoc)
    user_id = tokenizer.convert_tokens_to_ids(user_open)
    unterminated = sum(1 for ids in all_ids if not len(ids) or int(ids[-1]) != eoc_id)
    miscounted = sum(
        1
        for packed, ids in zip(packed_counts, all_ids, strict=True)
        if int((ids == eoc_id).sum()) != packed
    )
    unopened = sum(
        1
        for ids in all_ids
        for i in range(len(ids) - 1)
        if int(ids[i]) == eoc_id and int(ids[i + 1]) != user_id
    )
    dropped = sum(len(group) - packed for group, packed in zip(groups, packed_counts, strict=True))

    def label(group, j: int) -> str:
        kind, messages = group[j]
        if kind == "fresh":
            return "fresh"
        source = next(
            (k for k in range(j) if group[k][0] == "fresh" and group[k][1] is messages), None
        )
        if source is None:
            return "orphan"
        return "adjacent" if source == j - 1 else "gapped"

    orphaned = sum(label(group, j) == "orphan" for group in groups for j in range(len(group)))
    twice = sum(
        len(repeats) - len({id(m) for m in repeats})
        for repeats in ([m for kind, m in group if kind == "repeat"] for group in groups)
    )
    labels = [
        label(group, j)
        for group, packed in zip(groups, packed_counts, strict=True)
        for j in range(1, packed)
    ]
    n_repeat = labels.count("adjacent") + labels.count("gapped")
    print(
        f"[{ts()}] packing: {len(all_ids)} examples, {sum(packed_counts)} conversations, "
        f"{len(labels)} boundaries ({n_repeat} repeat: {labels.count('adjacent')} adjacent, "
        f"{labels.count('gapped')} gapped; {labels.count('fresh')} fresh)"
    )
    print(f"[{ts()}]   examples not ending at a boundary : {unterminated}  (must be 0)")
    print(f"[{ts()}]   boundaries != conversations packed: {miscounted}  (must be 0)")
    print(f"[{ts()}]   boundaries not opening a new turn : {unopened}  (must be 0)")
    print(f"[{ts()}]   repeats without an earlier source : {orphaned}  (must be 0)")
    print(f"[{ts()}]   sources repeated twice in a pack  : {twice}  (must be 0)")
    print(f"[{ts()}]   conversations dropped by --max-len: {dropped}  (informational)")

    for want in ("fresh", "adjacent", "gapped"):
        found = next(
            (
                (ids, j)
                for group, packed, ids in zip(groups, packed_counts, all_ids, strict=True)
                for j in range(1, packed)
                if label(group, j) == want
            ),
            None,
        )
        if found is None:
            print(f"[{ts()}]   no {want} boundary in this dataset")
            continue
        ids, j = found
        at = [i for i in range(len(ids)) if int(ids[i]) == eoc_id][j - 1]
        print(
            f"[{ts()}]   sample around a {want} boundary:\n"
            f"    ...{tokenizer.decode(ids[max(0, at - 60) : at])!r} "
            f">>{eoc}>> {tokenizer.decode(ids[at + 1 : at + 60])!r}..."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize conversation data for training")
    parser.add_argument("--input", help="Local JSONL file")
    parser.add_argument(
        "--hf-dataset", help="HuggingFace dataset ID (e.g. HuggingFaceH4/ultrachat_200k)"
    )
    parser.add_argument(
        "--hf-split", default="train_sft", help="Dataset split (default: train_sft)"
    )
    parser.add_argument(
        "--max-examples", type=int, default=None, help="Cap number of --hf-dataset examples loaded"
    )
    parser.add_argument("--output", default="data/train.pt", help="Output .pt file")
    parser.add_argument(
        "--max-len",
        type=int,
        default=32768,
        help="Max tokens per example (whole-turn truncation; chunked training handles long examples, so this only guards pathological outliers)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel tokenization workers")
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Pack 2-5 conversations per example, each closed by the model's "
        "conversation-boundary token (DISCUSSION-20260808 sec 2.10.9)",
    )
    parser.add_argument(
        "--repeat-rate",
        type=float,
        default=0.3,
        help="Probability that a packed slot after the first is a verbatim repeat of an "
        "earlier, not yet repeated conversation of the same pack (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Pool shuffle and packing RNG seed (default: %(default)s)",
    )
    args = parser.parse_args()
    if not (args.input or args.hf_dataset):
        parser.error("give --input, --hf-dataset, or both")

    all_ids: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_qoffs: list[int | None] = []
    skipped = 0

    def keep(ids: list[int], mask: list[bool], qoff: int | None) -> bool:
        if not any(mask):
            return False
        all_ids.append(torch.tensor(ids, dtype=torch.long))
        all_masks.append(torch.tensor(mask, dtype=torch.bool))
        all_qoffs.append(qoff)
        return True

    records = iter_records(args)
    model_name = os.getenv("MODEL_NAME", "mamba2_780m")
    model_mod = importlib.import_module(f"models.{model_name}")
    eoc = getattr(model_mod, "EOC", None)
    if args.pack and not eoc:
        raise SystemExit(
            f"--pack needs a conversation-boundary token; models.{model_name} defines no EOC"
        )

    groups = (
        pack_records(records, random.Random(args.seed), args.repeat_rate)
        if args.pack
        else [[("fresh", r.get("messages", []))] for r in records]
    )
    kept_groups: list[list[tuple[str, list[dict]]]] = []
    packed_counts: list[int] = []
    tokenizer = build_tokenizer(model_mod)

    def formatted():
        if args.workers > 1:
            with multiprocessing.Pool(
                args.workers,
                initializer=_worker_init,
                initargs=(model_name, args.max_len),
            ) as pool:
                yield from pool.imap(_worker_format, groups)
        else:
            for group in groups:
                yield format_pack(
                    group, tokenizer, args.max_len, model_mod.USER_OPEN, model_mod.ASST_OPEN, eoc
                )

    for i, (ids, mask, qoff, packed) in enumerate(formatted()):
        print(f"\r[{ts()}] {i + 1}/{len(groups)}", end="", flush=True)
        if keep(ids, mask, qoff):
            kept_groups.append(groups[i])
            packed_counts.append(packed)
        else:
            skipped += 1
    print()

    if eoc:
        report_packing(kept_groups, packed_counts, all_ids, tokenizer, eoc, model_mod.USER_OPEN)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": all_ids, "masks": all_masks, "question_offsets": all_qoffs}, out)
    n_q = sum(q is not None for q in all_qoffs)
    print(
        f"Saved {len(all_ids)} examples to {out} ({skipped} skipped — no assistant turns; "
        f"{n_q} with question offsets)"
    )


__all__ = [
    "format_conversation",
    "pack_records",
    "format_pack",
    "iter_records",
    "report_packing",
    "main",
]


if __name__ == "__main__":
    main()
