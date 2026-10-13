import argparse
import importlib
import json
import multiprocessing
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

# Add the repo root to sys.path so the models/ package is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.common import build_tokenizer

from preparation import dolly
from progress import ts

_worker_tokenizer = None
_worker_max_len = None
_worker_markers: tuple[str, str] | None = None
_worker_eoc: str | None = None
_worker_train_user_turns = False


def _worker_init(model_name: str, max_len: int, train_user_turns: bool) -> None:
    global _worker_tokenizer, _worker_max_len, _worker_markers, _worker_eoc
    global _worker_train_user_turns
    # Imported here, not at module top: importing a model package pulls in
    # mamba_ssm, which needs a working GPU even to import -- keeping it lazy
    # keeps format_conversation importable (and testable) without one.
    mod = importlib.import_module(f"models.{model_name}")
    _worker_tokenizer = build_tokenizer(mod)
    _worker_max_len = max_len
    _worker_markers = (mod.USER_OPEN, mod.ASST_OPEN)
    _worker_eoc = getattr(mod, "EOC", None)
    _worker_train_user_turns = train_user_turns


def _worker_format(
    group: list[tuple[str, list[dict]]],
) -> tuple[list[int], list[bool], int | None, int]:
    return format_pack(
        group,
        _worker_tokenizer,
        _worker_max_len,
        *_worker_markers,
        _worker_eoc,
        train_user_turns=_worker_train_user_turns,
    )


def format_conversation(
    messages: list[dict],
    tokenizer,
    max_len: int,
    user_open: str,
    asst_open: str,
    eoc: str | None = None,
    *,
    train_user_turns: bool = False,
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

    `train_user_turns` puts user turns under the loss too, so the model learns
    to open and write a `[USER]` turn (the dream prompt asks it to write both
    sides; DISCUSSION-20261011 sec 3.1).
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
            turn_mask = [train_user_turns] * len(turn_ids)
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


def _repeatable(record: dict) -> bool:
    """Dolly records repeat as a variant; UltraChat needs >= 6 turns to split."""
    return "dolly" in record or len(record.get("messages", [])) >= 6


def pack_records(
    records: list[dict], rng: random.Random, repeat_rate: float
) -> list[list[tuple[str, dict]]]:
    """Group records into examples of 2-5, each element tagged `fresh` (the
    next pool record) or `repeat` (the same record as an earlier fresh slot of
    the pack, never repeated twice; only `_repeatable` records are sources). A
    repeat does not consume a pool record; render_pack decides what each slot's
    messages are."""
    groups: list[list[tuple[str, dict]]] = []
    pending = records
    at = 0
    while at < len(pending):
        size = rng.randint(2, 5)
        group: list[tuple[str, dict]] = [("fresh", pending[at])]
        at += 1
        unrepeated = [pending[at - 1]] if _repeatable(pending[at - 1]) else []
        while len(group) < size:
            if unrepeated and rng.random() < repeat_rate:
                group.append(("repeat", unrepeated.pop(rng.randrange(len(unrepeated)))))
            elif at < len(pending):
                group.append(("fresh", pending[at]))
                if _repeatable(pending[at]):
                    unrepeated.append(pending[at])
                at += 1
            else:
                break
        groups.append(group)
    return groups


def _question(row: dict) -> list[dict]:
    return [
        {"role": "user", "content": row["instruction"]},
        {"role": "assistant", "content": row["response"]},
    ]


def _recall(row: dict, rng: random.Random) -> list[dict]:
    if row.get("summary"):
        request, answer = rng.choice(dolly.RECALL_GIST_REQUESTS), row["summary"]
    else:
        request, answer = rng.choice(dolly.RECALL_REQUESTS), row["context"]
    return [{"role": "user", "content": request}, {"role": "assistant", "content": answer}]


def _dolly_repeat(record: dict, rng: random.Random) -> tuple[str, list[dict]]:
    """A second occurrence drawn uniformly over the variants available to the
    record (DISCUSSION-20261013 sec 3.1)."""
    shape, rows = record["dolly"]["shape"], record["dolly"]["rows"]
    variants = ["verbatim", "question", "recall", "multi-turn"]
    if shape != "plain" and any(row.get("summary") for row in rows):
        variants.append("summary")
    variant = rng.choice(variants)
    if variant == "verbatim":
        return variant, record["messages"]
    if variant == "question":
        return variant, [m for row in rows for m in _question(row)]
    if variant == "recall":
        return variant, [m for row in rows for m in _recall(row, rng)]
    if variant == "summary":
        return variant, [
            m
            for row in rows
            for m in dolly.render_document(
                {**row, "context": row["summary"]} if row.get("summary") else row, rng
            )
        ]
    if len(rows) == 1:
        row = rows[0]
        exchanges = [dolly.render_document(row, rng), _question(row)]
        if row.get("summary"):
            exchanges.append(_recall(row, rng))
    else:
        exchanges = [
            rng.choice((dolly.render_document, lambda r, _: _question(r), _recall))(row, rng)
            for row in rng.sample(rows, min(3, len(rows)))
        ]
    rng.shuffle(exchanges)
    return variant, [m for exchange in exchanges for m in exchange]


def render_pack(group: list[tuple[str, dict]], rng: random.Random) -> list[tuple[str, list[dict]]]:
    """Resolve each slot of a pack_records group to (label, messages). A fresh
    slot is labelled `fresh`; a Dolly repeat by its variant; a repeated
    UltraChat conversation of >= 6 turns is split at an exchange boundary k,
    its fresh slot is turns 1..k and its repeat replays the last j of those and
    continues (`ultrachat k=<k> j=<j>`)."""
    cuts: dict[int, int] = {}
    rendered: list[tuple[str, list[dict]]] = []
    for kind, record in group:
        messages = record.get("messages", [])
        if kind == "fresh":
            splittable = "dolly" not in record and len(messages) >= 6
            if splittable and any(k == "repeat" and r is record for k, r in group):
                cuts[id(record)] = rng.randrange(2, len(messages) - 1, 2)
                messages = messages[: cuts[id(record)]]
            rendered.append(("fresh", messages))
        elif "dolly" in record:
            rendered.append(_dolly_repeat(record, rng))
        else:
            k = cuts[id(record)]
            j = rng.randrange(0, k + 1, 2)
            rendered.append((f"ultrachat k={k} j={j}", messages[k - j :]))
    return rendered


def format_pack(
    group: list[tuple[str, list[dict]]],
    tokenizer,
    max_len: int,
    user_open: str,
    asst_open: str,
    eoc: str | None,
    *,
    train_user_turns: bool = False,
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
            messages,
            tokenizer,
            max_len - len(ids),
            user_open,
            asst_open,
            eoc,
            train_user_turns=train_user_turns,
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
    groups, rendered, packed_counts, all_ids, tokenizer, eoc: str | None, user_open: str
) -> None:
    """Structural invariants whose correct value is zero, plus decoded text
    either side of one boundary of each kind (root CLAUDE.md: counts confirm
    the generator did what it was told, never that what it was told was right).
    `groups` is pack_records' structure, `rendered` render_pack's output."""
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

    def source(group, j: int) -> int | None:
        record = group[j][1]
        return next((k for k in range(j) if group[k][0] == "fresh" and group[k][1] is record), None)

    def label(group, j: int) -> str:
        if group[j][0] == "fresh":
            return "fresh"
        src = source(group, j)
        if src is None:
            return "orphan"
        return "adjacent" if src == j - 1 else "gapped"

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

    slots = [
        (label, messages)
        for group, packed in zip(rendered, packed_counts, strict=True)
        for label, messages in group[:packed]
    ]
    roles = [[m["role"] for m in messages] for _, messages in slots]
    unanswered = sum(
        1
        for r in roles
        for i, role in enumerate(r)
        if role == "user" and (i + 1 == len(r) or r[i + 1] != "assistant")
    )
    adjacent_roles = sum(1 for r in roles for a, b in zip(r, r[1:]) if a == b)
    bad_replays = sum(
        1
        for (label, _), r in zip(slots, roles, strict=True)
        if label.startswith("ultrachat k=") and r[:1] != ["user"]
    )
    shapes = Counter(
        record["dolly"]["shape"] if "dolly" in record else "ultrachat"
        for group in groups
        for kind, record in group
        if kind == "fresh"
    )
    variants = Counter(label for label, _ in slots if label != "fresh")
    print(f"[{ts()}]   user turns not answered next      : {unanswered}  (must be 0)")
    print(f"[{ts()}]   adjacent same-role turns          : {adjacent_roles}  (must be 0)")
    print(f"[{ts()}]   split replays not opening on user : {bad_replays}  (must be 0)")
    print(f"[{ts()}]   first occurrences by shape: {dict(sorted(shapes.items()))}")
    print(f"[{ts()}]   repeats by variant: {dict(sorted(variants.items()))}")

    def boundary(ids, j: int) -> int:
        return [i for i in range(len(ids)) if int(ids[i]) == eoc_id][j - 1]

    def kind_of(label: str) -> str:
        return "ultrachat split" if label.startswith("ultrachat k=") else label

    shown: set[str] = set()
    for group, rgroup, packed, ids in zip(groups, rendered, packed_counts, all_ids, strict=True):
        for j in range(1, packed):
            want = kind_of(rgroup[j][0])
            if want == "fresh" or want in shown:
                continue
            shown.add(want)
            at = boundary(ids, j)
            head = ""
            src = source(group, j)
            if want == "ultrachat split" and src is not None and src + 1 != j:
                end = boundary(ids, src + 1)
                tail = tokenizer.decode(ids[max(0, end - 60) : end])
                head = f"    end of the k-turn first occurrence: ...{tail!r}\n"
            print(
                f"[{ts()}]   sample of a {rgroup[j][0]} repeat:\n{head}"
                f"    ...{tokenizer.decode(ids[max(0, at - 60) : at])!r} "
                f">>{eoc}>> {tokenizer.decode(ids[at + 1 : at + 60])!r}..."
            )

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
        at = boundary(ids, j)
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
        help="Probability that a packed slot after the first repeats an earlier, not yet "
        "repeated conversation of the same pack: a Dolly record as a drawn variant "
        "(verbatim, question, summary, recall, multi-turn), an UltraChat conversation of "
        ">= 6 turns as a (k, j) split and replay; shorter ones never repeat (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Pool shuffle and packing RNG seed (default: %(default)s)",
    )
    parser.add_argument(
        "--train-user-turns",
        action="store_true",
        help="Put user turns under the loss as well as assistant turns",
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

    rng = random.Random(args.seed)
    groups = (
        pack_records(records, rng, args.repeat_rate)
        if args.pack
        else [[("fresh", r)] for r in records]
    )
    rendered = [render_pack(group, rng) for group in groups]
    kept_groups: list[list[tuple[str, dict]]] = []
    kept_rendered: list[list[tuple[str, list[dict]]]] = []
    packed_counts: list[int] = []
    tokenizer = build_tokenizer(model_mod)

    def formatted():
        if args.workers > 1:
            with multiprocessing.Pool(
                args.workers,
                initializer=_worker_init,
                initargs=(model_name, args.max_len, args.train_user_turns),
            ) as pool:
                yield from pool.imap(_worker_format, rendered)
        else:
            for group in rendered:
                yield format_pack(
                    group,
                    tokenizer,
                    args.max_len,
                    model_mod.USER_OPEN,
                    model_mod.ASST_OPEN,
                    eoc,
                    train_user_turns=args.train_user_turns,
                )

    for i, (ids, mask, qoff, packed) in enumerate(formatted()):
        print(f"\r[{ts()}] {i + 1}/{len(groups)}", end="", flush=True)
        if keep(ids, mask, qoff):
            kept_groups.append(groups[i])
            kept_rendered.append(rendered[i])
            packed_counts.append(packed)
        else:
            skipped += 1
    print()

    if eoc:
        report_packing(
            kept_groups,
            kept_rendered,
            packed_counts,
            all_ids,
            tokenizer,
            eoc,
            model_mod.USER_OPEN,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": all_ids, "masks": all_masks, "question_offsets": all_qoffs}, out)
    n_q = sum(q is not None for q in all_qoffs)
    print(
        f"Saved {len(all_ids)} examples to {out} ({skipped} skipped — nothing trainable; "
        f"{n_q} with question offsets)"
    )


__all__ = [
    "format_conversation",
    "pack_records",
    "render_pack",
    "format_pack",
    "iter_records",
    "report_packing",
    "main",
]


if __name__ == "__main__":
    main()
