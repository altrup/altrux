"""Dreamer corpus (DISCUSSION-20261024 sec 3.3, sec 6.3).

`generate` writes the candidate pools, the abstention set and the fixed sets;
`retained-pool` runs the known-or-unknown pass (sec 3.5 item 2 (vii)) on one
checkpoint. Every row the dreamer reads is materialised here, so the box loads
nothing from HF.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import torch
from transformers import PreTrainedTokenizerBase

from experiments.lama_ckl.data import build_candidates
from experiments.lama_ckl.evaluation import Scorable, encode_metric_batch, object_token_accuracy
from experiments.lama_ckl.split import DEFAULT_WARMSTART, pinned_warmstart_sha
from preparation.conversations import format_conversation, format_pack, pack_records, render_pack
from progress import fmt_duration, heartbeat, ts

Message = dict[str, str]
Question = dict[str, str]

ACKNOWLEDGEMENT = re.compile(r"(Got it|Okay|Sure|Thanks|Understood|Noted)\b")
ABSTENTION_STEMS = (
    "I don't know",
    "I'm not sure",
    "I don't have",
    "I'm sorry, but I",
    "I cannot",
    "I can't",
)
WORD = re.compile(r"\w")
WARM_START_ROWS = 20_000
PREFIX_CAP = 2_048
PROD, GATE, HELD_OUT_FACTS, HELD_OUT_WARM_START = 20, 5, 50, 200
# The warm start's packing pass (`make warm-start`, preparation/conversations.py defaults).
PACK_SEED, PACK_REPEAT_RATE, PACK_MAX_LEN = 0, 0.3, 8192
TEXT_KEYS = frozenset({"passage", "question", "answer", "content", "reply"})


def is_acknowledgement(turn: str) -> bool:
    """The pinned rule (DISCUSSION-20261024 sec 0): the turn read to its first
    newline is one to eight words and opens on an acknowledgement stem."""
    line = turn.strip().split("\n", 1)[0]
    return 1 <= len(line.split()) <= 8 and ACKNOWLEDGEMENT.match(line) is not None


def lama_matcher(strings: Iterable[str]) -> Callable[[str], str | None]:
    """Case-insensitive, word-bounded match of any of `strings`; returns the
    first lowered string found in a text, or None."""
    by_first: dict[str, list[str]] = defaultdict(list)
    odd: list[str] = []
    for s in {s.strip().lower() for s in strings} - {""}:
        head = re.match(r"\w+", s)
        if head:
            by_first[head.group()].append(s)
        else:
            odd.append(s)
    odd_re = re.compile("|".join(rf"(?<!\w){re.escape(s)}(?!\w)" for s in odd)) if odd else None

    def match(text: str) -> str | None:
        lowered = text.lower()
        for word in re.finditer(r"\w+", lowered):
            for s in by_first.get(word.group(), ()):
                end = word.start() + len(s)
                if lowered.startswith(s, word.start()) and not WORD.match(lowered[end : end + 1]):
                    return s
        hit = odd_re.search(lowered) if odd_re else None
        return hit.group() if hit else None

    return match


def valid_boundaries(
    messages: Sequence[Message], prefix_tokens: Callable[[list[Message]], int], cap: int
) -> tuple[list[dict[str, int]], int, int]:
    """Boundaries `at` with an assistant turn before, a user follow-up at `at`
    and an assistant target after it. Returns (valid, capped, acknowledgements):
    a boundary whose prefix exceeds `cap` tokens or whose target is an
    acknowledgement is not valid."""
    valid: list[dict[str, int]] = []
    capped = acks = 0
    for at in range(2, len(messages) - 1):
        if [m["role"] for m in messages[at - 1 : at + 2]] != ["assistant", "user", "assistant"]:
            continue
        if is_acknowledgement(messages[at + 1]["content"]):
            acks += 1
            continue
        n = prefix_tokens(list(messages[:at]))
        if n > cap:
            capped += 1
            continue
        valid.append({"at": at, "prefix_tokens": n})
    return valid, capped, acks


def pair_negatives(targets: Sequence[str], rng: random.Random) -> list[int]:
    """For each row, another row whose target differs as a string (C-)."""
    pairs = []
    for i, target in enumerate(targets):
        others = [j for j, other in enumerate(targets) if j != i and other != target]
        if not others:
            raise ValueError(f"row {i} has no other row with a different target")
        pairs.append(rng.choice(others))
    return pairs


def mine_abstentions(conversations: Iterable[Sequence[Message]]) -> list[str]:
    """Assistant turns of at most 30 words opening on an abstention stem, deduplicated."""
    phrases: dict[str, None] = {}
    for messages in conversations:
        for m in messages:
            text = m["content"].strip()
            if (
                m["role"] == "assistant"
                and len(text.split()) <= 30
                and text.startswith(ABSTENTION_STEMS)
            ):
                phrases.setdefault(text)
    return list(phrases)


@dataclass
class Passage:
    pid: str
    title: str
    text: str
    answerable: list[Question] = field(default_factory=list)
    unanswerable: list[Question] = field(default_factory=list)


def partition_passages(
    passages: Sequence[Passage],
    scores: dict[str, float],
    *,
    prod: int,
    gate: int,
    held_out: int,
    abstention_rows: int,
) -> dict[str, list[Passage]]:
    """Disjoint passage sets in `passages` order: prod and gate from first-question
    accuracy 0.0, held-out facts the `held_out` highest accuracies (ties in
    order, sec 4.51), abstention passages until `abstention_rows` unanswerable
    questions, the rest with >= 2 answerable questions as the fact pool."""
    unknown = [p for p in passages if scores.get(p.pid) == 0.0]
    scored = [p for p in passages if p.pid in scores]
    best = sorted(scored, key=lambda p: -scores[p.pid])[:held_out]
    if (
        len(unknown) < prod + gate
        or len(best) < held_out
        or any(p in best for p in unknown[: prod + gate])
    ):
        raise ValueError(f"scored {len(scored)} passages, {len(unknown)} unknown")
    sets = {
        "prod": unknown[:prod],
        "gate": unknown[prod : prod + gate],
        "held_out_facts": best,
    }
    taken = {p.pid for rows in sets.values() for p in rows}
    sets["abstention"] = []
    rows = 0
    for p in passages:
        if rows >= abstention_rows:
            break
        if p.unanswerable and p.pid not in taken:
            sets["abstention"].append(p)
            rows += len(p.unanswerable)
    if rows < abstention_rows:
        raise ValueError(
            f"only {rows} unanswerable questions for {abstention_rows} abstention rows"
        )
    taken |= {p.pid for p in sets["abstention"]}
    sets["facts"] = [p for p in passages if p.pid not in taken and len(p.answerable) >= 2]
    return sets


def _row_texts(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TEXT_KEYS and isinstance(item, str):
                yield item
            else:
                yield from _row_texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _row_texts(item)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _progress(label: str, done: int, total: int, start: float, extra: str = "") -> None:
    heartbeat()
    elapsed = time.monotonic() - start
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    print(
        f"\r[{ts()}] {label} {done}/{total} {extra} {rate:.1f}/s ETA {fmt_duration(eta)}",
        end="" if done < total else "\n",
        flush=True,
    )


class Renderer:
    """`format_conversation` under the warm start's settings, `<eoc>` appended."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, model_mod: ModuleType) -> None:
        self.tokenizer = tokenizer
        self.mod = model_mod

    def ids(self, messages: Sequence[Message]) -> list[int]:
        return format_conversation(
            list(messages),
            self.tokenizer,
            10**9,
            self.mod.USER_OPEN,
            self.mod.ASST_OPEN,
            self.mod.EOC,
            train_user_turns=True,
        )[0]

    def text(self, messages: Sequence[Message]) -> str:
        return self.tokenizer.decode(self.ids(messages))


def _qa(question: str, answer: str) -> list[Message]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def _question_row(q: Question, render: Renderer) -> dict[str, object]:
    messages = _qa(q["question"], q["answer"])
    return {
        "squad_id": q["id"],
        "question": q["question"],
        "answer": q["answer"],
        "messages": messages,
        "text": render.text(messages),
    }


def _load_model(model_name: str, adapter: Path, device: torch.device) -> Scorable:
    hooks = importlib.import_module(f"models.{model_name}.train_hooks")
    config = json.loads((adapter / "lora_config.json").read_text())
    model, _ = hooks.setup_training(device, int(config["rank"]), float(config["alpha"]), 0.0)
    from training.checkpoints import load_checkpoint

    load_checkpoint(model, adapter)
    return model


def token_correctness(
    logits: torch.Tensor, input_ids: torch.Tensor, positions: Sequence[Sequence[int]]
) -> list[list[int]]:
    """Per row, the positions in `positions` whose token the argmax predicts
    (the rule of `object_token_accuracy`, kept per token)."""
    predicted = logits[:, :-1].argmax(dim=-1)
    return [
        [i for i in tokens if int(predicted[row, i - 1]) == int(input_ids[row, i])]
        for row, tokens in enumerate(positions)
    ]


def score_questions(
    model: Scorable,
    render: Renderer,
    questions: Sequence[Question],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    """Teacher-forced answer accuracy of `[USER] q [ASSISTANT] a`, no passage,
    by the LAMA scorer (sec 3.5 item 2 (vii)), with the answer positions and
    the correctly predicted ones. Positions index `format_conversation`'s ids
    of the same messages."""
    user, asst = render.mod.USER_OPEN, render.mod.ASST_OPEN
    rows = [
        {"task": f"{user} {q['question']}\n{asst} {q['answer']}", "object": q["answer"]}
        for q in questions
    ]
    seams = sum(
        list(render.tokenizer(row["task"], add_special_tokens=True)["input_ids"])
        != render.ids(_qa(q["question"], q["answer"]))[:-2]
        for row, q in zip(rows, questions, strict=True)
    )
    print(f"[{ts()}] scorer ids != format_conversation ids: {seams}  (must be 0)")
    results: list[dict[str, object]] = []
    disagreements = 0
    start = time.monotonic()
    model.eval()
    with torch.no_grad():
        for at in range(0, len(rows), batch_size):
            batch = rows[at : at + batch_size]
            ids, positions = encode_metric_batch(render.tokenizer, batch, "task", 512, device)
            logits, _ = model(ids, state=None)
            accuracies = object_token_accuracy(logits, ids, positions)
            for tokens, correct, accuracy in zip(
                positions, token_correctness(logits, ids, positions), accuracies, strict=True
            ):
                disagreements += len(correct) / len(tokens) != accuracy
                results.append(
                    {"accuracy": accuracy, "answer_positions": tokens, "correct_positions": correct}
                )
            done = [float(r["accuracy"]) for r in results]
            _progress(
                "scored",
                len(results),
                len(rows),
                start,
                f"zeros={done.count(0.0)} ones={done.count(1.0)}",
            )
    print(f"[{ts()}] per-token correctness != scorer accuracy: {disagreements}  (must be 0)")
    return results


def load_passages(match: Callable[[str], str | None]) -> tuple[list[Passage], Counter[str]]:
    from datasets import load_dataset

    print(f"[{ts()}] loading rajpurkar/squad_v2 train")
    by_text: dict[str, Passage] = {}
    for row in load_dataset("rajpurkar/squad_v2", split="train"):
        text = row["context"]
        if text not in by_text:
            pid = hashlib.sha256(text.encode()).hexdigest()[:12]
            by_text[text] = Passage(pid=pid, title=row["title"], text=text)
        q = {"id": row["id"], "question": row["question"].strip()}
        if row["answers"]["text"]:
            by_text[text].answerable.append({**q, "answer": row["answers"]["text"][0].strip()})
        else:
            by_text[text].unanswerable.append(q)
    counts: Counter[str] = Counter()
    clean: list[Passage] = []
    for p in by_text.values():
        counts["passages"] += 1
        counts["answerable"] += len(p.answerable)
        counts["unanswerable"] += len(p.unanswerable)
        if match(p.text):
            counts["passages_excluded"] += 1
            counts["answerable_excluded_with_passage"] += len(p.answerable)
            counts["unanswerable_excluded_with_passage"] += len(p.unanswerable)
            continue
        kept = [q for q in p.answerable if not match(q["question"])]
        dropped = [q for q in p.unanswerable if match(q["question"])]
        counts["answerable_excluded_by_question"] += len(p.answerable) - len(kept)
        counts["unanswerable_excluded_by_question"] += len(dropped)
        p.answerable = kept
        p.unanswerable = [q for q in p.unanswerable if q not in dropped]
        clean.append(p)
    return clean, counts


def packer_dropped(
    ultrachat: Sequence[dict[str, object]], dolly_path: Path, render: Renderer
) -> set[int]:
    """UltraChat indices the warm start's seeded packing pass dropped whole
    (`format_pack` ran out of budget before their slot)."""
    records = [{**row, "ultrachat_index": i} for i, row in enumerate(ultrachat)]
    records += [json.loads(line) for line in dolly_path.read_text().splitlines() if line.strip()]
    rng = random.Random(PACK_SEED)
    rng.shuffle(records)
    groups = pack_records(records, rng, PACK_REPEAT_RATE)
    rendered = [render_pack(group, rng) for group in groups]
    dropped: set[int] = set()
    start = time.monotonic()
    for n, (group, slots) in enumerate(zip(groups, rendered, strict=True), 1):
        *_, packed = format_pack(
            slots,
            render.tokenizer,
            PACK_MAX_LEN,
            render.mod.USER_OPEN,
            render.mod.ASST_OPEN,
            render.mod.EOC,
            train_user_turns=True,
        )
        dropped.update(
            int(record["ultrachat_index"])
            for kind, record in group[packed:]
            if kind == "fresh" and "ultrachat_index" in record
        )
        if n % 500 == 0 or n == len(groups):
            _progress("packer replay", n, len(groups), start, f"dropped={len(dropped)}")
    return dropped


def _sample_lines(kind: str, rows: Sequence[dict[str, object]], render: Renderer) -> None:
    """Five decoded rows of `kind`, C- beside C where the kind has one."""
    rng = random.Random(1)
    picked = rng.sample(list(rows), min(5, len(rows)))
    print(f"[{ts()}] ===== {kind}: {len(rows)} rows, {len(picked)} decoded =====")
    if kind == "conversation":
        drawn = [(row, rng.choice(row["boundaries"])["at"]) for row in picked]
        replies = [row["messages"][at + 1]["content"] for row, at in drawn]
        for (row, at), j in zip(drawn, pair_negatives(replies, rng), strict=True):
            follow = row["messages"][at]
            print(f"--- ultrachat {row['ultrachat_index']} boundary {at}")
            print(f"W (tail): ...{render.text(row['messages'][:at])[-400:]!r}")
            print(f"C : {render.text([follow, row['messages'][at + 1]])[:600]!r}")
            print(
                f"C-: {render.text([follow, {'role': 'assistant', 'content': replies[j]}])[:600]!r}"
            )
        return
    if kind in ("fact", "prod", "gate", "held_out_facts"):
        drawn = [
            (row, rng.choice(row["questions"]) if kind == "fact" else row["question"])
            for row in picked
        ]
        answers = [q["answer"] for _, q in drawn]
        negatives = pair_negatives(answers, rng) if kind == "fact" else [None] * len(drawn)
        for (row, q), j in zip(drawn, negatives, strict=True):
            print(f"--- passage {row['passage_id']} ({row['title']}) squad {q['squad_id']}")
            print(f"W : {row['passage'][:400]!r}")
            print(f"C : {q['text']!r}")
            if j is not None:
                print(f"C-: {render.text(_qa(q['question'], answers[j]))!r}")
            elif "negative" in row:
                print(f"C-: {row['negative']['text']!r}")
        return
    for row in picked:
        source = {k: row[k] for k in ("squad_id", "ultrachat_index", "dolly_line") if k in row}
        print(f"--- {source}")
        print(f"   {row['text'][:600]!r}")


def generate(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    if not torch.cuda.is_available():
        raise SystemExit("the v4 known-or-unknown pass for the fixed sets requires a GPU")
    adapter_sha = _sha256(args.init_adapter / "trainable.pt")
    if adapter_sha != pinned_warmstart_sha():
        raise SystemExit(f"{args.init_adapter} is not the pinned warm start (v4)")
    rng = random.Random(args.seed)
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    lama = [s for row in build_candidates(args.lama_root) for s in (row["subject"], row["object"])]
    match = lama_matcher(lama)
    print(f"[{ts()}] LAMA strings: {len(set(lama))} from {args.lama_root}")

    model_mod = importlib.import_module(f"models.{args.model_name}")
    from models.common import build_tokenizer

    render = Renderer(build_tokenizer(model_mod), model_mod)

    print(f"[{ts()}] loading HuggingFaceH4/ultrachat_200k train_sft")
    ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    trained = ultrachat.select(range(WARM_START_ROWS)).to_list()

    mined = mine_abstentions(row["messages"] for row in trained)
    phrases = [p for p in mined if not match(p)]
    by_stem = Counter(next(s for s in ABSTENTION_STEMS if p.startswith(s)) for p in phrases)
    print(
        f"[{ts()}] abstention set: {len(mined)} mined, {len(mined) - len(phrases)} LAMA-excluded, "
        f"{len(phrases)} kept{'  (under 20: a reading before round 0)' if len(phrases) < 20 else ''}"
    )
    print(f"[{ts()}]   by stem: {dict(by_stem)}")
    for phrase in phrases:
        print(f"    {phrase!r}")

    passages, squad_counts = load_passages(match)
    rng.shuffle(passages)
    eligible = [p for p in passages if len(p.answerable) >= 2]
    print(f"[{ts()}] SQuAD: {dict(squad_counts)}")
    print(
        f"[{ts()}]   LAMA-clean passages {len(passages)}/{squad_counts['passages']} "
        f"({len(passages) / squad_counts['passages']:.1%}), with >= 2 answerable {len(eligible)}"
    )

    device = torch.device("cuda")
    model = _load_model(args.model_name, args.init_adapter, device)
    scan = dict(
        zip(
            (p.pid for p in eligible),
            score_questions(
                model, render, [p.answerable[0] for p in eligible], args.batch_size, device
            ),
            strict=True,
        )
    )
    scores = {pid: float(r["accuracy"]) for pid, r in scan.items()}
    print(
        f"[{ts()}] v4 first-question scan: {len(scores)} scored, accuracy histogram "
        f"{dict(sorted(Counter(f'{a:.1f}' for a in scores.values()).items()))}"
    )
    del model
    torch.cuda.empty_cache()
    sets = partition_passages(
        passages,
        scores,
        prod=PROD,
        gate=GATE,
        held_out=HELD_OUT_FACTS,
        abstention_rows=args.abstention_rows,
    )
    held = [scores[p.pid] for p in sets["held_out_facts"]]
    print(f"[{ts()}] held-out v4-best-known accuracy range {min(held):.3f}..{max(held):.3f}")

    corpus: list[dict[str, object]] = []
    fixed: dict[str, list[dict[str, object]]] = {}
    fixed_facts = [(name, p) for name in ("prod", "gate", "held_out_facts") for p in sets[name]]
    fixed_answers = [p.answerable[0]["answer"] for _, p in fixed_facts]
    for (name, p), j in zip(fixed_facts, pair_negatives(fixed_answers, rng), strict=True):
        q = p.answerable[0]
        row: dict[str, object] = {
            "kind": name,
            "passage_id": p.pid,
            "title": p.title,
            "passage": p.text,
            "question": _question_row(q, render),
            "v4_accuracy": scores[p.pid],
            "v4_correct_positions": scan[p.pid]["correct_positions"],
        }
        if name != "held_out_facts":
            negative = _qa(q["question"], fixed_answers[j])
            row["negative"] = {"messages": negative, "text": render.text(negative)}
        fixed.setdefault(name, []).append(row)

    facts = [
        {
            "kind": "fact",
            "passage_id": p.pid,
            "title": p.title,
            "passage": p.text,
            "questions": [_question_row(q, render) for q in p.answerable],
        }
        for p in sets["facts"]
    ]
    pool_questions = [q["question"] for row in facts for q in row["questions"]]
    unbound = sum(not any(w[:1].isupper() for w in q.split()[1:]) for q in pool_questions)
    print(
        f"[{ts()}] fact pool: {len(facts)} passages, {len(pool_questions)} answerable questions; "
        f"questions with no capitalised word after the first: {unbound / len(pool_questions):.1%}"
    )

    abstentions = []
    for p in sets["abstention"]:
        for q in p.unanswerable:
            reply = rng.choice(phrases)
            messages = _qa(q["question"], reply)
            abstentions.append(
                {
                    "kind": "abstention",
                    "squad_id": q["id"],
                    "passage_id": p.pid,
                    "question": q["question"],
                    "reply": reply,
                    "messages": messages,
                    "text": render.text(messages),
                }
            )

    def prefix_tokens(prefix: list[Message]) -> int:
        return len(render.ids(prefix))

    order = list(range(WARM_START_ROWS, len(ultrachat)))
    rng.shuffle(order)
    conversations: list[dict[str, object]] = []
    conv_counts: Counter[str] = Counter()
    start = time.monotonic()
    for index in order:
        if len(conversations) >= args.conversations:
            break
        row = ultrachat[index]
        conv_counts["read"] += 1
        if any(match(m["content"]) for m in row["messages"]):
            conv_counts["lama_excluded"] += 1
            continue
        valid, capped, acks = valid_boundaries(row["messages"], prefix_tokens, PREFIX_CAP)
        conv_counts["boundaries_capped"] += capped
        conv_counts["boundaries_acknowledgement"] += acks
        if not valid:
            conv_counts["no_valid_boundary"] += 1
            continue
        conversations.append(
            {
                "kind": "conversation",
                "ultrachat_index": index,
                "prompt_id": row["prompt_id"],
                "messages": row["messages"],
                "boundaries": valid,
                "text": render.text(row["messages"]),
            }
        )
        if len(conversations) % 50 == 0:
            _progress(
                "conversation candidates",
                len(conversations),
                args.conversations,
                start,
                str(dict(conv_counts)),
            )
    print(
        f"[{ts()}] conversation candidates: {len(conversations)} kept of {conv_counts['read']} read "
        f"({len(conversations) / conv_counts['read']:.1%}); {dict(conv_counts)}; "
        f"valid boundaries {sum(len(c['boundaries']) for c in conversations)}"
    )

    dropped = packer_dropped(trained, args.dolly, render)
    warm = [
        {
            "kind": "warm_start",
            "ultrachat_index": i,
            "prompt_id": row["prompt_id"],
            "messages": row["messages"],
            "text": render.text(row["messages"]),
            "packer_dropped": i in dropped,
        }
        for i, row in enumerate(trained)
        if not any(match(m["content"]) for m in row["messages"])
    ]
    rng.shuffle(warm)
    held_warm, warm = warm[:HELD_OUT_WARM_START], warm[HELD_OUT_WARM_START:]
    for row in held_warm:
        row["kind"] = "held_out_warm_start"
    fixed["held_out_warm_start"] = held_warm
    print(
        f"[{ts()}] warm-start UltraChat: {len(warm) + len(held_warm)}/{WARM_START_ROWS} LAMA-clean "
        f"({(len(warm) + len(held_warm)) / WARM_START_ROWS:.1%}); packer dropped whole "
        f"{len(dropped)} of all {WARM_START_ROWS}, {sum(r['packer_dropped'] for r in warm)} of the "
        f"{len(warm)} pool rows, {sum(r['packer_dropped'] for r in held_warm)} of the "
        f"{len(held_warm)} held-out rows"
    )

    dolly_lines = [json.loads(line) for line in args.dolly.read_text().splitlines() if line.strip()]
    plain = [(i, rec) for i, rec in enumerate(dolly_lines) if rec["dolly"]["shape"] == "plain"]
    dolly = [
        {
            "kind": "dolly_plain",
            "dolly_line": i,
            "messages": rec["messages"],
            "text": render.text(rec["messages"]),
        }
        for i, rec in plain
        if not any(match(m["content"]) for m in rec["messages"])
    ]
    print(
        f"[{ts()}] Dolly plain: {len(dolly)}/{len(plain)} LAMA-clean ({len(dolly) / len(plain):.1%})"
    )
    print(
        f"[{ts()}] abstention rows: {len(abstentions)} from {len(sets['abstention'])} passages, "
        f"{len(Counter(r['reply'] for r in abstentions))} distinct replies"
    )

    corpus = [
        *facts,
        *conversations,
        *abstentions,
        *dolly,
        *warm,
        *(row for rows in fixed.values() for row in rows),
    ]
    overlaps = sum(any(match(t) for t in _row_texts(row)) for row in corpus)
    print(f"[{ts()}] LAMA overlap over the written rows: {overlaps}  (must be 0)")
    assert overlaps == 0, "a written row carries a LAMA subject or object"
    by_pid: dict[str, set[str]] = defaultdict(set)
    for row in corpus:
        if "passage_id" in row:
            by_pid[str(row["passage_id"])].add(str(row["kind"]))
    shared = sum(len(kinds) > 1 for kinds in by_pid.values())
    print(f"[{ts()}] passages shared across kinds: {shared}  (must be 0)")
    assert shared == 0

    for kind in ("fact", "conversation", "abstention", "dolly_plain", "warm_start"):
        _sample_lines(kind, [r for r in corpus if r["kind"] == kind], render)
    for name, rows in fixed.items():
        _sample_lines(name, rows, render)

    meta = {"seed": args.seed, "warmstart_sha256": adapter_sha, "lama_root": str(args.lama_root)}
    (out / "dreamer_corpus.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in corpus)
    )
    (out / "dreamer_abstentions.json").write_text(
        json.dumps({"stems": ABSTENTION_STEMS, "mined": len(mined), "phrases": phrases}, indent=1)
    )
    (out / "dreamer_fixed_sets.json").write_text(json.dumps({**meta, **fixed}, indent=1))
    print(
        f"[{ts()}] wrote {len(corpus)} rows: {dict(Counter(str(r['kind']) for r in corpus))} to {out}"
    )


def retained_pool(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("the retained-pool pass requires a GPU")
    sha = _sha256(args.init_adapter / "trainable.pt")
    output = args.output or args.corpus.parent / f"retained_pool-{sha[:8]}.json"
    exclude = set(json.loads(args.exclude_passages.read_text())) if args.exclude_passages else set()
    questions: list[Question] = []
    for line in args.corpus.read_text().splitlines():
        row = json.loads(line)
        if row["kind"] == "fact" and row["passage_id"] not in exclude:
            questions += [
                {"id": q["squad_id"], "passage_id": row["passage_id"], **q}
                for q in row["questions"]
            ]
    drawn = random.Random(int(sha[:8], 16)).sample(questions, min(args.size, len(questions)))
    print(
        f"[{ts()}] {len(drawn)} drawn of {len(questions)} pool questions, {len(exclude)} passages excluded"
    )

    model_mod = importlib.import_module(f"models.{args.model_name}")
    from models.common import build_tokenizer

    render = Renderer(build_tokenizer(model_mod), model_mod)
    device = torch.device("cuda")
    model = _load_model(args.model_name, args.init_adapter, device)
    start = time.monotonic()
    scored = score_questions(model, render, drawn, args.batch_size, device)
    elapsed = time.monotonic() - start
    rows = [
        {
            "passage_id": q["passage_id"],
            "squad_id": q["id"],
            "question": q["question"],
            "answer": q["answer"],
            **result,
            "side": "unknown" if result["accuracy"] == 0.0 else "known",
        }
        for q, result in zip(drawn, scored, strict=True)
    ]
    sides = Counter(r["side"] for r in rows)
    histogram = Counter(f"{r['accuracy']:.1f}" for r in rows)
    correct = Counter(len(r["correct_positions"]) for r in rows if r["side"] == "known")
    print(f"[{ts()}] {len(rows)} scored in {fmt_duration(elapsed)} ({len(rows) / elapsed:.1f} q/s)")
    print(f"[{ts()}] sides {dict(sides)}; accuracy histogram {dict(sorted(histogram.items()))}")
    print(f"[{ts()}] known rows by correct answer-token count: {dict(sorted(correct.items()))}")
    for side in ("known", "unknown"):
        for r in [r for r in rows if r["side"] == side][:5]:
            ids = render.ids(_qa(r["question"], r["answer"]))
            marked = [
                (render.tokenizer.decode([ids[i]]), i in r["correct_positions"])
                for i in r["answer_positions"]
            ]
            print(f"    {side} {r['accuracy']:.2f}  {r['question']!r} -> {r['answer']!r}  {marked}")
    Path(output).write_text(
        json.dumps(
            {
                "checkpoint": str(args.init_adapter),
                "trainable_sha256": sha,
                "corpus": str(args.corpus),
                "sides": dict(sides),
                "rows": rows,
            },
            indent=1,
        )
    )
    print(f"[{ts()}] wrote {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="write the corpus, abstention set and fixed sets")
    gen.add_argument(
        "--lama-root", type=Path, required=True, help="the LAMA root the v4 split was built from"
    )
    gen.add_argument(
        "--init-adapter", type=Path, default=DEFAULT_WARMSTART, help="v4 (must match the pin)"
    )
    gen.add_argument("--dolly", type=Path, default=Path("data/dolly_context.jsonl"))
    gen.add_argument("--out-dir", type=Path, default=Path("data"))
    gen.add_argument("--conversations", type=int, default=1000, help="conversation candidates kept")
    gen.add_argument("--abstention-rows", type=int, default=200)
    gen.add_argument("--seed", type=int, default=0)
    gen.set_defaults(func=generate)
    pool = sub.add_parser("retained-pool", help="known-or-unknown pass on one checkpoint")
    pool.add_argument("--init-adapter", type=Path, required=True)
    pool.add_argument("--corpus", type=Path, default=Path("data/dreamer_corpus.jsonl"))
    pool.add_argument("--output", type=Path, help="default data/retained_pool-<sha8>.json")
    pool.add_argument("--size", type=int, default=2000)
    pool.add_argument(
        "--exclude-passages", type=Path, help="JSON list of passage ids (earlier rounds' P)"
    )
    pool.set_defaults(func=retained_pool)
    for p in (gen, pool):
        p.add_argument("--model-name", default="mamba2_2_7b")
        p.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
