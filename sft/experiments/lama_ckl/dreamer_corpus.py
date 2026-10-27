"""Dreamer corpus (DISCUSSION-20261024 sec 3.3, sec 6.3).

`generate` writes the candidate pools, the abstention set and the fixed sets;
`retained-pool` runs the generate-and-match known-or-unknown pass (sec 3.5 item 2 (vii)) on one
checkpoint. Every row the dreamer reads is materialised here, so the box loads
nothing from HF.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import random
import re
import string
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import cache
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
CLASSIFIER = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
HYPOTHESES = {
    "The speaker does not know the answer": "unknown",
    "The speaker cannot perform the action": "cannot",
    "The speaker answers the question": "answers",
}
REPLY_CAP = 48
ABSTENTION_WORDS = 30
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


def lama_matcher(pairs: Iterable[tuple[str, str]]) -> Callable[[str], tuple[str, str] | None]:
    """Case-insensitive, word-bounded match of a (subject, object) pair, both
    present in a text; returns the first lowered pair found, or None."""
    partners: dict[str, set[str]] = defaultdict(set)
    for subject, obj in pairs:
        subject, obj = subject.strip().lower(), obj.strip().lower()
        if subject and obj:
            partners[subject].add(obj)
    strings = set(partners) | {o for objs in partners.values() for o in objs}
    by_first: dict[str, list[str]] = defaultdict(list)
    odd: list[str] = []
    for s in strings:
        head = re.match(r"\w+", s)
        if head:
            by_first[head.group()].append(s)
        else:
            odd.append(s)
    odd_re = re.compile("|".join(rf"(?<!\w){re.escape(s)}(?!\w)" for s in odd)) if odd else None

    def match(text: str) -> tuple[str, str] | None:
        lowered = text.lower()
        found: set[str] = set()
        for word in re.finditer(r"\w+", lowered):
            for s in by_first.get(word.group(), ()):
                end = word.start() + len(s)
                if lowered.startswith(s, word.start()) and not WORD.match(lowered[end : end + 1]):
                    found.add(s)
        if odd_re:
            found.update(hit.group() for hit in odd_re.finditer(lowered))
        for subject in sorted(found & partners.keys()):
            hits = partners[subject] & found
            if hits:
                return subject, min(hits)
        return None

    return match


def normalize_answer(text: str) -> str:
    """SQuAD's exact-match normalisation: lowercase, strip punctuation and
    articles, collapse whitespace."""
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_in_reply(answer: str, reply: str) -> bool:
    """The generate-and-match rule (sec 4.56): the normalised answer occurs in
    the normalised reply on word boundaries."""
    normalised = normalize_answer(answer)
    # Word-bounded so "art" does not match inside "party".
    return bool(normalised) and f" {normalised} " in f" {normalize_answer(reply)} "


@cache
def _classifier(device: str) -> Callable[..., list[dict[str, list]]]:
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=CLASSIFIER, device=device)


def abstention_labels(
    replies: Sequence[str], device: str | torch.device, batch_size: int = 32
) -> list[str]:
    """Per reply, "unknown", "cannot" or "answers": the top entailment of the
    three hypotheses (sec 3.3, sec 4.56), no threshold."""
    if not replies:
        return []
    results = _classifier(str(device))(
        list(replies),
        candidate_labels=list(HYPOTHESES),
        hypothesis_template="{}",
        multi_label=False,
        batch_size=batch_size,
    )
    return [HYPOTHESES[r["labels"][0]] for r in results]


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


@dataclass
class Passage:
    pid: str
    title: str
    text: str
    answerable: list[Question] = field(default_factory=list)
    unanswerable: list[Question] = field(default_factory=list)


def partition_passages(
    passages: Sequence[Passage],
    known: dict[str, bool],
    *,
    prod: int,
    gate: int,
    held_out: int,
    abstention_rows: int,
    rng: random.Random,
    gate_pids: Sequence[str] = (),
) -> dict[str, list[Passage]]:
    """Disjoint passage sets: prod and gate from the first-question-unknown
    passages in `passages` order (gate = `gate_pids` when given), held-out facts
    a seeded draw of `held_out` known passages (sec 4.56), abstention passages
    until `abstention_rows` unanswerable questions, the rest with >= 2
    answerable questions as the fact pool."""
    by_pid = {p.pid: p for p in passages}
    gate_set = [by_pid[pid] for pid in gate_pids]
    unknown = [p for p in passages if known.get(p.pid) is False and p not in gate_set]
    if not gate_set:
        gate_set, unknown = unknown[prod : prod + gate], unknown[:prod] + unknown[prod + gate :]
    best = [p for p in passages if known.get(p.pid)]
    if len(unknown) < prod or len(gate_set) != gate or len(best) < held_out:
        raise ValueError(f"{len(known)} scored passages, {len(unknown)} unknown, {len(best)} known")
    if any(known.get(p.pid) is not False for p in gate_set):
        raise ValueError("a gate passage is not unknown")
    sets = {
        "prod": unknown[:prod],
        "gate": gate_set,
        "held_out_facts": rng.sample(best, held_out),
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


def _load_model(
    model_name: str, adapter: Path, device: torch.device, fp32: bool = False
) -> Scorable:
    hooks = importlib.import_module(f"models.{model_name}.train_hooks")
    config = json.loads((adapter / "lora_config.json").read_text())
    model, _ = hooks.setup_training(device, int(config["rank"]), float(config["alpha"]), 0.0)
    from training.checkpoints import load_checkpoint
    from training.master_weights import fp32_forward

    if fp32:
        fp32_forward(model)
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


def stack_states(states: Sequence[object]) -> object:
    stacked = copy.copy(states[0])
    for attr in ("conv_states", "ssm_states"):
        layers = zip(*(getattr(state, attr) for state in states), strict=True)
        setattr(stacked, attr, [torch.cat(tensors) for tensors in layers])
    return stacked


def greedy_replies(model, prompts: Sequence[Sequence[int]], stop_ids: set[int], max_tokens: int):
    """Greedy replies for all prompts in one decode batch: each prompt from empty
    state at batch 1, the states stacked, then one token per step for every row."""
    device = next(model.parameters()).device
    states, last = [], []
    with torch.no_grad():
        for prompt in prompts:
            logits, state = model(torch.tensor([list(prompt)], device=device), state=None)
            states.append(state)
            last.append(logits[:, -1])
        state, logits = stack_states(states), torch.cat(last)
        replies: list[list[int]] = [[] for _ in prompts]
        done = [False] * len(prompts)
        for _ in range(max_tokens):
            tokens = logits.float().argmax(-1)
            for row, token in enumerate(tokens.tolist()):
                if not done[row]:
                    done[row] = token in stop_ids
                    if not done[row]:
                        replies[row].append(token)
            if all(done):
                break
            out, state = model(tokens[:, None], state=state)
            logits = out[:, -1]
    return replies


def greedy_answers(
    model: Scorable, render: Renderer, questions: Sequence[Question], batch_size: int
) -> list[list[int]]:
    """Greedy reply tokens to `[USER] q [ASSISTANT]`, no passage, stopped at
    `<eos>` (or a marker) or `REPLY_CAP` tokens; a reply of `REPLY_CAP` tokens
    did not stop."""
    tokenizer, mod = render.tokenizer, render.mod
    asst = tokenizer.convert_tokens_to_ids(mod.ASST_OPEN)
    stops = {
        tokenizer.eos_token_id,
        asst,
        *tokenizer.convert_tokens_to_ids([mod.USER_OPEN, mod.EOC]),
    }
    prompts = []
    for q in questions:
        ids = render.ids(_qa(q["question"], q["answer"]))
        prompts.append(ids[: ids.index(asst) + 1])
    replies: list[list[int]] = []
    known = 0
    start = time.monotonic()
    model.eval()
    for at in range(0, len(prompts), batch_size):
        batch = greedy_replies(model, prompts[at : at + batch_size], stops, REPLY_CAP)
        known += sum(
            answer_in_reply(q["answer"], tokenizer.decode(r))
            for q, r in zip(questions[at : at + batch_size], batch, strict=True)
        )
        replies += batch
        _progress("generated", len(replies), len(prompts), start, f"known={known}")
    return replies


def mark_generated(
    rows: Sequence[dict[str, object]],
    questions: Sequence[Question],
    replies: Sequence[Sequence[int]],
    render: Renderer,
) -> None:
    """Adds `reply` and `generated_known` to each scored row and logs the
    agreement of the three known rules and decoded samples of each side."""
    for row, q, reply in zip(rows, questions, replies, strict=True):
        row["reply"] = render.tokenizer.decode(list(reply)).strip()
        row["generated_known"] = answer_in_reply(q["answer"], str(row["reply"]))
    table = Counter(
        (
            bool(r["correct_positions"]),
            r["answer_positions"][0] in r["correct_positions"],
            r["generated_known"],
        )
        for r in rows
    )
    print(f"[{ts()}] agreement (teacher-forced any token, first token, generated): count")
    for key, count in sorted(table.items()):
        print(f"    {key}: {count}")
    for side in (True, False):
        picked = [
            (r, q) for r, q in zip(rows, questions, strict=True) if r["generated_known"] == side
        ]
        for r, q in picked[:5]:
            print(
                f"    {'known' if side else 'unknown'} {q['question']!r} -> {q['answer']!r}  "
                f"reply {r['reply']!r}"
            )


def load_passages(
    match: Callable[[str], tuple[str, str] | None], hits: Counter[tuple[str, str]]
) -> tuple[list[Passage], Counter[str]]:
    """SQuAD 2.0 train by passage, LAMA-filtered: a question is dropped when its
    passage, question and answer together carry a pair; the passage is dropped
    when its own text does or its kept questions together do. Each drop's first
    pair is counted in `hits`."""
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

        def qa(q: Question) -> str:
            return f"{q['question']}\n{q.get('answer', '')}"

        def drop(text: str) -> bool:
            pair = match(text)
            if pair:
                hits[pair] += 1
            return pair is not None

        kept = [q for q in p.answerable if not drop(f"{p.text}\n{qa(q)}")]
        unanswerable = [q for q in p.unanswerable if not drop(f"{p.text}\n{qa(q)}")]
        counts["answerable_excluded_by_question"] += len(p.answerable) - len(kept)
        counts["unanswerable_excluded_by_question"] += len(p.unanswerable) - len(unanswerable)
        if drop("\n".join([p.text, *map(qa, kept + unanswerable)])):
            counts["passages_excluded"] += 1
            counts["answerable_excluded_with_passage"] += len(kept)
            counts["unanswerable_excluded_with_passage"] += len(unanswerable)
            continue
        p.answerable, p.unanswerable = kept, unanswerable
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

    lama = {(row["subject"], row["object"]) for row in build_candidates(args.lama_root)}
    match = lama_matcher(lama)
    print(f"[{ts()}] LAMA (subject, object) pairs: {len(lama)} from {args.lama_root}")
    hits: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)

    def lama_hit(pool: str, texts: Iterable[str]) -> bool:
        pair = match("\n".join(texts))
        if pair:
            hits[pool][pair] += 1
        return pair is not None

    def report_hits(pool: str) -> None:
        print(
            f"[{ts()}]   {pool}: {sum(hits[pool].values())} dropped by the pair rule; "
            f"top first matches {hits[pool].most_common(5)}"
        )

    model_mod = importlib.import_module(f"models.{args.model_name}")
    from models.common import build_tokenizer

    render = Renderer(build_tokenizer(model_mod), model_mod)

    print(f"[{ts()}] loading HuggingFaceH4/ultrachat_200k train_sft")
    ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    trained = ultrachat.select(range(WARM_START_ROWS)).to_list()

    passages, squad_counts = load_passages(match, hits["squad"])
    rng.shuffle(passages)
    eligible = [p for p in passages if len(p.answerable) >= 2]
    print(f"[{ts()}] SQuAD: {dict(squad_counts)}")
    print(
        f"[{ts()}]   LAMA-clean passages {len(passages)}/{squad_counts['passages']} "
        f"({len(passages) / squad_counts['passages']:.1%}), with >= 2 answerable {len(eligible)}"
    )
    report_hits("squad")

    device = torch.device("cuda")
    model = _load_model(args.model_name, args.init_adapter, device)
    first = [p.answerable[0] for p in eligible]
    scored = score_questions(model, render, first, args.batch_size, device)
    replies = greedy_answers(model, render, first, args.batch_size)
    mark_generated(scored, first, replies, render)
    scan = dict(zip((p.pid for p in eligible), scored, strict=True))
    known = {pid: bool(r["generated_known"]) for pid, r in scan.items()}
    histogram = Counter(f"{float(r['accuracy']):.1f}" for r in scan.values())
    print(
        f"[{ts()}] v4 first-question scan: {len(scan)} scored, {sum(known.values())} known by "
        f"generate-and-match, accuracy histogram {dict(sorted(histogram.items()))}"
    )

    previous = out / "dreamer_fixed_sets.json"
    old_gate = json.loads(previous.read_text())["gate"] if previous.exists() else []
    old_questions = [{"id": r["question"]["squad_id"], **r["question"]} for r in old_gate]
    first_id = {p.pid: p.answerable[0]["id"] for p in eligible}
    old_replies = greedy_answers(model, render, old_questions, args.batch_size)
    gate_pids = []
    for r, q, reply in zip(old_gate, old_questions, old_replies, strict=True):
        text = render.tokenizer.decode(reply).strip()
        unknown = not answer_in_reply(q["answer"], text)
        same = first_id.get(r["passage_id"]) == q["id"]
        print(
            f"[{ts()}] previous gate row {r['passage_id']} {q['squad_id']}: "
            f"{'unknown' if unknown else 'KNOWN'}, scanned as its passage's first question: "
            f"{same}; {q['question']!r} -> {q['answer']!r} reply {text!r}"
        )
        gate_pids.append(r["passage_id"] if unknown and same else None)
    keep_gate = bool(gate_pids) and None not in gate_pids
    if keep_gate:
        print(f"[{ts()}] previous gate rows kept")
    else:
        print(f"[{ts()}] WARNING: fresh gate draw; gate_dreams.json must be rewritten for it")

    abstention_candidates = list(
        dict.fromkeys(
            r["reply"]
            for r, reply in zip(scored, replies, strict=True)
            if not r["generated_known"]
            and len(reply) < REPLY_CAP
            and 0 < len(str(r["reply"]).split()) <= ABSTENTION_WORDS
        )
    )
    del model
    torch.cuda.empty_cache()
    labels = abstention_labels(abstention_candidates, device)
    by_label: dict[str, list[str]] = defaultdict(list)
    for reply, label in zip(abstention_candidates, labels, strict=True):
        by_label[label].append(reply)
    print(
        f"[{ts()}] abstention candidates: {len(abstention_candidates)} distinct finished replies "
        f"of <= {ABSTENTION_WORDS} words to unknown questions; by label "
        f"{ {k: len(v) for k, v in by_label.items()} }"
    )
    for label, rows in by_label.items():
        for reply in rows[:5]:
            print(f"    {label}: {reply!r}")
    phrases = [p for p in by_label["unknown"] if not lama_hit("abstention_set", [p])]
    print(
        f"[{ts()}] abstention set: {len(phrases)} phrasings"
        f"{'  (under 20: a reading before round 0)' if len(phrases) < 20 else ''}"
    )
    for phrase in phrases:
        print(f"    {phrase!r}")
    if not phrases:
        raise SystemExit("empty abstention set")

    sets = partition_passages(
        passages,
        known,
        prod=PROD,
        gate=GATE,
        held_out=HELD_OUT_FACTS,
        abstention_rows=args.abstention_rows,
        rng=rng,
        gate_pids=gate_pids if keep_gate else (),
    )
    held = [scan[p.pid]["accuracy"] for p in sets["held_out_facts"]]
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
            "v4_accuracy": scan[p.pid]["accuracy"],
            "v4_correct_positions": scan[p.pid]["correct_positions"],
            "v4_reply": scan[p.pid]["reply"],
            "v4_generated_known": scan[p.pid]["generated_known"],
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
            reply = rng.choice([r for r in phrases if not match(f"{q['question']}\n{r}")])
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
        if lama_hit("conversation", (m["content"] for m in row["messages"])):
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
    report_hits("conversation")

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
        if not lama_hit("warm_start", (m["content"] for m in row["messages"]))
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
    report_hits("warm_start")

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
        if not lama_hit("dolly_plain", (m["content"] for m in rec["messages"]))
    ]
    print(
        f"[{ts()}] Dolly plain: {len(dolly)}/{len(plain)} LAMA-clean ({len(dolly) / len(plain):.1%})"
    )
    report_hits("dolly_plain")
    report_hits("abstention_set")
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
    overlaps = sum(match("\n".join(_row_texts(row))) is not None for row in corpus)
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
        json.dumps(
            {
                "classifier": CLASSIFIER,
                "hypotheses": HYPOTHESES,
                "candidates": len(abstention_candidates),
                "by_label": {k: len(v) for k, v in by_label.items()},
                "phrases": phrases,
            },
            indent=1,
        )
    )
    (out / "dreamer_fixed_sets.json").write_text(json.dumps({**meta, **fixed}, indent=1))
    print(
        f"[{ts()}] wrote {len(corpus)} rows: {dict(Counter(str(r['kind']) for r in corpus))} to {out}"
    )


def retained_pool_path(init_adapter: Path, corpus: Path, fp32: bool) -> Path:
    """data/retained_pool-<sha8>.json, -fp32 before the suffix for the fp32 forward."""
    sha = _sha256(init_adapter / "trainable.pt")
    return corpus.parent / f"retained_pool-{sha[:8]}{'-fp32' if fp32 else ''}.json"


def retained_pool(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("the retained-pool pass requires a GPU")
    sha = _sha256(args.init_adapter / "trainable.pt")
    output = args.output or retained_pool_path(args.init_adapter, args.corpus, args.fp32_forward)
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
    model = _load_model(args.model_name, args.init_adapter, device, args.fp32_forward)
    start = time.monotonic()
    scored = score_questions(model, render, drawn, args.batch_size, device)
    mark_generated(scored, drawn, greedy_answers(model, render, drawn, args.batch_size), render)
    elapsed = time.monotonic() - start
    rows = [
        {
            "passage_id": q["passage_id"],
            "squad_id": q["id"],
            "question": q["question"],
            "answer": q["answer"],
            **result,
            "side": "known" if result["generated_known"] else "unknown",
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
            print(
                f"    {side} {r['accuracy']:.2f}  {r['question']!r} -> {r['answer']!r}  {marked}  "
                f"reply {r['reply']!r}"
            )
    Path(output).write_text(
        json.dumps(
            {
                "checkpoint": str(args.init_adapter),
                "trainable_sha256": sha,
                "fp32_forward": args.fp32_forward,
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
    pool.add_argument("--output", type=Path, help="default data/retained_pool-<sha8>[-fp32].json")
    pool.add_argument("--fp32-forward", action="store_true", help="score in the fp32 forward")
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
