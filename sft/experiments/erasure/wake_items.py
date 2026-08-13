"""Wake-item and bystander construction for erasure experiments."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

from experiments.facts import FILLER_SENTENCES, Fact
from progress import ts


# Off-format bystanders: same transcript, different relation templates, no
# digits (the transcript's digit-free-filler invariant is what makes the code
# probes meaningful). Disjoint from consolidation_null's ENTITY_POOL.
PEOPLE = ["Alice", "Bertram", "Clara", "Dmitri", "Elena", "Farid", "Greta", "Hugo"]
CITIES = ["Paris", "Lisbon", "Oslo", "Nagoya", "Cairo", "Toronto", "Perth", "Dublin"]
EVENTS = ["meeting", "rehearsal", "inspection", "briefing", "handover", "audit"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
OBJECTS = ["parcel", "crate", "canister", "satchel", "toolbox", "kettle"]
WEIGHTS = ["nine kilograms", "four kilograms", "seven kilograms", "twelve kilograms", "three kilograms", "six kilograms"]

# Near-cone bystanders: numeric answers in the same spaced-digit shape as a
# code, but a different relation -- the hard case for a rank-1 erase aimed at
# "the code for X is". Pools disjoint from PEOPLE above.
SENDERS = ["Ingrid", "Joaquin", "Kenji", "Lucia", "Marek", "Noor", "Ottoline", "Priya"]
FIRMS = ["Halverson", "Redmond", "Castellane", "Ophir", "Vantage", "Kirkwall", "Solvay", "Norbury"]


class Bystander:
    def __init__(self, label: str, question: str, statement: str, prompt: str, answer: str,
                 cls: str = "offformat") -> None:
        self.label, self.question, self.statement = label, question, statement
        self.prompt, self.answer, self.cls = prompt, answer, cls


def collisions(text: str, facts: Sequence[Fact], battery_answers: Sequence[str]) -> list[str]:
    """Every fact name, fact code or battery answer `text` mentions, matched on
    word boundaries. Non-empty means this content cannot sit in a wake
    transcript: probing it would be probing something else (the parcel ->
    shipment precedent)."""
    low = text.lower()
    needles = [f.entity for f in facts] + [f.code for f in facts] + list(battery_answers)
    return sorted({n for n in needles if re.search(rf"\b{re.escape(n.lower())}\b", low)})


def assert_no_collisions(items: Sequence[Bystander], facts: Sequence[Fact],
                         battery_answers: Sequence[str]) -> None:
    """Refuse to build a wake transcript whose distractor content overlaps what
    the run measures. The pools are filtered before sampling, so this firing
    means a filter was bypassed, not that a draw was unlucky."""
    bad = {item.label: hit for item in items
           if (hit := collisions(f"{item.question} {item.statement}", facts, battery_answers))}
    if bad:
        raise SystemExit(
            "wake transcript refused: distractor content collides with what this run measures -- "
            + "; ".join(f"{label} -> {', '.join(hits)}" for label, hits in sorted(bad.items()))
        )


def build_dialogue(records: Sequence[dict], n: int, rng: random.Random, user_open: str,
                   asst_open: str, facts: Sequence[Fact],
                   battery_answers: Sequence[str]) -> list[Bystander]:
    """n ordinary chat exchanges drawn from `records` ({"messages": [...]}),
    shaped as probe items so the same fresh-state QA that scores a bystander
    scores them. Conversations mentioning a fact or a battery answer are
    skipped -- real prose is filtered, not repaired."""
    out: list[Bystander] = []
    skipped = 0
    for i in rng.sample(range(len(records)), len(records)):
        if len(out) == n:
            break
        messages = records[i].get("messages", [])
        pair = next(((messages[j]["content"], messages[j + 1]["content"])
                     for j in range(len(messages) - 1)
                     if messages[j]["role"] == "user" and messages[j + 1]["role"] == "assistant"), None)
        if pair is None:
            continue
        question, answer = (" ".join(p.split())[:200] for p in pair)
        if collisions(f"{question} {answer}", facts, battery_answers):
            skipped += 1
            continue
        out.append(Bystander(f"dialogue{len(out)}", question, answer,
                             f"{user_open} {question}{asst_open}", f" {answer}", "dialogue"))
    if len(out) < n:
        raise SystemExit(
            f"wake transcript refused: only {len(out)} of {n} dialogue exchanges are free of the "
            f"facts and the battery answers ({skipped} skipped over {len(records)} candidates). "
            f"Widen the candidate pool or lower the dialogue count."
        )
    print(f"[{ts()}] wake dialogue: kept {len(out)} of {len(records)} candidates ({skipped} skipped on collision)")
    return out


def build_wake_items(facts: Sequence[Fact], bystanders: int, nearcone: int, dialogue: int,
                     dialogue_records: Sequence[dict], rng: random.Random, user_open: str,
                     asst_open: str, battery_answers: Sequence[str]):
    """The rich wake transcript's items (sec 2.10.11): the facts plus
    heterogeneous distractor content -- off-format bystanders, near-cone
    numeric bystanders, and a slice of ordinary dialogue -- shuffled together.

    The A-vs-B4 contrast is entirely about non-fact state content, which 40
    tokens of filler starves. Returns (items, distractors); `distractors` is
    what the context-leakage probe scores.
    """
    avoid = [f.entity for f in facts] + list(battery_answers)
    distractors = build_bystanders(bystanders, rng, user_open, asst_open, avoid=avoid)
    distractors += build_nearcone(nearcone, rng, user_open, asst_open, avoid=avoid)
    distractors += build_dialogue(dialogue_records, dialogue, rng, user_open, asst_open,
                                  facts, battery_answers)
    assert_no_collisions(distractors, facts, battery_answers)
    items = [*facts, *distractors]
    rng.shuffle(items)
    return items, distractors


def report_distractors(decoded: str, distractors: Sequence[Bystander]) -> None:
    """Structural invariants for the distractor half of a rich wake transcript,
    plus decoded text around the first item of each class."""
    if not distractors:
        return
    misstated = sum(1 for d in distractors if decoded.count(d.statement) != 1)
    duplicates = len(distractors) - len({d.label for d in distractors})
    classes = {d.cls for d in distractors}
    print(f"[{ts()}]   distractors: {len(distractors)} -- "
          + ", ".join(f"{sum(d.cls == c for d in distractors)} {c}" for c in sorted(classes)))
    print(f"[{ts()}]   distractors not stated exactly once: {misstated}  (must be 0)")
    print(f"[{ts()}]   duplicate distractor labels        : {duplicates}  (must be 0)")
    for cls in sorted(classes):
        first = next(d for d in distractors if d.cls == cls)
        at = decoded.find(first.statement)
        print(f"[{ts()}]   sample around the first {cls} item ({first.label}):\n"
              f"    ...{decoded[max(0, at - 150) : at + 200]!r}...")


def _pool(values: Sequence[str], rng: random.Random, avoid: Sequence[str]) -> list[str]:
    """A pool drawn without replacement, minus anything the run measures.
    Identical rng consumption to a plain sample when nothing is avoided."""
    kept = [v for v in values if v.lower() not in {a.lower() for a in avoid}]
    return rng.sample(kept, len(kept))


def build_nearcone(n: int, rng: random.Random, user_open: str, asst_open: str,
                   avoid: Sequence[str] = ()) -> list[Bystander]:
    """n numeric-but-off-relation facts: spaced digits like a code, different
    relation. Digit counts (3, 4) differ from CODE_DIGITS so no answer can
    collide with a code as a full string."""
    senders, firms = _pool(SENDERS, rng, avoid), _pool(FIRMS, rng, avoid)
    out = []
    for i in range(n):
        if i % 2 == 0:
            who, val = senders.pop(), " ".join(str(rng.randrange(10)) for _ in range(3))
            label, q = who, f"How much does the shipment from {who} weigh?"
            stmt, stem, ans = f"The shipment from {who} weighs {val} kilograms.", f"The shipment from {who} weighs", f" {val}"
        else:
            org, val = firms.pop(), " ".join(str(rng.randrange(10)) for _ in range(4))
            label, q = org, f"How much does the invoice from {org} total?"
            stmt, stem, ans = f"The invoice from {org} totals {val} dollars.", f"The invoice from {org} totals", f" {val}"
        out.append(Bystander(label, q, stmt, f"{user_open} {q}{asst_open} {stem}", ans, "nearcone"))
    return out


def build_bystanders(n: int, rng: random.Random, user_open: str, asst_open: str,
                     avoid: Sequence[str] = ()) -> list[Bystander]:
    """n off-format facts, cycling three relation templates with fillers drawn
    without replacement. `stem` is the assistant prefix the probe leaks, so the
    answer log-prob is scored on the value alone. `avoid` drops pool entries
    that would collide with what the run measures."""
    pools = {k: _pool(v, rng, avoid) for k, v in
             (("people", PEOPLE), ("cities", CITIES), ("events", EVENTS),
              ("weekdays", WEEKDAYS), ("objects", OBJECTS), ("weights", WEIGHTS))}
    out = []
    for i in range(n):
        if i % 3 == 0:
            who, where = pools["people"].pop(), pools["cities"].pop()
            label, q, stmt, stem, ans = who, f"Where does {who} live?", f"{who} lives in {where}.", f"{who} lives in", f" {where}"
        elif i % 3 == 1:
            ev, day = pools["events"].pop(), pools["weekdays"].pop()
            label, q, stmt, stem, ans = ev, f"When is the {ev}?", f"The {ev} is on {day}.", f"The {ev} is on", f" {day}"
        else:
            obj, w = pools["objects"].pop(), pools["weights"].pop()
            label, q, stmt, stem, ans = obj, f"How much does the {obj} weigh?", f"The {obj} weighs {w}.", f"The {obj} weighs", f" {w}"
        out.append(Bystander(label, q, stmt, f"{user_open} {q}{asst_open} {stem}", ans))
    return out


def build_mixed_turns(items, filler_tokens, token_len, rng):
    """build_turns, but the item sequence may interleave off-format bystanders
    with the code facts. Identical rng consumption when there are none."""
    turns: list[tuple[str, str]] = []
    for i, item in enumerate(items):
        if i:
            used = 0
            while used < filler_tokens:
                for role in ("user", "assistant"):
                    sentence = rng.choice(FILLER_SENTENCES)
                    turns.append((role, sentence))
                    used += token_len(sentence)
        if isinstance(item, Fact):
            turns.extend([("user", f"What is the code for the {item.entity}?"),
                          ("assistant", f"The code for the {item.entity} is {item.code}.")])
        else:
            turns.extend([("user", item.question), ("assistant", item.statement)])
    return turns
