"""Experiment: shared

Synthetic fact, transcript, cue, and answer primitives."""

import itertools
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar


CODE_DIGITS = 5
DISTRACTOR_SALT = 0x5EED

ENTITY_POOL: dict[str, list[str]] = {
    "bird": ["heron", "magpie", "falcon", "sparrow", "kestrel", "pelican", "curlew", "osprey", "plover", "warbler"],
    "tree": ["maple", "cedar", "birch", "willow", "aspen", "juniper", "hemlock", "poplar", "alder", "hazel"],
    "mineral": ["quartz", "basalt", "gypsum", "feldspar", "calcite", "olivine", "pyrite", "garnet", "topaz", "jasper"],
    "instrument": ["cello", "oboe", "banjo", "marimba", "clarinet", "bassoon", "trombone", "zither", "dulcimer", "viola"],
    "spice": ["cumin", "saffron", "cardamom", "paprika", "turmeric", "coriander", "fennel", "nutmeg", "anise", "clove"],
    "vessel": ["schooner", "frigate", "trawler", "galleon", "sloop", "ketch", "barque", "corvette", "dinghy", "cutter"],
}

FILLER_SENTENCES = [
    "The weather in the valley stayed mild for most of the season.",
    "A good soup starts with onions cooked slowly until they turn golden.",
    "The train from the coast arrives twice a day, once at dawn and once at dusk.",
    "Most of the library's east wing is dedicated to maritime history.",
    "She repainted the fence a pale shade of green last spring.",
    "Migrating flocks tend to follow the river south this time of year.",
    "The old mill has been converted into a small museum of local crafts.",
    "He prefers cycling to work when the mornings are dry.",
    "Fresh basil loses its aroma quickly once the leaves are bruised.",
    "The lighthouse keeper kept meticulous logs of every passing storm.",
    "Their garden produces more zucchini than the whole street can eat.",
    "A thin layer of fog settled over the harbor before sunrise.",
    "The concert hall's acoustics favor the string section.",
    "The bakery on the corner sells out of rye bread before noon.",
]

T = TypeVar("T")
Turn = tuple[str, str]


@dataclass(frozen=True)
class Fact:
    entity: str
    category: str
    code: str


def build_facts(n: int, rng: random.Random) -> list[Fact]:
    """Build facts on distinct entities with fresh random digit codes."""
    pool = [(name, category) for category, names in ENTITY_POOL.items() for name in names]
    if n > len(pool):
        raise ValueError(f"only {len(pool)} entities available, asked for {n}")
    return [
        Fact(name, category, " ".join(str(rng.randrange(10)) for _ in range(CODE_DIGITS)))
        for name, category in rng.sample(pool, n)
    ]


def build_distractors(
    facts: Sequence[Fact], seed: int, taken: Sequence[str] = ()
) -> dict[str, str]:
    rng = random.Random(seed ^ DISTRACTOR_SALT)
    used = {fact.code for fact in facts} | set(taken)
    distractors: dict[str, str] = {}
    for fact in facts:
        code = " ".join(str(rng.randrange(10)) for _ in range(CODE_DIGITS))
        while code in used:
            code = " ".join(str(rng.randrange(10)) for _ in range(CODE_DIGITS))
        used.add(code)
        distractors[fact.entity] = code
    return distractors


def fact_turns(fact: Fact) -> list[Turn]:
    """Build the two turns that state one fact."""
    return [
        ("user", f"What is the code for the {fact.entity}?"),
        ("assistant", f"The code for the {fact.entity} is {fact.code}."),
    ]


def build_turns(
    facts: Sequence[Fact],
    filler_tokens: int,
    token_len: Callable[[str], int],
    rng: random.Random,
) -> list[Turn]:
    """Build a wake transcript with digit-free filler between facts."""
    turns: list[Turn] = []
    for index, fact in enumerate(facts):
        if index:
            used = 0
            while used < filler_tokens:
                for role in ("user", "assistant"):
                    sentence = rng.choice(FILLER_SENTENCES)
                    turns.append((role, sentence))
                    used += token_len(sentence)
        turns.extend(fact_turns(fact))
    return turns


def render_turns(turns: Iterable[Turn], user_open: str, asst_open: str) -> str:
    """Render turns in the training chat format."""
    return "".join(f"{user_open if role == 'user' else asst_open} {text}" for role, text in turns)


def role_adjacency_violations(turns: Sequence[Turn]) -> int:
    return sum(1 for first, second in itertools.pairwise(turns) if first[0] == second[0])


def normalize(text: str) -> str:
    return " ".join(text.split()).strip(" .,;:!?\"'")


def digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def extract_answer(text: str, stops: Sequence[str] = (".", "\n")) -> str:
    cut = len(text)
    for stop in stops:
        index = text.find(stop)
        if index != -1:
            cut = min(cut, index)
    return normalize(text[:cut])


def exact_match(text: str, code: str, stops: Sequence[str] = (".", "\n")) -> bool:
    answer = extract_answer(text, stops)
    return bool(answer) and digits(answer) == digits(code)


def contains_code(text: str, code: str) -> bool:
    return digits(code) in digits(text)


def pass_at_k(samples: Sequence[str], code: str) -> float:
    if not samples:
        return 0.0
    return sum(contains_code(sample, code) for sample in samples) / len(samples)


def first_success(rungs: Sequence[T], probe: Callable[[T], bool]) -> int:
    """Return the one-based first successful rung, or zero."""
    for index, rung in enumerate(rungs, 1):
        if probe(rung):
            return index
    return 0


def cue_rungs(fact: Fact, user_open: str, asst_open: str) -> list[tuple[str, str]]:
    """Build free-recall, category-hint, and first-digit cue rungs."""
    stem = f"{asst_open} The code for the {fact.entity} is"
    free = f"{user_open} What is the code for the {fact.entity}?{stem}"
    hinted = f"{user_open} What is the code for the {fact.entity}, the {fact.category}?{stem}"
    first = fact.code.split()[0]
    return [(free, ""), (hinted, ""), (f"{free} {first}", f" {first}")]
