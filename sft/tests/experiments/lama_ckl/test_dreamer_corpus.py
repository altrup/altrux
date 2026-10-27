import random

import pytest
import torch

from experiments.lama_ckl.dreamer_corpus import (
    Passage,
    is_acknowledgement,
    lama_matcher,
    mine_abstentions,
    pair_negatives,
    partition_passages,
    token_correctness,
    valid_boundaries,
)


def test_matcher_is_word_bounded_and_case_insensitive():
    match = lama_matcher(["Spain", "New York", "C++"])
    assert match("He lived in spain.") == "spain"
    assert match("A Spaniard sailed.") is None
    assert match("Moved to NEW YORK, then back.") == "new york"
    assert match("New Yorker magazine") is None
    assert match("written in C++ mostly") == "c++"
    assert match("nothing here") is None


def test_acknowledgement_reads_the_first_line_only():
    assert is_acknowledgement("Got it.")
    assert is_acknowledgement("Sure, I can help with that.\n1. First step ...")
    assert not is_acknowledgement("Surely this is a longer answer about things.")
    assert not is_acknowledgement("Okay so here is a very long first line of the reply text")
    assert not is_acknowledgement("The capital is Paris.")


def _turns(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in pairs]


def test_boundaries_need_an_assistant_turn_each_side_and_respect_the_cap():
    messages = _turns(
        ("user", "a b"),
        ("assistant", "c d e"),
        ("user", "f"),
        ("assistant", "g h i j"),
        ("user", "k"),
        ("assistant", "l m n"),
    )

    def words(prefix: list[dict[str, str]]) -> int:
        return sum(len(m["content"].split()) for m in prefix)

    valid, capped, acks = valid_boundaries(messages, words, cap=100)
    assert [b["at"] for b in valid] == [2, 4]
    assert [b["prefix_tokens"] for b in valid] == [5, 10]
    assert (capped, acks) == (0, 0)

    valid, capped, acks = valid_boundaries(messages, words, cap=9)
    assert [b["at"] for b in valid] == [2]
    assert (capped, acks) == (1, 0)


def test_acknowledgement_target_is_redrawn_else_the_row_is_dropped():
    messages = _turns(
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
        ("assistant", "Thanks!"),
        ("user", "q3"),
        ("assistant", "A real reply with content."),
    )
    valid, _, acks = valid_boundaries(messages, lambda p: len(p), cap=100)
    assert [b["at"] for b in valid] == [4]
    assert acks == 1

    only_acks = _turns(("user", "q"), ("assistant", "a"), ("user", "ok"), ("assistant", "Noted."))
    valid, _, acks = valid_boundaries(only_acks, lambda p: len(p), cap=100)
    assert valid == [] and acks == 1


def test_negative_pairing_never_self_or_identical():
    targets = ["Paris", "Paris", "Rome", "Oslo", "Rome"]
    for seed in range(50):
        pairs = pair_negatives(targets, random.Random(seed))
        for i, j in enumerate(pairs):
            assert i != j
            assert targets[i] != targets[j]


def test_negative_pairing_raises_when_no_distinct_target_exists():
    with pytest.raises(ValueError):
        pair_negatives(["same", "same"], random.Random(0))


def test_abstention_mining_stems_word_cap_and_dedup():
    convs = [
        _turns(("user", "x"), ("assistant", "I don't know the answer to that.")),
        _turns(("user", "y"), ("assistant", "I don't know the answer to that.")),
        _turns(("user", "z"), ("assistant", "I'm sorry, but I cannot browse the web.")),
        _turns(("user", "I don't know"), ("assistant", "Here is a long answer.")),
        _turns(("user", "w"), ("assistant", "I cannot " + "word " * 40)),
        _turns(("user", "v"), ("assistant", "Well, I'm not sure.")),
    ]
    assert mine_abstentions(convs) == [
        "I don't know the answer to that.",
        "I'm sorry, but I cannot browse the web.",
    ]


def _passage(pid: str, answerable: int, unanswerable: int) -> Passage:
    return Passage(
        pid=pid,
        title="t",
        text=f"passage {pid}",
        answerable=[
            {"id": f"{pid}a{i}", "question": "q", "answer": "a"} for i in range(answerable)
        ],
        unanswerable=[{"id": f"{pid}u{i}", "question": "q"} for i in range(unanswerable)],
    )


def test_partition_is_disjoint_at_the_passage_level():
    passages = [_passage(str(i), answerable=i % 4, unanswerable=i % 3) for i in range(60)]
    eligible = [p for p in passages if len(p.answerable) >= 2]
    scores = {p.pid: (k % 4) / 4 for k, p in enumerate(eligible)}
    sets = partition_passages(passages, scores, prod=4, gate=2, held_out=3, abstention_rows=5)

    assert [len(sets[k]) for k in ("prod", "gate", "held_out_facts")] == [4, 2, 3]
    assert all(scores[p.pid] == 0.0 for p in sets["prod"] + sets["gate"])
    best = [p.pid for p in eligible if scores[p.pid] == 0.75][:3]
    assert [p.pid for p in sets["held_out_facts"]] == best
    assert sum(len(p.unanswerable) for p in sets["abstention"]) >= 5
    assert all(len(p.answerable) >= 2 for p in sets["facts"])
    ids = [p.pid for rows in sets.values() for p in rows]
    assert len(ids) == len(set(ids))


def test_token_correctness_lists_the_correctly_predicted_answer_positions():
    ids = torch.tensor([[5, 6, 7, 8], [5, 6, 7, 8]])
    logits = torch.zeros(2, 4, 10)
    for row, predicted in enumerate(([6, 7, 1, 0], [6, 2, 8, 0])):
        for position, token in enumerate(predicted):
            logits[row, position, token] = 1.0
    assert token_correctness(logits, ids, [[2, 3], [1, 2, 3]]) == [[2], [1, 3]]


def test_partition_raises_when_fixed_supply_is_short():
    passages = [_passage(str(i), answerable=2, unanswerable=1) for i in range(5)]
    with pytest.raises(ValueError):
        partition_passages(
            passages, {p.pid: 0.5 for p in passages}, prod=4, gate=2, held_out=1, abstention_rows=1
        )
