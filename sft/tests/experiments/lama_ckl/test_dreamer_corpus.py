import random

import pytest
import torch

from experiments.lama_ckl.dreamer_corpus import (
    Passage,
    abstention_labels,
    answer_in_reply,
    is_acknowledgement,
    lama_matcher,
    normalize_answer,
    pair_negatives,
    partition_passages,
    token_correctness,
    valid_boundaries,
)


def test_matcher_is_word_bounded_and_case_insensitive():
    match = lama_matcher([("Spain", "Madrid"), ("New York", "C++")])
    assert match("He lived in spain, near MADRID.") == ("spain", "madrid")
    assert match("A Spaniard sailed to Madrid.") is None
    assert match("Moved to NEW YORK and wrote C++ mostly") == ("new york", "c++")
    assert match("New Yorker magazine, C++") is None
    assert match("nothing here") is None


def test_matcher_needs_both_halves_of_one_pair():
    match = lama_matcher([("Spain", "Madrid"), ("France", "Paris")])
    assert match("Spain alone") is None
    assert match("Spain and Paris, not one pair") is None
    assert match("Paris is the capital of France") == ("france", "paris")


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


def test_squad_normalisation_and_word_bounded_match():
    assert normalize_answer("The  Eiffel-Tower!") == "eiffeltower"
    assert normalize_answer("an Apple a day") == "apple day"
    assert answer_in_reply("Gustave Eiffel", "It was built by gustave eiffel.")
    assert answer_in_reply("the 1990s", "In 1990s, yes")
    assert not answer_in_reply("art", "It was a party.")
    assert not answer_in_reply("The", "The answer.")


def test_abstention_labels_map_the_top_hypothesis(monkeypatch):
    from experiments.lama_ckl import dreamer_corpus

    calls = []

    def classifier(replies, **kwargs):
        calls.append((list(replies), kwargs))
        top = {
            "I don't know.": "The speaker does not know the answer",
            "I can't do that.": "The speaker cannot perform the action",
        }
        return [
            {"labels": [top.get(r, "The speaker answers the question"), "x"], "scores": [0.6, 0.4]}
            for r in replies
        ]

    monkeypatch.setattr(dreamer_corpus, "_classifier", lambda device: classifier)
    labels = abstention_labels(["I don't know.", "Paris.", "I can't do that."], "cpu")
    assert labels == ["unknown", "answers", "cannot"]
    assert calls[0][1]["hypothesis_template"] == "{}" and not calls[0][1]["multi_label"]


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
    known = {p.pid: k % 3 == 0 for k, p in enumerate(eligible)}
    sets = partition_passages(
        passages, known, prod=4, gate=2, held_out=3, abstention_rows=5, rng=random.Random(0)
    )

    assert [len(sets[k]) for k in ("prod", "gate", "held_out_facts")] == [4, 2, 3]
    assert not any(known[p.pid] for p in sets["prod"] + sets["gate"])
    assert all(known[p.pid] for p in sets["held_out_facts"])
    assert sum(len(p.unanswerable) for p in sets["abstention"]) >= 5
    assert all(len(p.answerable) >= 2 for p in sets["facts"])
    ids = [p.pid for rows in sets.values() for p in rows]
    assert len(ids) == len(set(ids))


def test_partition_keeps_a_given_gate_set():
    passages = [_passage(str(i), answerable=2, unanswerable=1) for i in range(40)]
    known = {p.pid: False for p in passages} | {"0": True, "1": True, "2": True}
    sets = partition_passages(
        passages,
        known,
        prod=4,
        gate=2,
        held_out=2,
        abstention_rows=1,
        rng=random.Random(0),
        gate_pids=["30", "31"],
    )
    assert [p.pid for p in sets["gate"]] == ["30", "31"]
    assert "30" not in {p.pid for p in sets["prod"]}


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
            passages,
            {p.pid: False for p in passages},
            prod=4,
            gate=2,
            held_out=1,
            abstention_rows=1,
            rng=random.Random(0),
        )
