import json
import random
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.lama_ckl.protocol import WAKE_FRAME
from preparation.dolly import (
    OPENERS,
    PREFACES,
    RECALL_GIST_REQUESTS,
    RECALL_REQUESTS,
    acknowledgement,
    partition,
    render_plain,
    write_rows,
)


def _rows(n: int, category: str = "closed_qa") -> list[dict]:
    return [
        {
            "instruction": f"q{i}",
            "context": f"context {i}",
            "response": f"a{i}",
            "category": category,
        }
        for i in range(n)
    ]


def _args(tmp_path, max_examples=None) -> Namespace:
    return Namespace(
        max_examples=max_examples,
        seed=0,
        output=str(tmp_path / "out.jsonl"),
        held_out_output=str(tmp_path / "heldout.json"),
    )


def _records(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _written_contexts(path) -> list[str]:
    return [row["context"] for rec in _records(path) for row in rec["dolly"]["rows"]]


def test_partition_is_deterministic_disjoint_and_holds_out_50():
    rows = _rows(200)

    probe, pool = partition(rows)

    assert len(probe) == 50
    assert len(pool) == 150
    assert (probe, pool) == partition(rows)
    assert not {r["context"] for r in probe} & {r["context"] for r in pool}


def test_partition_is_pinned():
    probe, _ = partition(_rows(200))

    assert [r["context"] for r in probe[:5]] == [
        "context 131",
        "context 181",
        "context 22",
        "context 172",
        "context 144",
    ]


@pytest.mark.parametrize("phrases", [OPENERS, PREFACES, RECALL_REQUESTS, RECALL_GIST_REQUESTS])
def test_phrase_sets_are_large_distinct_and_frame_free(phrases):
    frame = WAKE_FRAME.strip().lower()

    assert len(phrases) >= 40
    assert len(set(phrases)) == len(phrases)
    assert not [p for p in phrases if frame in p.lower()]


def test_held_out_contexts_never_reach_the_output(tmp_path):
    args = _args(tmp_path)

    write_rows(_rows(200), args)

    held_out = set(json.loads(Path(args.held_out_output).read_text()))
    assert len(held_out) == 50
    assert not held_out & set(_written_contexts(args.output))


def test_default_keeps_every_pool_row_exactly_once(tmp_path):
    args = _args(tmp_path)

    write_rows(_rows(200), args)

    contexts = _written_contexts(args.output)
    assert len(contexts) == 150
    assert len(set(contexts)) == 150
    assert len(_records(args.output)) < 150


def test_max_examples_samples_from_the_pool(tmp_path):
    args = _args(tmp_path, max_examples=20)

    write_rows(_rows(200), args)

    held_out = set(json.loads(Path(args.held_out_output).read_text()))
    contexts = _written_contexts(args.output)
    assert len(set(contexts)) == 20
    assert not held_out & set(contexts)


def test_every_record_carries_its_rows_with_the_four_fields(tmp_path):
    args = _args(tmp_path)

    write_rows(_rows(100) + _rows(100, "summarization"), args)

    for rec in _records(args.output):
        assert rec["dolly"]["shape"] in {"plain", "document", "multi"}
        for row in rec["dolly"]["rows"]:
            assert set(row) == {"context", "instruction", "response", "summary"}


def test_summary_field_is_set_only_for_summarization_rows(tmp_path):
    args = _args(tmp_path)
    rows = _rows(100) + [
        {**r, "context": "s" + r["context"], "category": "summarization"} for r in _rows(100)
    ]

    write_rows(rows, args)

    for rec in _records(args.output):
        for row in rec["dolly"]["rows"]:
            expected = row["response"] if row["context"].startswith("s") else None
            assert row["summary"] == expected


def test_shape_shares_are_about_40_40_20(tmp_path):
    args = _args(tmp_path)

    write_rows(_rows(2050), args)

    counts = {"plain": 0, "document": 0, "multi": 0}
    for rec in _records(args.output):
        counts[rec["dolly"]["shape"]] += len(rec["dolly"]["rows"])
    total = sum(counts.values())
    assert counts["plain"] / total == pytest.approx(0.4, abs=0.05)
    assert counts["document"] / total == pytest.approx(0.4, abs=0.05)
    assert counts["multi"] / total == pytest.approx(0.2, abs=0.05)


def test_plain_puts_the_instruction_first_about_half_the_time():
    rng = random.Random(0)
    rows = _rows(1000)

    first = sum(render_plain(r, rng)[0]["content"].startswith(r["instruction"]) for r in rows)

    assert 400 < first < 600


def test_multi_records_hold_two_or_three_rows_in_alternating_turns(tmp_path):
    args = _args(tmp_path)

    write_rows(_rows(500), args)

    multi = [rec for rec in _records(args.output) if rec["dolly"]["shape"] == "multi"]
    assert multi
    for rec in multi:
        n = len(rec["dolly"]["rows"])
        assert n in (2, 3)
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["user", "assistant"] * n
        for row, user in zip(rec["dolly"]["rows"], rec["messages"][::2]):
            assert user["content"].endswith(row["context"])


def test_summarization_acknowledgement_is_the_summary_with_or_without_an_opener():
    rng = random.Random(0)
    row = {**_rows(1, "summarization")[0], "summary": "the human summary"}

    acks = [acknowledgement(row, rng) for _ in range(200)]

    assert all(a.endswith("the human summary") for a in acks)
    bare = sum(a == "the human summary" for a in acks)
    assert 60 < bare < 140
    assert all(
        a == "the human summary" or a.removesuffix(" the human summary") in OPENERS for a in acks
    )


def test_other_acknowledgements_are_an_opener_or_a_one_sentence_response():
    rng = random.Random(0)
    short = {**_rows(1)[0], "response": "Paris is the capital.", "summary": None}
    long = {**short, "response": "First sentence. Second sentence."}

    short_acks = {acknowledgement(short, rng) for _ in range(200)}
    long_acks = {acknowledgement(long, rng) for _ in range(200)}

    assert "Paris is the capital." in short_acks
    assert short_acks - {"Paris is the capital."} <= set(OPENERS)
    assert long_acks <= set(OPENERS)
