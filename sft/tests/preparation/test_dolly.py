import json
import random
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.lama_ckl.protocol import WAKE_FRAME
from preparation.dolly import (
    ACK_FRAME,
    ACK_TEMPLATES,
    acknowledgement_messages,
    dolly_messages,
    partition,
    write_rows,
)


def _rows(n: int) -> list[dict]:
    return [
        {
            "instruction": f"q{i}",
            "context": f"context {i}",
            "response": f"a{i}",
            "category": "closed_qa",
        }
        for i in range(n)
    ]


def _args(tmp_path, max_examples=None, acknowledge=0) -> Namespace:
    return Namespace(
        max_examples=max_examples,
        acknowledge=acknowledge,
        seed=0,
        output=str(tmp_path / "out.jsonl"),
        held_out_output=str(tmp_path / "heldout.json"),
    )


def _written_user_contents(path) -> list[str]:
    return [
        json.loads(line)["messages"][0]["content"] for line in Path(path).read_text().splitlines()
    ]


def test_a_dolly_row_becomes_one_exchange_of_context_then_instruction():
    row = _rows(1)[0]

    assert dolly_messages(row) == {
        "messages": [
            {"role": "user", "content": row["context"] + "\n\n" + row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ]
    }


def test_partition_is_deterministic_disjoint_and_holds_out_50():
    rows = _rows(200)

    probe, pool = partition(rows)

    assert len(probe) == 50
    assert len(pool) == 150
    assert (probe, pool) == partition(rows)
    assert not {r["context"] for r in probe} & {r["context"] for r in pool}


def test_ack_frame_matches_the_wake_prompt():
    assert ACK_FRAME == WAKE_FRAME


def test_an_acknowledgement_row_is_the_framed_context_and_a_template_reply():
    row = _rows(1)[0]

    user, assistant = acknowledgement_messages(row, random.Random(0))["messages"]

    assert user == {"role": "user", "content": ACK_FRAME + row["context"]}
    assert assistant["role"] == "assistant"
    assert assistant["content"] in ACK_TEMPLATES


def test_acknowledgement_contexts_come_from_outside_the_held_out_set(tmp_path):
    args = _args(tmp_path, acknowledge=100)

    write_rows(_rows(200), args)

    held_out = set(json.loads(Path(args.held_out_output).read_text()))
    contexts = [c.removeprefix(ACK_FRAME) for c in _written_user_contents(args.output)]
    assert len(held_out) == 50
    assert len(contexts) == 100
    assert not held_out & set(contexts)


def test_more_acknowledgements_than_pool_rows_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        write_rows(_rows(60), _args(tmp_path, acknowledge=11))


def test_default_max_examples_keeps_every_pool_row(tmp_path):
    args = _args(tmp_path)

    write_rows(_rows(200), args)

    held_out = set(json.loads(Path(args.held_out_output).read_text()))
    written = {c.split("\n\n")[0] for c in _written_user_contents(args.output)}
    assert len(written) == 150
    assert not held_out & written


def test_max_examples_samples_from_the_pool(tmp_path):
    args = _args(tmp_path, max_examples=20)

    write_rows(_rows(200), args)

    held_out = set(json.loads(Path(args.held_out_output).read_text()))
    written = {c.split("\n\n")[0] for c in _written_user_contents(args.output)}
    assert len(written) == 20
    assert not held_out & written
