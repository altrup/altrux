import json
import random

import pytest
import torch

from experiments.lama_ckl.data import build_candidates, select_split
from experiments.lama_ckl.evaluation import (
    encode_metric_batch,
    last_object_token_positions,
    object_token_accuracy,
    score_records,
)
from experiments.lama_ckl.split import freeze_split


def test_build_candidates_uses_longest_masked_sentence_and_not_paper_token_rule(tmp_path):
    (tmp_path / "relations.jsonl").write_text(
        json.dumps(
            {
                "relation": "P530",
                "label": "diplomatic relation",
                "template": "[X] knows [Y] .",
            }
        )
        + "\n"
    )
    trex = tmp_path / "TREx"
    trex.mkdir()
    long = "Ada " + "x" * 205 + " [MASK]"
    (trex / "P530.jsonl").write_text(
        json.dumps(
            {
                "uuid": "u1",
                "sub_label": "Ada",
                "obj_label": "London",
                "evidences": [{"masked_sentence": "Ada [MASK]"}, {"masked_sentence": long}],
            }
        )
        + "\n"
    )

    rows = list(build_candidates(tmp_path))

    assert len(rows) == 1
    assert rows[0]["masked_evidence"] == long
    assert rows[0]["evidence"].endswith(" London")
    assert rows[0]["task_descriptive"] == "Ada knows London ."
    assert rows[0]["invariant"] is False


def test_last_object_token_positions_uses_last_character_occurrence():
    text = "York borders New York ."
    offsets = [(0, 4), (5, 12), (13, 16), (17, 21), (22, 23)]

    assert last_object_token_positions(text, "York", offsets) == [3]


def test_object_token_accuracy_scores_each_aligned_token():
    input_ids = torch.tensor([[9, 3, 4, 5]])
    logits = torch.zeros(1, 4, 10)
    logits[0, 0, 3] = 1
    logits[0, 1, 0] = 1
    logits[0, 2, 5] = 1

    assert object_token_accuracy(logits, input_ids, [[1, 2, 3]]) == pytest.approx([2 / 3])


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, text, **_kwargs):
        words = text.split()
        ids = [1] + [len(word.strip(".")) + 1 for word in words]
        offsets = [(0, 0)]
        cursor = 0
        for word in words:
            start = text.index(word, cursor)
            offsets.append((start, start + len(word)))
            cursor = start + len(word)
        return {"input_ids": ids, "offset_mapping": offsets}


def test_encode_metric_batch_pads_and_aligns_last_object_occurrence():
    rows = [
        {"task_descriptive": "York knows New York .", "object": "York"},
        {"task_descriptive": "Ada knows Rome .", "object": "Rome"},
    ]

    ids, positions = encode_metric_batch(_Tokenizer(), rows, "task_descriptive", 32, "cpu")

    assert ids.shape == (2, 6)
    assert positions == [[4], [3]]
    assert ids[1, -1] == 0


def test_score_records_batches_fresh_state_and_streams_scores():
    class Model:
        def eval(self):
            return self

        def __call__(self, ids, state=None):
            assert state is None
            logits = torch.zeros(*ids.shape, 20)
            for row in range(ids.shape[0]):
                for token in range(ids.shape[1] - 1):
                    logits[row, token, ids[row, token + 1]] = 1
            return logits, object()

    rows = [
        {"task_descriptive": f"Ada knows Rome {index} .", "object": "Rome"} for index in range(3)
    ]

    scores = score_records(Model(), _Tokenizer(), rows, "task_descriptive", 2, 32, "cpu")

    assert scores == [1.0, 1.0, 1.0]


def test_select_split_applies_official_zero_one_rules_and_seeded_sampling():
    rows = []
    for index in range(4):
        rows.append(
            {
                "uuid": f"v{index}",
                "invariant": False,
                "relation_code": "P1",
                "scores": {"descriptive": 0.0, "schematic": 0.0},
            }
        )
        rows.append(
            {
                "uuid": f"i{index}",
                "invariant": True,
                "relation_code": "P2",
                "scores": {"descriptive": 1.0, "schematic": 0.5},
            }
        )
    rows.append(
        {
            "uuid": "partial",
            "invariant": False,
            "relation_code": "P1",
            "scores": {"descriptive": 0.0, "schematic": 0.5},
        }
    )

    learned, retained = select_split(rows, size=2, rng=random.Random(42))

    assert [row["uuid"] for row in learned] == ["v0", "v3"]
    assert [row["uuid"] for row in retained] == ["i2", "i0"]


def test_freeze_split_writes_hashed_artifacts_once(tmp_path):
    learned = [{"uuid": "v1", "evidence": "Ada in Rome", "subject": "Ada", "object": "Rome"}]
    retained = [{"uuid": "i1", "evidence": "Bob in Oslo", "subject": "Bob", "object": "Oslo"}]

    manifest = freeze_split(tmp_path, learned, retained, {"source_sha256": "abc"})

    assert manifest["artifacts"]["variant.jsonl"]["rows"] == 1
    assert len(manifest["artifacts"]["variant.jsonl"]["sha256"]) == 64
    assert json.loads((tmp_path / "manifest.json").read_text())["source_sha256"] == "abc"
    with pytest.raises(RuntimeError, match="different content"):
        freeze_split(tmp_path, learned + learned, retained, {"source_sha256": "abc"})
