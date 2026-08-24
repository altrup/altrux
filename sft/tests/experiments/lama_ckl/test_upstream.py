import json

import pytest

from experiments.lama_ckl.upstream import (
    RELEASE_FILES,
    reproduction_summary,
    verify_release,
)


def test_verify_release_checks_count_schema_and_hash(tmp_path):
    path = tmp_path / "variant.jsonl"
    row = {
        "uuid": "id-1",
        "relation_code": "P1",
        "subject": "Ada",
        "object": "London",
        "evidence": "Ada lived in London.",
        "task_descriptive": "Ada lives in London.",
        "invariant": False,
    }
    path.write_text(json.dumps(row) + "\n")
    files = {"variant.jsonl": {"rows": 1, "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}}

    report = verify_release(tmp_path, files)

    assert report["variant.jsonl"]["rows"] == 1
    assert report["variant.jsonl"]["invariants"]["malformed"] == 0


def test_verify_release_rejects_changed_artifact(tmp_path):
    path = tmp_path / "variant.jsonl"
    path.write_text("{}\n")

    with pytest.raises(ValueError, match="sha256"):
        verify_release(tmp_path, {"variant.jsonl": {"rows": 1, "sha256": "0" * 64}})


def test_release_manifest_pins_the_four_official_files():
    assert RELEASE_FILES["variant.jsonl"]["rows"] == 500
    assert RELEASE_FILES["invariant_descriptive.jsonl"]["rows"] == 500
    assert RELEASE_FILES["invariant_schematic.jsonl"]["rows"] == 500
    assert RELEASE_FILES["train_attention_traindata.jsonl"]["rows"] == 4166


def test_reproduction_summary_uses_first_peak_and_checks_frozen_tolerance():
    curve = [
        {"epoch": 15, "to_learn_accuracy": 0.10, "not_to_forget_accuracy": 0.83},
        {"epoch": 16, "to_learn_accuracy": 0.115, "not_to_forget_accuracy": 0.8174},
        {"epoch": 17, "to_learn_accuracy": 0.115, "not_to_forget_accuracy": 0.80},
    ]

    result = reproduction_summary(curve)

    assert result == {
        "top_accuracy": 0.115,
        "epoch": 16,
        "not_to_forget_accuracy": 0.8174,
        "total_knowledge": 0.9324,
        "passes_gate": True,
    }
