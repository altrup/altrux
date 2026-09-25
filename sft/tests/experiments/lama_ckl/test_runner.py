from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.lama_ckl.runner import _write_cycle_result, compact_dream_payload, curve_summary


def test_compact_dream_payload_keeps_reconstruction_data_but_not_logits():
    dreams = [
        SimpleNamespace(
            dream_ids=[3, 4, 9],
            token_texts=["[U]", "[A]", "x"],
            prefix_len=2,
            stop_reason="eoc",
            dream_sha="abc",
            teacher_logits=torch.zeros(3, 100),
        )
    ]

    payload = compact_dream_payload(dreams, [1001], "teacher", {"unique_dreams": 1})

    assert payload["teacher_sha256"] == "teacher"
    assert payload["generation_seeds"] == [1001]
    assert payload["dreams"][0]["token_ids"] == [3, 4, 9]
    assert payload["dreams"][0]["text"] == "[U][A]x"
    assert "teacher_logits" not in str(payload)
    assert payload["ephemeral_teacher_logit_bytes"] == 1200


def test_curve_summary_uses_first_acquisition_peak_and_paired_retention():
    curve = [
        {"cycle": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
        {"cycle": 1, "to_learn_accuracy": 0.2, "not_to_forget_accuracy": 0.9},
        {"cycle": 2, "to_learn_accuracy": 0.2, "not_to_forget_accuracy": 0.8},
    ]

    assert curve_summary(curve) == {
        "top_accuracy": 0.2,
        "cycle": 1,
        "not_to_forget_accuracy": 0.9,
        "total_knowledge": 1.1,
    }


def test_write_cycle_result_includes_itself_in_artifact_bytes(tmp_path: Path):
    (tmp_path / "trainable.pt").write_bytes(b"adapter")
    result: dict[str, object] = {"cycle": 1}

    _write_cycle_result(tmp_path, result)

    actual = sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())
    assert result["artifact_bytes"] == actual
