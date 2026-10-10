import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.lama_ckl.protocol import lama_dream_diagnostics
from experiments.lama_ckl.runner import (
    _write_epoch_result,
    compact_dream_payload,
    curve_summary,
    cycles_per_epoch,
    run_epochs,
)
from experiments.lama_ckl.split import pinned_warmstart_sha


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
        {"epoch": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
        {"epoch": 1, "to_learn_accuracy": 0.2, "not_to_forget_accuracy": 0.9},
        {"epoch": 2, "to_learn_accuracy": 0.2, "not_to_forget_accuracy": 0.8},
    ]

    assert curve_summary(curve) == {
        "top_accuracy": 0.2,
        "epoch": 1,
        "not_to_forget_accuracy": 0.9,
        "total_knowledge": 1.1,
    }


def test_write_epoch_result_includes_itself_in_artifact_bytes(tmp_path: Path):
    (tmp_path / "trainable.pt").write_bytes(b"adapter")
    result: dict[str, object] = {"epoch": 1}

    _write_epoch_result(tmp_path, result)

    actual = sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())
    assert result["artifact_bytes"] == actual


class FakeSteps:
    def __init__(self, dream_shas=("a", "b"), fail_distil=False, wake_invariants=None):
        self.dream_shas = dream_shas
        self.fail_distil = fail_distil
        self.wake_invariants = wake_invariants
        self.wakes: list[tuple[int, object]] = []
        self.carried: list[object] = []
        self.evaluated: list[int] = []
        self.trained: list[int] = []

    def wake(self, rows, state):
        self.wakes.append((len(rows), state))
        invariants = self.wake_invariants or {"turns": len(rows), "internal_eoc": 0}
        wake = {
            "invariants": invariants,
            "forced_closes": 1,
            "transcript_sha256": f"wake-{len(self.wakes)}",
            "transcript_token_ids": [7] * len(rows),
        }
        return wake, object()

    def dream(self, epoch, cycle, rows, open_state, wake):
        dreams = [
            SimpleNamespace(
                token_texts=["x"], dream_ids=[1, 2], prefix_len=1, stop_reason="eoc", dream_sha=sha
            )
            for sha in self.dream_shas
        ]
        return dreams, {
            "set_sha256": "set",
            "diagnostics": lama_dream_diagnostics(dreams, rows, [7]),
        }

    def carry(self, open_state):
        state = object()
        self.carried.append(state)
        return state

    def distil(self, dreams, open_state, label):
        if self.fail_distil:
            raise RuntimeError("distillation failed")
        return {"optimizer_steps": len(dreams), "token_gradients": 3, "generated_tokens": 1}

    def train_documents(self, epoch):
        self.trained.append(epoch)
        return {"optimizer_steps": 1, "token_gradients": 2, "review_tokens": 0}

    def evaluate(self, epoch):
        self.evaluated.append(epoch)
        return {"epoch": epoch, "to_learn_accuracy": 0.1 * epoch, "not_to_forget_accuracy": 1.0}

    def save(self, directory, result):
        (directory / "trainable.pt").write_bytes(b"adapter")


ROWS = [{"subject": f"S{index}", "object": f"O{index}", "evidence": "e"} for index in range(5)]


def _settings(arm, epochs=2, docs_per_wake=2):
    return {"arm": arm, "seed": 42, "epochs": epochs, "docs_per_wake": docs_per_wake}


def test_cycles_per_epoch_rounds_up():
    assert cycles_per_epoch(500, 10) == 50
    assert cycles_per_epoch(5, 2) == 3


def test_altrux_epoch_wakes_in_slices_and_carries_state_only_within_an_epoch(tmp_path: Path):
    steps = FakeSteps()

    run_epochs(tmp_path, _settings("altrux"), ROWS, steps, [])

    assert [size for size, _ in steps.wakes] == [2, 2, 1, 2, 2, 1]
    states = [state for _, state in steps.wakes]
    assert states[0] is None and states[3] is None
    assert states[1] is steps.carried[0] and states[2] is steps.carried[1]
    assert states[4] is steps.carried[3]


def test_evaluation_runs_once_per_epoch_not_per_cycle(tmp_path: Path):
    steps = FakeSteps()

    curve = run_epochs(tmp_path, _settings("altrux"), ROWS, steps, [])

    assert steps.evaluated == [0, 1, 2]
    assert [row["epoch"] for row in curve] == [0, 1, 2]
    assert sorted(path.name for path in tmp_path.glob("epoch-*")) == [
        "epoch-00",
        "epoch-01",
        "epoch-02",
    ]
    assert len(list((tmp_path / "epoch-01").glob("cycle-*/wake.json"))) == 3
    assert json.loads((tmp_path / "summary.json").read_text())["checkpoint"]["epoch"] == 2


def test_duplicate_dreams_train_and_are_counted(tmp_path: Path):
    steps = FakeSteps(dream_shas=("a", "a"))

    curve = run_epochs(tmp_path, _settings("altrux", epochs=1), ROWS, steps, [])

    cycles = curve[1]["treatment"]["cycles"]
    assert all(cycle["duplicate_dreams"] == 1 for cycle in cycles)
    assert curve[1]["treatment"]["optimizer_steps"] == 2 * len(cycles)


@pytest.mark.parametrize("arm", ["lora", "mix-review"])
def test_document_arms_never_wake(tmp_path: Path, arm: str):
    steps = FakeSteps()

    curve = run_epochs(tmp_path, _settings(arm), ROWS, steps, [])

    assert steps.wakes == []
    assert steps.trained == [1, 2]
    assert curve[1]["wake_sha256"] is None and curve[1]["wake_tokens"] is None


def test_frozen_arm_is_one_evaluation_and_no_wake(tmp_path: Path):
    steps = FakeSteps()

    curve = run_epochs(tmp_path, _settings("frozen", epochs=0), ROWS, steps, [])

    assert steps.wakes == [] and steps.trained == []
    assert steps.evaluated == [0]
    assert len(curve) == 1
    assert (tmp_path / "summary.json").exists()


def test_cycle_artifacts_are_written_before_a_later_step_fails(tmp_path: Path):
    with pytest.raises(RuntimeError):
        run_epochs(tmp_path, _settings("altrux"), ROWS, FakeSteps(fail_distil=True), [])

    cycle = next((tmp_path / "work").glob("epoch-01.*/cycle-00"))
    assert (cycle / "wake.json").exists() and (cycle / "dreams.json").exists()
    assert not (tmp_path / "epoch-01").exists()


def test_broken_wake_invariants_refuse_after_writing_artifacts(tmp_path: Path):
    steps = FakeSteps(wake_invariants={"turns": 2, "internal_eoc": 1})

    with pytest.raises(RuntimeError):
        run_epochs(tmp_path, _settings("altrux"), ROWS, steps, [])

    cycle = next((tmp_path / "work").glob("epoch-01.*/cycle-00"))
    assert (cycle / "wake.json").exists()
    assert not (cycle / "dreams.json").exists()
    assert steps.carried == []


def test_resume_starts_at_the_epoch_after_the_curve(tmp_path: Path):
    steps = FakeSteps()
    curve = [{"epoch": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0}]

    run_epochs(tmp_path, _settings("altrux", epochs=1), ROWS, steps, curve)

    assert steps.evaluated == [1]
    assert steps.wakes[0][1] is None


def test_unset_warmstart_pin_refuses(tmp_path: Path):
    pin = tmp_path / "warmstart.sha256"
    pin.write_text("unset\n")

    with pytest.raises(SystemExit):
        pinned_warmstart_sha(pin)

    pin.write_text("a" * 64 + "\n")
    assert pinned_warmstart_sha(pin) == "a" * 64
