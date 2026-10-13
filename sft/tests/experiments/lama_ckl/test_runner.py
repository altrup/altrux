import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.lama_ckl.protocol import lama_dream_diagnostics
from experiments.lama_ckl.runner import (
    _write_epoch_result,
    build_parser,
    compact_dream_payload,
    curve_summary,
    cycles_per_epoch,
    parse_args,
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
        self.evaluated: list[int] = []
        self.trained: list[int] = []
        self.distilled = 0

    def wake(self, rows, state, epoch, cycle):
        self.wakes.append((len(rows), state))
        invariants = self.wake_invariants or {"turns": len(rows), "internal_eoc": 0}
        wake = {
            "invariants": invariants,
            "forced_closes": 1,
            "role_marker_closes": 1,
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

    def distil(self, dreams, open_state, label):
        if self.fail_distil:
            raise RuntimeError("distillation failed")
        self.distilled += 1
        return {"optimizer_steps": 1, "token_gradients": 3, "generated_tokens": 1, "loss": 0.5}

    def train_documents(self, epoch):
        self.trained.append(epoch)
        return {"optimizer_steps": 1, "token_gradients": 2, "review_tokens": 0}

    def evaluate(self, epoch):
        self.evaluated.append(epoch)
        return {
            "epoch": epoch,
            "to_learn_accuracy": 0.1 * epoch,
            "not_to_forget_accuracy": 1.0,
            "evaluation_seconds": 0.0,
        }

    def save_cycle(self, directory):
        (directory / "trainable.pt").write_bytes(b"adapter")
        return "cycle-sha"

    def save(self, directory, result):
        (directory / "trainable.pt").write_bytes(b"adapter")


ROWS = [{"subject": f"S{index}", "object": f"O{index}", "evidence": "e"} for index in range(5)]


def _settings(arm, epochs=2, docs_per_wake=2):
    return {"arm": arm, "seed": 42, "epochs": epochs, "docs_per_wake": docs_per_wake}


def test_cycles_per_epoch_rounds_up():
    assert cycles_per_epoch(500, 10) == 50
    assert cycles_per_epoch(5, 2) == 3


def test_dream_epoch_wakes_in_slices_from_fresh_state_every_cycle(tmp_path: Path):
    steps = FakeSteps()

    run_epochs(tmp_path, _settings("dream"), ROWS, steps, [])

    assert [size for size, _ in steps.wakes] == [2, 2, 1, 2, 2, 1]
    assert all(state is None for _, state in steps.wakes)


def test_dream_saves_and_scores_every_cycle_and_every_epoch(tmp_path: Path):
    steps = FakeSteps()

    curve = run_epochs(tmp_path, _settings("dream"), ROWS, steps, [])

    assert steps.evaluated == [0, 1, 1, 1, 2, 2, 2]
    assert [row["epoch"] for row in curve] == [0, 1, 2]
    assert sorted(path.name for path in tmp_path.glob("epoch-*")) == [
        "epoch-00",
        "epoch-01",
        "epoch-02",
    ]
    assert len(list((tmp_path / "epoch-01").glob("cycle-*/wake.json"))) == 3
    assert len(list((tmp_path / "epoch-01").glob("cycle-*/trainable.pt"))) == 3
    cycle_result = json.loads((tmp_path / "epoch-01" / "cycle-02" / "result.json").read_text())
    assert cycle_result["cycle"] == 2 and cycle_result["epoch"] == 1
    record = curve[1]["treatment"]["cycles"][0]
    for key in (
        "role_marker_closes",
        "to_learn_accuracy",
        "not_to_forget_accuracy",
        "evaluation_seconds",
        "adapter_sha256",
        "loss",
    ):
        assert key in record
    assert record["role_marker_closes"] == 1
    assert json.loads((tmp_path / "summary.json").read_text())["checkpoint"]["epoch"] == 2


@pytest.mark.parametrize("arm", ["lora", "mix-review"])
def test_document_arms_evaluate_once_per_epoch(tmp_path: Path, arm: str):
    steps = FakeSteps()

    run_epochs(tmp_path, _settings(arm), ROWS, steps, [])

    assert steps.evaluated == [0, 1, 2]


def test_duplicate_dreams_train_and_are_counted(tmp_path: Path):
    steps = FakeSteps(dream_shas=("a", "a"))

    curve = run_epochs(tmp_path, _settings("dream", epochs=1), ROWS, steps, [])

    cycles = curve[1]["treatment"]["cycles"]
    assert all(cycle["duplicate_dreams"] == 1 for cycle in cycles)
    assert curve[1]["treatment"]["optimizer_steps"] == len(cycles)


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
        run_epochs(tmp_path, _settings("dream"), ROWS, FakeSteps(fail_distil=True), [])

    cycle = next((tmp_path / "work").glob("epoch-01.*/cycle-00"))
    assert (cycle / "wake.json").exists() and (cycle / "dreams.json").exists()
    assert not (tmp_path / "epoch-01").exists()


def test_broken_wake_invariants_refuse_after_writing_artifacts(tmp_path: Path):
    steps = FakeSteps(wake_invariants={"turns": 2, "internal_eoc": 1})

    with pytest.raises(RuntimeError):
        run_epochs(tmp_path, _settings("dream"), ROWS, steps, [])

    cycle = next((tmp_path / "work").glob("epoch-01.*/cycle-00"))
    assert (cycle / "wake.json").exists()
    assert not (cycle / "dreams.json").exists()
    assert steps.distilled == 0


def test_role_marker_closes_do_not_refuse_the_wake(tmp_path: Path):
    curve = run_epochs(tmp_path, _settings("dream", epochs=1), ROWS, FakeSteps(), [])

    assert curve[1]["treatment"]["role_marker_closes"] == 3


def test_resume_starts_at_the_epoch_after_the_curve(tmp_path: Path):
    steps = FakeSteps()
    curve = [{"epoch": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0}]

    run_epochs(tmp_path, _settings("dream", epochs=1), ROWS, steps, curve)

    assert 0 not in steps.evaluated and steps.evaluated[-1] == 1
    assert steps.wakes[0][1] is None


def test_unset_warmstart_pin_refuses(tmp_path: Path):
    pin = tmp_path / "warmstart.sha256"
    pin.write_text("unset\n")

    with pytest.raises(SystemExit):
        pinned_warmstart_sha(pin)

    pin.write_text("a" * 64 + "\n")
    assert pinned_warmstart_sha(pin) == "a" * 64


def test_cycles_cap_writes_one_truncated_epoch_and_skips_its_evaluation(tmp_path: Path):
    steps = FakeSteps()

    run_epochs(tmp_path, {**_settings("dream", epochs=3), "cycles": 1}, ROWS, steps, [])

    assert steps.evaluated == [0]
    assert len(steps.wakes) == 1
    assert [path.name for path in (tmp_path / "epoch-01").glob("cycle-*")] == ["cycle-00"]
    assert (tmp_path / "epoch-01" / "cycle-00" / "trainable.pt").exists()
    assert not (tmp_path / "epoch-01" / "cycle-00" / "result.json").exists()
    assert (tmp_path / "epoch-01" / "result.json").exists()
    assert not (tmp_path / "epoch-02").exists()


def test_max_hours_stops_after_the_epoch_that_crosses_the_cap(tmp_path: Path, monkeypatch):
    from experiments.lama_ckl import runner

    clock = [0.0]
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    steps = FakeSteps()
    evaluate = steps.evaluate

    def timed_evaluate(epoch):
        clock[0] += 3600.0 * epoch
        return evaluate(epoch)

    steps.evaluate = timed_evaluate

    curve = run_epochs(
        tmp_path, {**_settings("dream", epochs=3), "max_hours": 0.5}, ROWS, steps, []
    )

    assert sorted(path.name for path in tmp_path.glob("epoch-*")) == ["epoch-00", "epoch-01"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert [row["epoch"] for row in summary["curve"]] == [0, 1]
    assert len(curve) == 2


def test_epochs_flag_defaults_to_thirty():
    assert build_parser().parse_args(["--arm", "dream"]).epochs == 30
    assert build_parser().parse_args(["--arm", "dream", "--epochs", "1"]).epochs == 1


def test_dream_batch_size_defaults_to_dreams_per_cycle():
    assert parse_args(["--arm", "dream", "--dreams-per-cycle", "7"]).dream_batch_size == 7
    explicit = ["--arm", "dream", "--dream-batch-size", "2"]
    assert parse_args(explicit).dream_batch_size == 2
