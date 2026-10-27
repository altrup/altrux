import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.lama_ckl import runner
from experiments.lama_ckl.protocol import WAKE_FRAME, WAKE_FRAMES, lama_dream_diagnostics
from experiments.lama_ckl.runner import (
    _BoxSteps,
    _resume,
    _same_run,
    _save_trainable,
    _write_epoch_result,
    build_parser,
    compact_dream_payload,
    curve_summary,
    cycles_per_epoch,
    draw_wake_frame,
    parse_args,
    run_epochs,
    treatment_settings,
)
from experiments.lama_ckl.split import pinned_warmstart_sha
from training.master_weights import MasterAdamW, MasterOptimizer


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
    def __init__(self, dream_shas=("a", "b"), fail_after=None, wake_invariants=None):
        self.dream_shas = dream_shas
        self.fail_after = fail_after
        self.wake_invariants = wake_invariants
        self.wakes: list[tuple[int, object]] = []
        self.cycles: list[tuple[int, int]] = []
        self.evaluated: list[int] = []
        self.trained: list[int] = []
        self.restored: list[str] = []
        self.distilled = 0

    def wake(self, rows, state, epoch, cycle):
        self.wakes.append((len(rows), state))
        self.cycles.append((epoch, cycle))
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
            "dreams": [{"sha256": sha} for sha in self.dream_shas],
            "set_sha256": "set",
            "diagnostics": lama_dream_diagnostics(dreams, rows, [7]),
        }

    def distil(self, dreams, label):
        if self.distilled == self.fail_after:
            raise RuntimeError("distillation failed")
        self.distilled += 1
        return {
            "optimizer_steps": 1,
            "token_gradients": 3,
            "generated_tokens": 1,
            "loss": 0.5,
            "dream_losses": [0.5, None],
            "max_dream_loss": 0.5,
            "scored_dreams": 1,
            "grad_norm": 2.0,
            "clipped": False,
        }

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

    def restore_cycle(self, directory):
        self.restored.append(directory.name)

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
    cycle_result = json.loads((tmp_path / "epoch-01" / "cycle-002" / "result.json").read_text())
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
        run_epochs(tmp_path, _settings("dream"), ROWS, FakeSteps(fail_after=0), [])

    cycle = tmp_path / "work" / "epoch-01" / "cycle-000.tmp"
    assert (cycle / "wake.json").exists() and (cycle / "dreams.json").exists()
    assert not (tmp_path / "epoch-01").exists()


def test_broken_wake_invariants_refuse_after_writing_artifacts(tmp_path: Path):
    steps = FakeSteps(wake_invariants={"turns": 2, "internal_eoc": 1})

    with pytest.raises(RuntimeError):
        run_epochs(tmp_path, _settings("dream"), ROWS, steps, [])

    cycle = tmp_path / "work" / "epoch-01" / "cycle-000.tmp"
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
    assert [path.name for path in (tmp_path / "epoch-01").glob("cycle-*")] == ["cycle-000"]
    assert (tmp_path / "epoch-01" / "cycle-000" / "trainable.pt").exists()
    assert not (tmp_path / "epoch-01" / "cycle-000" / "result.json").exists()
    assert (tmp_path / "epoch-01" / "result.json").exists()
    assert not (tmp_path / "epoch-02").exists()


def test_max_hours_stops_after_the_cycle_that_crosses_the_cap(tmp_path: Path, monkeypatch):
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

    assert sorted(path.name for path in tmp_path.glob("epoch-*")) == ["epoch-00"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert [row["epoch"] for row in summary["curve"]] == [0]
    assert summary["cycles_run"] == 1
    assert len(curve) == 1

    curve = run_epochs(tmp_path, _settings("dream", epochs=1), ROWS, FakeSteps(), curve)
    assert len(curve[1]["treatment"]["cycles"]) == 3


def test_epochs_flag_defaults_to_thirty():
    assert build_parser().parse_args(["--arm", "dream"]).epochs == 30
    assert build_parser().parse_args(["--arm", "dream", "--epochs", "1"]).epochs == 1


def test_dream_batch_size_defaults_to_dreams_per_cycle():
    assert parse_args(["--arm", "dream", "--dreams-per-cycle", "7"]).dream_batch_size == 7
    explicit = ["--arm", "dream", "--dream-batch-size", "2"]
    assert parse_args(explicit).dream_batch_size == 2


USER, ASST, EOC, EOS, VOCAB = 1, 2, 3, 4, 8


class State:
    def __init__(self, conv, ssm):
        self.conv_states, self.ssm_states = [conv], [ssm]


class StateModel(torch.nn.Module):
    """Records the state of every forward; logits never pick EOC."""

    def __init__(self):
        super().__init__()
        table = torch.randn(VOCAB, VOCAB, generator=torch.Generator().manual_seed(0))
        table[:, EOC] = -1e4
        self.table = torch.nn.Parameter(table)
        self.states: list[object] = []

    def forward(self, tokens, state=None):
        self.states.append(state)
        return self.table[tokens], state


class Tokenizer:
    eos_token_id, pad_token_id = EOS, 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [EOC]}

    def convert_tokens_to_ids(self, token):
        return {"[U]": USER, "[A]": ASST, "<EOC>": EOC}[token]

    def decode(self, ids):
        return f"<{ids[0]}>"


BOX_SETTINGS = {
    "train_batch_size": 2,
    "dream_tokens": 4,
    "lora_rank": 1,
    "lora_alpha": 1.0,
    "wake_frame": {"mode": "single"},
    "reply_temperature": 0.7,
    "dream_temperature": 0.7,
    "dream_loss_weights": "turn",
    "clip_norm": None,
}


def _box(model, optimizer=None, dreams=3, **settings):
    return _BoxSteps(
        args=SimpleNamespace(seed=42, dream_batch_size=dreams),
        settings={**BOX_SETTINGS, "dreams_per_cycle": dreams, **settings},
        model=model,
        tokenizer=Tokenizer(),
        optimizer=optimizer or torch.optim.SGD(model.parameters(), lr=0.0),
        markers=("[U]", "[A]", "<EOC>"),
        learned=ROWS,
        retained=ROWS,
        device="cpu",
        teacher_sha="warm",
    )


def test_student_state_is_empty():
    model = StateModel()
    steps = _box(model)
    wake_state = State(torch.arange(2.0).view(1, 2), torch.arange(3.0).view(1, 3) + 1)
    wake = {"transcript_token_ids": [7]}

    dreams, _ = steps.dream(1, 0, ROWS[:2], wake_state, wake)
    first = model.states[0]
    assert torch.equal(first.conv_states[0], wake_state.conv_states[0].repeat(3, 1))
    assert torch.equal(first.ssm_states[0], wake_state.ssm_states[0].repeat(3, 1))

    model.states.clear()
    steps.distil(dreams, "test")
    assert model.states == [None]


def test_dream_losses_recorded_and_floor_warning(tmp_path: Path, capsys):
    run_epochs(tmp_path, {**_settings("dream", epochs=1), "cycles": 1}, ROWS, FakeSteps(), [])
    recorded = json.loads((tmp_path / "epoch-01" / "cycle-000" / "distil.json").read_text())
    assert recorded == {
        "loss": 0.5,
        "dream_losses": [0.5, None],
        "max_dream_loss": 0.5,
        "scored_dreams": 1,
        "optimizer_steps": 1,
        "token_gradients": 3,
        "generated_tokens": 1,
        "grad_norm": 2.0,
        "clipped": False,
        "subjects_mentioned": 0,
    }
    epoch = json.loads((tmp_path / "epoch-01" / "result.json").read_text())
    cycle = epoch["treatment"]["cycles"][0]
    assert {key: cycle[key] for key in recorded} == recorded

    model = StateModel()
    steps = _box(model)
    wake_state = State(torch.zeros(1, 2), torch.zeros(1, 3))
    dreams, _ = steps.dream(1, 0, ROWS[:2], wake_state, {"transcript_token_ids": [7]})
    capsys.readouterr()

    # The student matches the teacher it was generated from: every loss is zero.
    distilled = steps.distil(dreams, "floor")
    assert distilled["max_dream_loss"] == pytest.approx(0.0, abs=1e-6)
    assert "WARNING" in capsys.readouterr().out

    for dream in dreams:
        dream.teacher_logits = torch.randn(dream.teacher_logits.shape) * 10
    distilled = steps.distil(dreams, "signal")
    assert distilled["max_dream_loss"] > 0.01
    assert "WARNING" not in capsys.readouterr().out

    for dream in dreams:
        dream.dream_ids, dream.token_texts = dream.dream_ids[:1], dream.token_texts[:1]
        dream.teacher_logits = dream.teacher_logits[:1]
    distilled = steps.distil(dreams, "empty")
    assert distilled["dream_losses"] == [None] * 3 and distilled["max_dream_loss"] is None
    out = capsys.readouterr().out
    assert "max n/a" in out and "WARNING" not in out


def test_fp32_master_weights_saved_and_required_on_resume(tmp_path: Path):
    model = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)
    optimizer = MasterAdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    _save_trainable(model, tmp_path / "trainable.pt", 4, 8.0, optimizer)
    saved = torch.load(tmp_path / "trainable.pt", weights_only=True)
    assert saved["weight"].dtype == torch.float32

    with pytest.raises(FileNotFoundError):
        _resume(model, optimizer, tmp_path)
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    _resume(model, optimizer, tmp_path)


ONE_DOC_ROWS = ROWS  # five documents, one per wake: five cycles


def test_eval_every_scores_on_the_rule_and_once_on_the_last_cycle(tmp_path: Path):
    steps = FakeSteps()
    settings = {**_settings("dream", epochs=1, docs_per_wake=1), "eval_every": 2}

    curve = run_epochs(tmp_path, settings, ONE_DOC_ROWS, steps, [])

    assert steps.evaluated == [0, 1, 1, 1]
    epoch = tmp_path / "epoch-01"
    assert sorted(path.parent.name for path in epoch.glob("cycle-*/result.json")) == [
        "cycle-001",
        "cycle-003",
        "cycle-004",
    ]
    assert len(list(epoch.glob("cycle-*/trainable.pt"))) == 5
    assert curve[1]["to_learn_accuracy"] == pytest.approx(0.1)


def test_killed_run_resumes_at_the_next_cycle(tmp_path: Path):
    settings = _settings("dream", epochs=1, docs_per_wake=1)
    with pytest.raises(RuntimeError):
        run_epochs(tmp_path, settings, ONE_DOC_ROWS, FakeSteps(fail_after=2), [])
    work = tmp_path / "work" / "epoch-01"
    assert sorted(path.name for path in work.glob("cycle-*")) == [
        "cycle-000",
        "cycle-001",
        "cycle-002.tmp",
    ]

    steps = FakeSteps()
    curve = run_epochs(tmp_path, settings, ONE_DOC_ROWS, steps, curve_from(tmp_path))

    assert steps.restored == ["cycle-001"]
    assert steps.cycles == [(1, 2), (1, 3), (1, 4)]
    cycles = curve[1]["treatment"]["cycles"]
    assert [cycle["cycle"] for cycle in cycles] == [0, 1, 2, 3, 4]
    assert curve[1]["treatment"]["optimizer_steps"] == 5
    assert cycles[0]["to_learn_accuracy"] == pytest.approx(0.1)
    assert not (tmp_path / "work" / "epoch-01").exists()


def curve_from(output: Path) -> list[dict[str, object]]:
    return [
        json.loads((path / "result.json").read_text()) for path in sorted(output.glob("epoch-*"))
    ]


def test_cycles_per_run_stops_with_a_summary_and_resumes(tmp_path: Path):
    settings = {**_settings("dream", epochs=1, docs_per_wake=1), "cycles_per_run": 2}
    steps = FakeSteps()

    run_epochs(tmp_path, settings, ONE_DOC_ROWS, steps, [])

    assert steps.cycles == [(1, 0), (1, 1)]
    assert not (tmp_path / "epoch-01").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["cycles_run"] == 2
    assert [row["epoch"] for row in summary["curve"]] == [0]

    steps = FakeSteps()
    curve = run_epochs(
        tmp_path, {**settings, "cycles_per_run": 4}, ONE_DOC_ROWS, steps, curve_from(tmp_path)
    )
    assert steps.cycles == [(1, 2), (1, 3)]
    assert json.loads((tmp_path / "summary.json").read_text())["cycles_run"] == 4
    assert len(curve) == 1

    curve = run_epochs(
        tmp_path, {**settings, "cycles_per_run": None}, ONE_DOC_ROWS, FakeSteps(), curve
    )
    assert len(curve[1]["treatment"]["cycles"]) == 5
    assert "cycles_run" not in json.loads((tmp_path / "summary.json").read_text())


def _args(*extra: str):
    return parse_args(["--arm", "dream", *extra])


CELL = (
    "--dream-temperature",
    "1.0",
    "--wake-frames",
    "set",
    "--dream-loss-weights",
    "uniform",
    "--fp32-forward",
    "--optimizer",
    "sgd",
    "--lr",
    "0.05",
    "--clip-norm",
    "3.0",
    "--eval-every",
    "5",
    "--cycles-per-run",
    "100",
)


def test_run_settings_carry_the_cell_flags():
    settings = treatment_settings(_args(*CELL))

    assert settings["reply_temperature"] == settings["dream_temperature"] == 1.0
    assert settings["optimizer"] == "sgd"
    assert (settings["dream_lr"], settings["clip_norm"]) == (0.05, 3.0)
    assert (
        settings["wake_frame"]["mode"] == "set" and len(settings["wake_frame"]["set_sha256"]) == 64
    )
    assert settings["dream_loss_weights"] == "uniform" and settings["fp32_forward"] is True
    assert (settings["eval_every"], settings["cycles_per_run"]) == (5, 100)

    default = treatment_settings(_args())
    assert default["dream_temperature"] == 0.7 and default["wake_frame"] == {"mode": "single"}
    assert (default["optimizer"], default["dream_lr"], default["clip_norm"]) == (
        "adamw",
        None,
        None,
    )


def test_resume_guard_ignores_only_the_session_limits():
    settings = treatment_settings(_args(*CELL))
    later = treatment_settings(_args(*CELL[:-1], "500", "--max-hours", "4"))
    assert _same_run(settings, later)
    other_lr = treatment_settings(_args(*CELL[:10], "0.1", *CELL[11:]))
    assert not _same_run(settings, other_lr)


@pytest.mark.parametrize(
    "extra",
    [
        ("--optimizer", "sgd"),
        ("--optimizer", "sgd", "--lr", "0.1"),
        ("--optimizer", "sgd", "--clip-norm", "1.0"),
        ("--clip-norm", "1.0"),
        ("--lr", "0.1"),
        ("--eval-every", "0"),
        ("--cycles-per-run", "2", "--cycles", "1"),
    ],
)
def test_incoherent_optimizer_and_schedule_flags_are_refused(extra):
    with pytest.raises(SystemExit):
        _args(*extra)


def test_sgd_is_refused_outside_the_dream_arm():
    with pytest.raises(SystemExit):
        parse_args(["--arm", "lora", "--optimizer", "sgd", "--lr", "0.1", "--clip-norm", "1"])


def test_set_frames_are_drawn_per_wake_and_single_draws_none(monkeypatch):
    frames: list[str] = []

    def fake_wake(*args, frame, **kwargs):
        frames.append(frame)
        return {"frame": frame}, None

    monkeypatch.setattr(runner, "run_conversational_wake", fake_wake)
    steps = _box(StateModel(), wake_frame={"mode": "set", "set_sha256": "x"})
    drawn = [steps.wake(ROWS[:1], None, 1, cycle)[0]["frame"] for cycle in range(20)]

    assert drawn == frames and set(drawn) <= set(WAKE_FRAMES) and len(set(drawn)) > 5
    assert drawn == [draw_wake_frame("set", 1, cycle) for cycle in range(20)]
    assert draw_wake_frame("set", 2, 0) != draw_wake_frame("set", 1, 0) or (
        draw_wake_frame("set", 2, 1) != draw_wake_frame("set", 1, 1)
    )
    single = _box(StateModel())
    assert {single.wake(ROWS[:1], None, 1, cycle)[0]["frame"] for cycle in range(5)} == {WAKE_FRAME}


def test_distil_passes_the_loss_weights_and_clip(monkeypatch):
    from experiments import dream_distillation

    seen: dict[str, object] = {}
    real = dream_distillation.distill_dream_batch

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(dream_distillation, "distill_dream_batch", spy)
    model = StateModel()
    steps = _box(model, dream_loss_weights="uniform", clip_norm=1e-3)
    dreams, _ = steps.dream(
        1, 0, ROWS[:2], State(torch.zeros(1, 2), torch.zeros(1, 3)), {"transcript_token_ids": [7]}
    )
    for dream in dreams:
        dream.teacher_logits = torch.randn(dream.teacher_logits.shape) * 10

    distilled = steps.distil(dreams, "uniform")

    assert (seen["loss_weights"], seen["clip_norm"]) == ("uniform", 1e-3)
    assert distilled["clipped"] is True and distilled["grad_norm"] > 1e-3


def test_clip_bounds_the_sgd_step_on_the_gradient():
    model = StateModel()
    # Small enough that an fp32 step of 1e-4 is not rounded away.
    model.table.data[:, EOC] = -30.0
    optimizer = MasterOptimizer(model.parameters(), torch.optim.SGD, lr=1.0, momentum=0.0)
    steps = _box(model, optimizer, clip_norm=1e-3)
    dreams, _ = steps.dream(
        1, 0, ROWS[:2], State(torch.zeros(1, 2), torch.zeros(1, 3)), {"transcript_token_ids": [7]}
    )
    for dream in dreams:
        dream.teacher_logits = torch.randn(dream.teacher_logits.shape) * 10
    before = model.table.detach().clone()

    distilled = steps.distil(dreams, "clip")

    assert distilled["clipped"] and distilled["grad_norm"] > 1e-3
    assert (model.table.detach() - before).norm().item() == pytest.approx(1e-3, rel=1e-3)


def test_cycle_saves_adapter_and_optimizer_and_restores_masters_unrounded(tmp_path: Path):
    model = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)
    optimizer = MasterAdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    steps = _box(model, optimizer)
    exact = torch.tensor([[1.0 + 2**-12, 0.5], [0.25, -1.0 - 2**-12]])
    with torch.no_grad():
        optimizer.masters[0].copy_(exact)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    saved = optimizer.masters[0].detach().clone()
    moments = optimizer.state_dict()["optimizer"]["state"][0]["exp_avg"].clone()

    steps.save_cycle(tmp_path)
    assert (tmp_path / "trainable.pt").exists() and (tmp_path / "optimizer.pt").exists()

    fresh = MasterAdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    restored = _box(model, fresh)
    restored.restore_cycle(tmp_path)

    assert torch.equal(fresh.masters[0], saved)
    assert torch.equal(fresh.state_dict()["optimizer"]["state"][0]["exp_avg"], moments)
    assert restored.teacher_sha == steps.teacher_sha


def test_sgd_cycle_writes_no_optimizer_state_and_restores_from_the_adapter(tmp_path: Path):
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = MasterOptimizer(model.parameters(), torch.optim.SGD, lr=1.0, momentum=0.0)
    steps = _box(model, optimizer)
    steps.save_cycle(tmp_path)
    saved = model.weight.detach().clone()
    assert not (tmp_path / "optimizer.pt").exists()

    with torch.no_grad():
        model.weight.zero_()
    steps.restore_cycle(tmp_path)
    assert torch.equal(model.weight, saved)
