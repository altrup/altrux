import json
import sys

import pytest

from adaptive_wake import (
    AdaptiveWakeError,
    CommandUserGenerator,
    ExperimentConfig,
    LiveWakeHarness,
    WakePlan,
    load_wake_plan,
)


def test_wake_plan_requires_an_explicit_length_and_injection_schedule():
    with pytest.raises(AdaptiveWakeError, match="turn_count"):
        WakePlan.from_dict({"injection_turns": [1, 2, 3, 4]})

    with pytest.raises(AdaptiveWakeError, match="four"):
        WakePlan.from_dict({"turn_count": 8, "injection_turns": [1, 2]})


def test_live_wake_stores_the_realized_turns_and_refuses_a_resume_reset(tmp_path):
    script = tmp_path / "generator.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "if request.get('session_id') == 'lost': raise SystemExit(2)\n"
        "print(json.dumps({'message': request['goal'], 'session_id': request.get('session_id', 's1'), 'resume_status': 'resumed' if request.get('session_id') else 'started'}))\n"
    )
    plan = WakePlan.from_dict({"turn_count": 4, "injection_turns": [1, 2, 3, 4]})
    harness = LiveWakeHarness(plan, CommandUserGenerator([sys.executable, str(script)]), tmp_path)

    artifact = harness.run(
        arm="replay", wake=1, facts=[("a", "1 2 3 4 5")] * 4,
        local_reply=lambda message: f"reply:{message}", scenario="test", session_id=None,
    )

    assert [turn["user"] for turn in artifact["turns"]] == [
        "State that a has code 1 2 3 4 5.",
    ] * 4
    assert json.loads((tmp_path / "replay_w1.json").read_text())["transcript_sha256"] == artifact["transcript_sha256"]

    with pytest.raises(AdaptiveWakeError, match="resume"):
        harness.run("replay", 2, [("a", "1 2 3 4 5")] * 4, lambda _: "reply", "test", "lost")


def test_live_wake_finalizes_state_metadata_after_the_last_reply(tmp_path):
    script = tmp_path / "generator.py"
    script.write_text("import json, sys; r=json.load(sys.stdin); print(json.dumps({'message': 'user', 'session_id': r.get('session_id', 's'), 'resume_status': 'resumed' if r.get('session_id') else 'started'}))")
    harness = LiveWakeHarness(WakePlan.from_dict({"turn_count": 4, "injection_turns": [1, 2, 3, 4]}),
                              CommandUserGenerator([sys.executable, str(script)]), tmp_path)
    calls = []
    artifact = harness.run("replay", 1, [("a", "1 2 3 4 5")] * 4,
                           lambda _: calls.append(1) or "reply", "test", None,
                           state_metadata=lambda: {"turns": len(calls)})

    assert artifact["state_metadata"] == {"turns": 4}


def test_load_wake_plan_reads_the_freeze_file(tmp_path):
    path = tmp_path / "wake.json"
    path.write_text(json.dumps({"turn_count": 5, "injection_turns": [1, 2, 3, 5]}))

    assert load_wake_plan(path).turn_count == 5


def test_registered_experiment_shape_is_three_arms_six_wakes_three_seeds():
    config = ExperimentConfig.from_dict({
        "arms": ["replay", "nosleep", "sft-ref"], "wakes": 6, "seeds": [1, 2, 3],
    })

    assert config.arms == ("replay", "nosleep", "sft-ref")

    with pytest.raises(AdaptiveWakeError, match="six"):
        ExperimentConfig.from_dict({"arms": ["replay", "nosleep", "sft-ref"], "wakes": 4, "seeds": [1, 2, 3]})


def test_fake_backbone_smoke_runs_all_registered_arms_and_wakes(tmp_path):
    import subprocess

    completed = subprocess.run(
        [sys.executable, "adaptive_wake_smoke.py", "--out", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert len(list(tmp_path.glob("*_w*.json"))) == 18
