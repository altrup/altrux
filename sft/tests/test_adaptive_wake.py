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


def test_live_wake_records_generator_provenance_and_token_usage(tmp_path):
    script = tmp_path / "generator.py"
    script.write_text(
        "import json, sys; r=json.load(sys.stdin); print(json.dumps("
        "{'message': 'user', 'session_id': r.get('session_id', 's'), "
        "'resume_status': 'resumed' if r.get('session_id') else 'started', "
        "'token_usage': {'input': 3, 'output': 2}}))"
    )
    generator = CommandUserGenerator(
        [sys.executable, str(script)],
        provenance={"provider": "fake", "model": "fake-1", "version": "1.2"},
    )
    harness = LiveWakeHarness(
        WakePlan.from_dict({"turn_count": 4, "injection_turns": [1, 2, 3, 4]}), generator, tmp_path,
    )

    artifact = harness.run("replay", 1, [("a", "1 2 3 4 5")] * 4,
                           lambda _: "reply", "test", None)

    assert artifact["generator"]["provider"] == "fake"
    assert artifact["turns"][0]["token_usage"] == {"input": 3, "output": 2}


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


def test_manifest_requires_all_frozen_live_wake_inputs(tmp_path):
    from adaptive_wake import load_experiment_manifest

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"arms": ["replay", "nosleep", "sft-ref"], "wakes": 6, "seeds": [1, 2, 3]}))

    with pytest.raises(AdaptiveWakeError, match="wakes"):
        load_experiment_manifest(path)

    path.write_text(json.dumps({
        "arms": ["replay", "nosleep", "sft-ref"], "wakes": 6, "seeds": [1, 2, 3],
        "generator": {"command": ["adapter"], "provider": "fake", "model": "fake-1", "version": "1"},
        "output_root": "out", "batch_sizes": {"dream": 2, "probe": 2, "battery": 2},
        "wakes": [{
            "scenario": f"wake-{wake}",
            "facts": [{"entity": f"e{wake}-{fact}", "code": f"{wake}{fact}000"} for fact in range(4)],
            "turn_count": 4, "injection_turns": [1, 2, 3, 4],
        } for wake in range(1, 7)],
    }))

    manifest = load_experiment_manifest(path)
    assert manifest.config.wakes == 6 and manifest.generator["model"] == "fake-1"


def test_fake_backbone_smoke_runs_all_registered_arms_and_wakes(tmp_path):
    import subprocess

    completed = subprocess.run(
        [sys.executable, "adaptive_wake_smoke.py", "--out", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert len(list(tmp_path.glob("*_w*.json"))) == 19
    shared = json.loads((tmp_path / "shared_w1.json").read_text())
    first = [json.loads((tmp_path / f"{arm}_w1.json").read_text())
             for arm in ("replay", "nosleep", "sft-ref")]
    assert {item["transcript_sha256"] for item in first} == {shared["transcript_sha256"]}
    assert {item["state_sha256"] for item in first} == {shared["state_sha256"]}


def test_multisleep_coordinator_sleeps_before_each_arm_specific_later_wake(tmp_path):
    from adaptive_wake import MultiSleepCoordinator

    script = tmp_path / "generator.py"
    script.write_text(
        "import json, sys; r=json.load(sys.stdin); s=r.get('session_id', 's')\n"
        "print(json.dumps({'message': r['goal'], 'session_id': s, "
        "'resume_status': 'resumed' if r.get('session_id') else 'started'}))\n"
    )
    harness = LiveWakeHarness(
        WakePlan.from_dict({"turn_count": 4, "injection_turns": [1, 2, 3, 4]}),
        CommandUserGenerator([sys.executable, str(script)]), tmp_path,
    )
    wake_states: list[tuple[str, int, int]] = []
    sleeps: list[tuple[str, int, int]] = []

    def wake(arm: str, wake: int, state: int, facts: list[tuple[str, str]], scenario: str):
        artifact = harness.run(
            arm, wake, facts, lambda user: wake_states.append((arm, wake, state)) or user,
            scenario, None, state_metadata={"state": state},
        )
        return artifact, state + 1

    coordinator = MultiSleepCoordinator(("replay", "nosleep", "sft-ref"), wake, lambda state: state, 0, harness.store)
    coordinator.run(
        facts_for_wake=lambda wake: [(f"e{wake}-{i}", f"{wake}{i}000") for i in range(4)],
        scenario_for_wake=lambda wake: f"scenario-{wake}",
        sleep=lambda arm, wake, state, artifact: sleeps.append((arm, wake, state)) or state + 10,
        probe=lambda arm, wake, state, artifact: None,
    )

    assert sleeps[:3] == [("replay", 1, 1), ("nosleep", 1, 1), ("sft-ref", 1, 1)]
    assert {(arm, state) for arm, wake, state in wake_states if wake == 2} == {
        ("replay", 11), ("nosleep", 11), ("sft-ref", 11)
    }


def test_manifest_runtime_records_every_arm_wake_and_batch_topology(tmp_path):
    from adaptive_wake import ExperimentRuntime, load_experiment_manifest

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "arms": ["replay", "nosleep", "sft-ref"], "seeds": [1, 2, 3],
        "generator": {"command": ["fake"], "provider": "fake", "model": "fake", "version": "1"},
        "output_root": str(tmp_path / "out"), "batch_sizes": {"dream": 2, "probe": 3, "battery": 4},
        "wakes": [{"scenario": str(i), "facts": [{"entity": f"e{i}{j}", "code": "12345"} for j in range(4)],
                   "turn_count": 4, "injection_turns": [1, 2, 3, 4]} for i in range(6)],
    }))
    manifest = load_experiment_manifest(manifest_path)
    runtime = ExperimentRuntime(manifest, initial_state=0, fork_state=lambda state: state,
                                wake=lambda arm, wake, state, facts, scenario: ({"arm": arm, "wake": wake}, state + 1),
                                sleep=lambda arm, wake, state, artifact: state + 1,
                                probe=lambda arm, wake, state, artifact: {"margin": state})

    result = runtime.run(seed=1)

    assert len(result["records"]) == 18
    assert result["execution"]["batch_sizes"] == {"dream": 2, "probe": 3, "battery": 4}
