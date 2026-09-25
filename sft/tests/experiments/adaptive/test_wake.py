import json
import sys

import pytest

from experiments.adaptive.manifest import (
    AdaptiveWakeError,
    ExperimentConfig,
    WakePlan,
    load_wake_plan,
)
from experiments.adaptive.wake import CommandUserGenerator, LiveWakeHarness

GOALS = ["advance one", "advance two", "advance three", "close the scene"]


def wake_plan() -> WakePlan:
    return WakePlan.from_dict(
        {"turn_count": 4, "injection_turns": [1, 2, 3, 4], "turn_goals": GOALS}
    )


def test_wake_plan_requires_an_explicit_length_and_injection_schedule():
    with pytest.raises(AdaptiveWakeError, match="turn_count"):
        WakePlan.from_dict({"injection_turns": [1, 2, 3, 4]})

    with pytest.raises(AdaptiveWakeError, match="four"):
        WakePlan.from_dict({"turn_count": 8, "injection_turns": [1, 2]})


def test_wake_plan_requires_one_registered_intent_goal_per_turn():
    with pytest.raises(AdaptiveWakeError, match="turn_goals"):
        WakePlan.from_dict({"turn_count": 4, "injection_turns": [1, 2, 3, 4]})

    plan = WakePlan.from_dict(
        {"turn_count": 4, "injection_turns": [1, 2, 3, 4], "turn_goals": ["g1", "g2", "g3", "g4"]}
    )

    assert plan.turn_goals == ("g1", "g2", "g3", "g4")


def test_live_wake_stores_the_realized_turns_and_refuses_a_resume_reset(tmp_path):
    script = tmp_path / "generator.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "if request.get('session_id') == 'lost': raise SystemExit(2)\n"
        "print(json.dumps({'message': request['goal'], 'session_id': request.get('session_id', 's1'), 'resume_status': 'resumed' if request.get('session_id') else 'started'}))\n"
    )
    plan = wake_plan()
    harness = LiveWakeHarness(plan, CommandUserGenerator([sys.executable, str(script)]), tmp_path)

    artifact = harness.run(
        arm="replay",
        wake=1,
        facts=[("a", "1 2 3 4 5")] * 4,
        local_reply=lambda message, turn: f"reply:{message}",
        scenario="test",
        session_id=None,
    )

    assert [goal in turn["user"] for goal, turn in zip(GOALS, artifact["turns"], strict=True)] == [
        True
    ] * 4
    assert all("1 2 3 4 5" in turn["user"] for turn in artifact["turns"])
    assert (
        json.loads((tmp_path / "replay_w1.json").read_text())["transcript_sha256"]
        == artifact["transcript_sha256"]
    )

    with pytest.raises(AdaptiveWakeError, match="resume"):
        harness.run("replay", 2, [("a", "1 2 3 4 5")] * 4, lambda _, turn: "reply", "test", "lost")


def test_live_wake_refuses_an_injection_that_omits_the_entity_or_code(tmp_path):
    script = tmp_path / "generator.py"
    script.write_text(
        "import json, sys; r=json.load(sys.stdin); print(json.dumps({'message': 'hello', "
        "'session_id': r.get('session_id', 's'), 'resume_status': 'resumed' if r.get('session_id') else 'started'}))"
    )
    harness = LiveWakeHarness(
        wake_plan(),
        CommandUserGenerator([sys.executable, str(script)]),
        tmp_path,
    )

    with pytest.raises(AdaptiveWakeError, match="communicate"):
        harness.run(
            "replay", 1, [("osprey", "1 2 3 4 5")] * 4, lambda _, turn: "reply", "test", None
        )


def test_live_wake_refuses_a_bare_entity_code_dump(tmp_path):
    script = tmp_path / "generator.py"
    script.write_text(
        "import json, sys; r=json.load(sys.stdin); print(json.dumps({'message': 'osprey 1 2 3 4 5', "
        "'session_id': r.get('session_id', 's'), 'resume_status': 'resumed' if r.get('session_id') else 'started'}))"
    )

    with pytest.raises(AdaptiveWakeError, match="natural-language"):
        LiveWakeHarness(
            wake_plan(), CommandUserGenerator([sys.executable, str(script)]), tmp_path
        ).run(
            "replay",
            1,
            [("osprey", "1 2 3 4 5")] * 4,
            lambda _, turn: "reply",
            "test",
            None,
        )


def test_live_wake_finalizes_state_metadata_after_the_last_reply(tmp_path):
    script = tmp_path / "generator.py"
    script.write_text(
        "import json, sys; r=json.load(sys.stdin); print(json.dumps({'message': r['goal'], 'session_id': r.get('session_id', 's'), 'resume_status': 'resumed' if r.get('session_id') else 'started'}))"
    )
    harness = LiveWakeHarness(
        wake_plan(), CommandUserGenerator([sys.executable, str(script)]), tmp_path
    )
    calls = []
    artifact = harness.run(
        "replay",
        1,
        [("a", "1 2 3 4 5")] * 4,
        lambda _, turn: calls.append(1) or "reply",
        "test",
        None,
        state_metadata=lambda: {"turns": len(calls)},
        fact_distances=lambda turns: {"a": len(turns)},
    )

    assert artifact["state_metadata"] == {"turns": 4}
    assert artifact["fact_token_distances"] == {"a": 4}


def test_live_wake_records_generator_provenance_and_token_usage(tmp_path):
    script = tmp_path / "generator.py"
    script.write_text(
        "import json, sys; r=json.load(sys.stdin); print(json.dumps("
        "{'message': r['goal'], 'session_id': r.get('session_id', 's'), "
        "'resume_status': 'resumed' if r.get('session_id') else 'started', "
        "'token_usage': {'input': 3, 'output': 2}}))"
    )
    generator = CommandUserGenerator(
        [sys.executable, str(script)],
        provenance={"provider": "fake", "model": "fake-1", "version": "1.2"},
    )
    harness = LiveWakeHarness(
        wake_plan(),
        generator,
        tmp_path,
    )

    artifact = harness.run(
        "replay", 1, [("a", "1 2 3 4 5")] * 4, lambda _, turn: "reply", "test", None
    )

    assert artifact["generator"]["provider"] == "fake"
    assert artifact["turns"][0]["token_usage"] == {"input": 3, "output": 2}


def test_live_wake_preserves_consumed_token_ids_including_assistant_eos(tmp_path):
    script = tmp_path / "generator.py"
    script.write_text(
        "import json, sys; r=json.load(sys.stdin); print(json.dumps({'message': r['goal'], "
        "'session_id': r.get('session_id', 's'), "
        "'resume_status': 'resumed' if r.get('session_id') else 'started'}))"
    )
    calls = []

    def reply(user, turn):
        index = len(calls)
        value = {
            "text": f"decoded-{index}",
            "prompt_token_ids": [10 + index, 11 + index],
            "assistant_token_ids": [20 + index, 2],
        }
        calls.append(value)
        return value

    artifact = LiveWakeHarness(
        wake_plan(), CommandUserGenerator([sys.executable, str(script)]), tmp_path
    ).run(
        "replay",
        1,
        [(f"e{i}", f"{i} 2 3 4 5") for i in range(4)],
        reply,
        "test",
        None,
    )

    expected = [
        token
        for call in calls
        for key in ("prompt_token_ids", "assistant_token_ids")
        for token in call[key]
    ]
    assert artifact["transcript_token_ids"] == expected
    assert artifact["turns"][0]["assistant_token_ids"][-1] == 2
    assert artifact["transcript_token_sha256"]


def test_sleep_transcript_uses_stored_ids_without_decoding_or_retokenizing(monkeypatch):
    import torch
    from types import SimpleNamespace
    from experiments.adaptive.backend import artifact_transcript_ids, DreamSleepBackend
    from experiments.dreams import distillation

    artifact = {
        "transcript_token_ids": [7, 99, 2],
        "transcript_token_sha256": "260cb17b37d3296338f7e5cd227694dccd87bf66b838a94ade6cd61a38fa36c8",
    }

    assert artifact_transcript_ids(artifact) == [7, 99, 2]

    consumed = []

    class Model:
        def train(self):
            return None

        def eval(self):
            return None

    monkeypatch.setattr(
        distillation,
        "distill_sft",
        lambda model, optimizer, ids, steps, chunk, on_step: consumed.extend(ids[0].tolist()) or 2,
    )
    backend = DreamSleepBackend.__new__(DreamSleepBackend)
    backend.torch, backend.device = torch, torch.device("cpu")
    backend.runtime = {"chunk_len": 8}
    backend.contexts = {"sft-ref": SimpleNamespace(model=Model(), optimizer=object())}
    backend.treatments, backend.topology = {}, []

    assert backend.sleep("sft-ref", 1, object(), artifact) is None
    assert consumed == [7, 99, 2]


def test_local_wake_reply_refuses_an_in_wake_eoc():
    import torch
    from types import SimpleNamespace
    from experiments.adaptive.backend import DreamSleepBackend

    class State:
        def detach(self):
            return self

    class Model:
        def eval(self):
            return None

        def __call__(self, tokens, state=None):
            logits = torch.full((1, tokens.shape[1], 16), -100.0)
            logits[:, -1, 9] = 100.0
            return logits, State()

    backend = DreamSleepBackend.__new__(DreamSleepBackend)
    backend.torch, backend.device = torch, torch.device("cpu")
    backend.runtime = {"reply_tokens": 2, "reply_temperature": 0.0}
    backend.seed = 1
    backend.eoc_id, backend.turn_id = 9, 2
    backend.user_open, backend.asst_open = "[U]", "[A]"
    backend._context = lambda arm: SimpleNamespace(model=Model())
    backend._encode = lambda text: torch.tensor([[1, 3]])
    backend._decode = lambda ids: "decoded"

    with pytest.raises(RuntimeError, match="EOC"):
        backend._sample_reply("replay", 1, 1, "hello", None)


def test_local_wake_reply_refuses_max_token_truncation():
    import torch
    from types import SimpleNamespace
    from experiments.adaptive.backend import DreamSleepBackend

    class State:
        def detach(self):
            return self

    class Model:
        def eval(self):
            return None

        def __call__(self, tokens, state=None):
            logits = torch.full((1, tokens.shape[1], 16), -100.0)
            logits[:, -1, 8] = 100.0
            return logits, State()

    backend = DreamSleepBackend.__new__(DreamSleepBackend)
    backend.torch, backend.device = torch, torch.device("cpu")
    backend.runtime = {"reply_tokens": 1, "reply_temperature": 0.0}
    backend.seed = 1
    backend.eoc_id, backend.turn_id = 9, 2
    backend.user_open, backend.asst_open = "[U]", "[A]"
    backend._context = lambda arm: SimpleNamespace(model=Model())
    backend._encode = lambda text: torch.tensor([[1, 3]])
    backend._decode = lambda ids: "decoded"

    with pytest.raises(RuntimeError, match="reply_tokens"):
        backend._sample_reply("replay", 1, 1, "hello", None)


def test_reply_sampling_is_turn_scoped_across_interruption_resume():
    import torch
    from types import SimpleNamespace
    from experiments.adaptive.backend import DreamSleepBackend

    class State:
        def detach(self):
            return self

    class Model:
        def eval(self):
            return None

        def __call__(self, tokens, state=None):
            logits = torch.full((1, tokens.shape[1], 16), -100.0)
            logits[:, -1, 7:9] = 0.0
            logits[:, -1, 2] = -1.0
            if int(tokens[0, -1]) in (7, 8):
                logits[:, -1] = -100.0
                logits[:, -1, 2] = 100.0
            return logits, State()

    backend = DreamSleepBackend.__new__(DreamSleepBackend)
    backend.torch, backend.device = torch, torch.device("cpu")
    backend.runtime = {"reply_tokens": 2, "reply_temperature": 1.0}
    backend.seed = 11
    backend.manifest = SimpleNamespace(
        config=SimpleNamespace(arms=("replay", "nosleep", "sft-ref"))
    )
    backend.eoc_id, backend.turn_id = 9, 2
    backend.user_open, backend.asst_open = "[U]", "[A]"
    backend._context = lambda arm: SimpleNamespace(model=Model())
    backend._encode = lambda text: torch.tensor([[1, 3]])
    backend._decode = lambda ids: str(ids)

    uninterrupted = [
        backend._sample_reply("replay", 3, turn, "hello", None)[0]["assistant_token_ids"]
        for turn in range(1, 5)
    ]
    torch.manual_seed(999)
    _ = torch.rand(100)
    resumed = [
        backend._sample_reply("replay", 3, turn, "hello", None)[0]["assistant_token_ids"]
        for turn in range(3, 5)
    ]

    assert resumed == uninterrupted[2:]


def test_live_wake_resumes_after_a_persisted_provider_turn(tmp_path):
    class Generator:
        command = ("fake",)
        provenance = {"provider": "fake"}
        calls = 0

        def next_user(self, request):
            self.calls += 1
            return {
                "message": request["goal"],
                "session_id": request.get("session_id", "session"),
                "resume_status": "resumed" if request.get("session_id") else "started",
            }

    generator = Generator()
    harness = LiveWakeHarness(wake_plan(), generator, tmp_path)
    replies = 0

    def interrupted(user, turn):
        nonlocal replies
        replies += 1
        if replies == 2:
            raise RuntimeError("interrupted after provider returned")
        return {"text": "reply", "prompt_token_ids": [replies], "assistant_token_ids": [2]}

    with pytest.raises(RuntimeError, match="interrupted"):
        harness.run(
            "replay", 2, [(f"e{i}", f"{i} 2 3 4 5") for i in range(4)], interrupted, "test", None
        )

    replayed = []
    artifact = harness.run(
        "replay",
        2,
        [(f"e{i}", f"{i} 2 3 4 5") for i in range(4)],
        lambda user, turn: {"text": "reply", "prompt_token_ids": [9], "assistant_token_ids": [2]},
        "test",
        None,
        replay_turn=lambda turn: replayed.append(turn["turn"]),
    )

    assert generator.calls == 4
    assert replayed == [1]
    assert len(artifact["turns"]) == 4
    assert not (tmp_path / "replay_w2.partial.json").exists()


def test_load_wake_plan_reads_the_freeze_file(tmp_path):
    path = tmp_path / "wake.json"
    path.write_text(
        json.dumps(
            {
                "turn_count": 5,
                "injection_turns": [1, 2, 3, 5],
                "turn_goals": [f"goal {i}" for i in range(5)],
            }
        )
    )

    assert load_wake_plan(path).turn_count == 5


def test_registered_experiment_shape_is_three_arms_six_wakes_three_seeds():
    config = ExperimentConfig.from_dict(
        {
            "arms": ["replay", "nosleep", "sft-ref"],
            "wakes": 6,
            "seeds": [1, 2, 3],
        }
    )

    assert config.arms == ("replay", "nosleep", "sft-ref")

    with pytest.raises(AdaptiveWakeError, match="six"):
        ExperimentConfig.from_dict(
            {"arms": ["replay", "nosleep", "sft-ref"], "wakes": 4, "seeds": [1, 2, 3]}
        )

    with pytest.raises(AdaptiveWakeError, match="distinct"):
        ExperimentConfig.from_dict(
            {"arms": ["replay", "nosleep", "sft-ref"], "wakes": 6, "seeds": [1, 1, 2]}
        )


def test_fact_positions_rotate_across_seeds_and_wakes():
    from experiments.adaptive.manifest import counterbalanced_facts

    facts = [(str(index), str(index)) for index in range(4)]

    assert counterbalanced_facts(facts, seed_index=0, wake=1) == facts
    assert counterbalanced_facts(facts, seed_index=1, wake=1) == facts[1:] + facts[:1]
    assert counterbalanced_facts(facts, seed_index=0, wake=2) == facts[1:] + facts[:1]


def test_manifest_requires_all_frozen_live_wake_inputs(tmp_path):
    from experiments.adaptive.manifest import load_experiment_manifest

    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"arms": ["replay", "nosleep", "sft-ref"], "wakes": 6, "seeds": [1, 2, 3]})
    )

    with pytest.raises(AdaptiveWakeError, match="wakes"):
        load_experiment_manifest(path)

    path.write_text(
        json.dumps(
            {
                "arms": ["replay", "nosleep", "sft-ref"],
                "wakes": 6,
                "seeds": [1, 2, 3],
                "generator": {
                    "command": ["adapter"],
                    "provider": "fake",
                    "model": "fake-1",
                    "version": "1",
                },
                "output_root": "out",
                "batch_sizes": {"dream": 2, "probe": 2, "battery": 2},
                "runtime": {
                    "model": "fake",
                    "warm_start": "checkpoint",
                    "warm_start_sha256": "a" * 64,
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "learning_rate": 1e-4,
                    "chunk_len": 48,
                    "dream_count": 300,
                    "dream_tokens": 64,
                    "dream_temperature": 1.0,
                    "reply_tokens": 16,
                    "reply_temperature": 0.0,
                    "probe_tokens": 8,
                    "kl_temperature": 1.0,
                    "battery": "battery.json",
                },
                "wakes": [
                    {
                        "scenario": f"wake-{wake}",
                        "facts": [
                            {"entity": f"e{wake}-{fact}", "code": f"{wake}{fact}000"}
                            for fact in range(4)
                        ],
                        "turn_count": 4,
                        "injection_turns": [1, 2, 3, 4],
                        "turn_goals": GOALS,
                    }
                    for wake in range(1, 7)
                ],
            }
        )
    )

    manifest = load_experiment_manifest(path)
    assert manifest.config.wakes == 6 and manifest.generator["model"] == "fake-1"

    value = json.loads(path.read_text())
    value["wakes"][1]["facts"][0]["code"] = " ".join(value["wakes"][0]["facts"][0]["code"])
    path.write_text(json.dumps(value))
    with pytest.raises(AdaptiveWakeError, match="code"):
        load_experiment_manifest(path)


def test_fake_backbone_smoke_runs_all_registered_arms_and_wakes(tmp_path):
    import subprocess

    completed = subprocess.run(
        [sys.executable, "-m", "experiments.adaptive.runner", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    wake_dir = tmp_path / "seed-1" / "wakes"
    assert len([path for path in wake_dir.glob("*_w*.json") if ".partial." not in path.name]) == 19
    shared = json.loads((wake_dir / "shared_w1.json").read_text())
    assert {turn["assistant_stop_reason"] for turn in shared["turns"]} == {"eos"}
    first = [
        json.loads((wake_dir / f"{arm}_w1.json").read_text())
        for arm in ("replay", "nosleep", "sft-ref")
    ]
    assert {item["transcript_sha256"] for item in first} == {shared["transcript_sha256"]}
    assert {item["state_sha256"] for item in first} == {shared["state_sha256"]}
    result = json.loads((tmp_path / "seed-1.json").read_text())
    assert len(result["records"]) == 18
    treatment = {
        (record["arm"], record["wake"]): record["treatment"] for record in result["records"]
    }
    assert treatment["replay", 6]["dreams_generated"] == 300
    cache_metadata = json.loads((tmp_path / "seed-1" / "dreams" / "replay_w6.json").read_text())
    assert len(cache_metadata["bound_rehearsals"]) == 24
    assert len(treatment["replay", 6]["bound_rehearsals"]) == 24
    assert {
        key: treatment["nosleep", 6][key]
        for key in ("dream_passes", "dreams_generated", "raw_transcript_passes")
    } == {
        "dream_passes": 0,
        "dreams_generated": 0,
        "raw_transcript_passes": 0,
    }
    assert treatment["sft-ref", 6]["raw_transcript_passes"] == 1


def test_multisleep_coordinator_sleeps_before_each_arm_specific_later_wake(tmp_path):
    from experiments.adaptive.coordinator import MultiSleepCoordinator

    script = tmp_path / "generator.py"
    script.write_text(
        "import json, sys; r=json.load(sys.stdin); s=r.get('session_id', 's')\n"
        "print(json.dumps({'message': r['goal'], 'session_id': s, "
        "'resume_status': 'resumed' if r.get('session_id') else 'started'}))\n"
    )
    harness = LiveWakeHarness(
        wake_plan(),
        CommandUserGenerator([sys.executable, str(script)]),
        tmp_path,
    )
    wake_states: list[tuple[str, int, int]] = []
    sleeps: list[tuple[str, int, int]] = []

    def wake(arm: str, wake: int, state: int, facts: list[tuple[str, str]], scenario: str):
        artifact = harness.run(
            arm,
            wake,
            facts,
            lambda user, turn: wake_states.append((arm, wake, state)) or user,
            scenario,
            None,
            state_metadata={"state": state},
        )
        return artifact, state + 1

    coordinator = MultiSleepCoordinator(
        ("replay", "nosleep", "sft-ref"), wake, lambda state: state, 0, harness.store
    )
    coordinator.run(
        facts_for_wake=lambda wake: [(f"e{wake}-{i}", f"{wake}{i}000") for i in range(4)],
        scenario_for_wake=lambda wake: f"scenario-{wake}",
        sleep=lambda arm, wake, state, artifact: sleeps.append((arm, wake, state)) or state + 10,
        probe=lambda arm, wake, state, artifact: None,
    )

    assert sleeps[:3] == [("replay", 1, 1), ("nosleep", 1, 1), ("sft-ref", 1, 1)]
    assert {(arm, state) for arm, wake, state in wake_states if wake == 2} == {
        ("replay", 11),
        ("nosleep", 11),
        ("sft-ref", 11),
    }


def test_manifest_runtime_records_every_arm_wake_and_batch_topology(tmp_path):
    from experiments.adaptive.coordinator import ExperimentRuntime
    from experiments.adaptive.manifest import load_experiment_manifest

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "arms": ["replay", "nosleep", "sft-ref"],
                "seeds": [1, 2, 3],
                "generator": {
                    "command": ["fake"],
                    "provider": "fake",
                    "model": "fake",
                    "version": "1",
                },
                "output_root": str(tmp_path / "out"),
                "batch_sizes": {"dream": 2, "probe": 3, "battery": 4},
                "runtime": {
                    "model": "fake",
                    "warm_start": "checkpoint",
                    "warm_start_sha256": "a" * 64,
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "learning_rate": 1e-4,
                    "chunk_len": 48,
                    "dream_count": 300,
                    "dream_tokens": 64,
                    "dream_temperature": 1.0,
                    "reply_tokens": 16,
                    "reply_temperature": 0.0,
                    "probe_tokens": 8,
                    "kl_temperature": 1.0,
                    "battery": "battery.json",
                },
                "wakes": [
                    {
                        "scenario": str(i),
                        "facts": [{"entity": f"e{i}{j}", "code": f"{i}{j}000"} for j in range(4)],
                        "turn_count": 4,
                        "injection_turns": [1, 2, 3, 4],
                        "turn_goals": GOALS,
                    }
                    for i in range(6)
                ],
            }
        )
    )
    manifest = load_experiment_manifest(manifest_path)
    runtime = ExperimentRuntime(
        manifest,
        initial_state=0,
        fork_state=lambda state: state,
        wake=lambda arm, wake, state, facts, scenario: ({"arm": arm, "wake": wake}, state + 1),
        sleep=lambda arm, wake, state, artifact: state + 1,
        probe=lambda arm, wake, state, artifact: {"margin": state},
    )

    result = runtime.run(seed=1)

    assert len(result["records"]) == 18
    assert result["execution"]["batch_sizes"] == {"dream": 2, "probe": 3, "battery": 4}


def test_registered_runtime_configuration_has_no_scientific_defaults(tmp_path):
    from experiments.adaptive.manifest import load_experiment_manifest

    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "arms": ["replay", "nosleep", "sft-ref"],
                "seeds": [1, 2, 3],
                "generator": {
                    "command": ["fake"],
                    "provider": "fake",
                    "model": "fake",
                    "version": "1",
                },
                "output_root": "out",
                "batch_sizes": {"dream": 2, "probe": 2, "battery": 2},
                "runtime": {"model": "fake"},
                "wakes": [
                    {
                        "scenario": str(i),
                        "facts": [{"entity": f"e{i}{j}", "code": f"{i}{j}000"} for j in range(4)],
                        "turn_count": 4,
                        "injection_turns": [1, 2, 3, 4],
                        "turn_goals": GOALS,
                    }
                    for i in range(6)
                ],
            }
        )
    )

    with pytest.raises(AdaptiveWakeError, match="warm_start"):
        load_experiment_manifest(path)


def test_floor_correction_uses_the_matched_nosleep_cell():
    from experiments.adaptive.analysis import floor_correct_records, retention_summary

    records = [
        {
            "seed": 7,
            "arm": arm,
            "wake": wake,
            "facts": {
                "old": {"margin": margin, "fact_wave": 1},
                **({"new": {"margin": margin + 1.0, "fact_wave": 2}} if wake == 2 else {}),
            },
        }
        for wake in (1, 2)
        for arm, margin in (("replay", 4.0), ("nosleep", 1.5), ("sft-ref", 2.0))
    ]

    corrected = floor_correct_records(records)

    replay = next(
        record for record in corrected if record["arm"] == "replay" and record["wake"] == 2
    )
    assert replay["facts"]["old"]["floor_corrected_margin"] == 2.5
    assert replay["facts"]["old"]["installed"] is True
    summary = retention_summary(corrected, wakes=2)["replay"]
    assert summary["floor_corrected_r_matrix"] == [[2.5, None], [2.5, 2.5]]
    assert summary["cumulative_installed"] == [1, 2]


def test_floor_correction_fails_closed_without_each_matched_nosleep_fact():
    from experiments.adaptive.analysis import floor_correct_records

    with pytest.raises(AdaptiveWakeError, match="no-sleep floor"):
        floor_correct_records(
            [
                {"seed": 1, "arm": "nosleep", "wake": 1, "facts": {}},
                {
                    "seed": 1,
                    "arm": "replay",
                    "wake": 1,
                    "facts": {"missing": {"margin": 2.0, "fact_wave": 1}},
                },
            ]
        )


def test_production_entrypoint_runs_backend_and_writes_corrected_seed_output(tmp_path):
    from experiments.adaptive.runner import run_registered_experiment
    from experiments.adaptive.manifest import load_experiment_manifest

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "arms": ["replay", "nosleep", "sft-ref"],
                "seeds": [1, 2, 3],
                "generator": {
                    "command": ["fake"],
                    "provider": "fake",
                    "model": "fake",
                    "version": "1",
                },
                "output_root": str(tmp_path / "out"),
                "batch_sizes": {"dream": 2, "probe": 3, "battery": 4},
                "runtime": {
                    "model": "fake",
                    "warm_start": "checkpoint",
                    "warm_start_sha256": "a" * 64,
                    "lora_rank": 16,
                    "lora_alpha": 32,
                    "learning_rate": 1e-4,
                    "chunk_len": 48,
                    "dream_count": 300,
                    "dream_tokens": 64,
                    "dream_temperature": 1.0,
                    "reply_tokens": 16,
                    "reply_temperature": 0.0,
                    "probe_tokens": 8,
                    "kl_temperature": 1.0,
                    "battery": "battery.json",
                },
                "wakes": [
                    {
                        "scenario": str(i),
                        "facts": [{"entity": f"e{i}{j}", "code": f"{i}{j}000"} for j in range(4)],
                        "turn_count": 4,
                        "injection_turns": [1, 2, 3, 4],
                        "turn_goals": GOALS,
                    }
                    for i in range(6)
                ],
            }
        )
    )
    manifest = load_experiment_manifest(manifest_path)

    class Backend:
        initial_state = 0

        @staticmethod
        def fork_state(state):
            return state

        @staticmethod
        def wake(arm, wake, state, facts, scenario):
            return {
                "arm": arm,
                "wake": wake,
                "artifact_sha256": f"{arm}-{wake}",
                "transcript_token_sha256": "shared",
                "state_sha256": "shared",
            }, state + 1

        @staticmethod
        def sleep(arm, wake, state, artifact):
            return state + (0 if arm == "nosleep" else 1)

        @staticmethod
        def probe(arm, wake, state, artifact):
            margin = 1.0 if arm == "nosleep" else 3.0
            return {
                "facts": {
                    f"e{fact_wake}{index}": {"margin": margin}
                    for fact_wake in range(wake)
                    for index in range(4)
                }
            }

        @staticmethod
        def execution_metadata():
            return {"peak_vram_bytes": 0, "retries": 0}

    result = run_registered_experiment(manifest, 1, Backend())

    assert len(result["records"]) == 18
    replay = next(
        record for record in result["records"] if record["arm"] == "replay" and record["wake"] == 1
    )
    assert replay["facts"]["e00"]["floor_corrected_margin"] == 2.0
    assert (tmp_path / "out" / "seed-1.json").exists()


@pytest.mark.parametrize("failure", ["wake", "sleep"])
def test_interrupted_seed_restarts_from_base_and_preserves_numbered_logs(tmp_path, failure):
    from experiments.adaptive.runner import run_registered_experiment
    from experiments.adaptive.manifest import ExperimentConfig, ExperimentManifest, WakeSpec

    config = ExperimentConfig.from_dict(
        {"arms": ["replay", "nosleep", "sft-ref"], "wakes": 6, "seeds": [1, 2, 3]}
    )
    plan = wake_plan()
    wakes = tuple(
        WakeSpec(
            f"scenario-{wave}",
            tuple((f"e{wave}{index}", f"{wave}{index}000") for index in range(4)),
            plan,
        )
        for wave in range(6)
    )
    manifest = ExperimentManifest(
        config,
        wakes,
        {"command": ["fake"], "provider": "fake", "model": "fake", "version": "1"},
        str(tmp_path),
        {"dream": 1, "probe": 1, "battery": 1},
        {},
    )

    class Backend:
        initial_state = 0

        def __init__(self, fail):
            self.fail, self.history = fail, []

        @staticmethod
        def fork_state(state):
            return state

        def wake(self, arm, wave, state, facts, scenario):
            self.history.append(("wake", arm, wave, state))
            if self.fail == "wake" and arm == "replay" and wave == 3:
                raise RuntimeError("provider interrupted")
            return {
                "arm": arm,
                "wake": wave,
                "artifact_sha256": f"{arm}-{wave}",
                "transcript_token_sha256": "shared" if wave == 1 else f"{arm}-{wave}",
                "state_sha256": "shared" if wave == 1 else f"state-{arm}-{wave}",
            }, state + 1

        def sleep(self, arm, wave, state, artifact):
            self.history.append(("sleep", arm, wave, state))
            if self.fail == "sleep" and arm == "replay" and wave == 2:
                raise RuntimeError("sleep interrupted")
            return state + 10

        def probe(self, arm, wave, state, artifact):
            margin = 1.0 if arm == "nosleep" else 3.0
            return {
                "facts": {
                    entity: {"margin": margin, "fact_wave": learned}
                    for learned, spec in enumerate(wakes[:wave], 1)
                    for entity, _ in spec.facts
                }
            }

        @staticmethod
        def execution_metadata():
            return {}

    with pytest.raises(RuntimeError, match="interrupted"):
        run_registered_experiment(manifest, 1, Backend(failure))
    original = (tmp_path / "seed-1.jsonl").read_text()
    restarted = Backend(None)

    result = run_registered_experiment(manifest, 1, restarted)

    assert len(result["records"]) == 18
    assert restarted.history[0] == ("wake", "shared", 1, 0)
    assert (tmp_path / "seed-1.jsonl").read_text() == original
    assert (tmp_path / "seed-1.resume-1.jsonl").exists()
    assert (tmp_path / "seed-1.json").exists()


def test_seed_aggregate_keeps_per_seed_values_and_reports_uncertainty():
    from experiments.adaptive.analysis import aggregate_seed_results

    results = [
        {
            "seed": seed,
            "retention": {
                "replay": {
                    "floor_corrected_r_matrix": [[value]],
                    "cumulative_installed": [installed],
                    "backward_transfer": value,
                }
            },
        }
        for seed, value, installed in ((1, 1.0, 1), (2, 3.0, 3), (3, 5.0, 5))
    ]

    aggregate = aggregate_seed_results(results, wakes=1)

    assert aggregate["seeds"] == [1, 2, 3]
    assert aggregate["arms"]["replay"]["floor_corrected_r_matrix"][0][0] == {
        "mean": 3.0,
        "stderr": pytest.approx(1.154700538),
        "values": [1.0, 3.0, 5.0],
    }


def test_seed_aggregate_reports_all_registered_metric_families():
    from experiments.adaptive.analysis import aggregate_seed_results

    results = [
        {
            "seed": seed,
            "retention": {
                "replay": {
                    "floor_corrected_r_matrix": [[margin]],
                    "cumulative_installed": [1],
                    "backward_transfer": None,
                }
            },
            "records": [
                {
                    "arm": "replay",
                    "wake": 1,
                    "facts": {
                        "f": {
                            "floor_corrected_margin": margin,
                            "exact_match": True,
                            "paraphrase_rate": 0.5,
                        }
                    },
                    "battery": {
                        "lost": seed,
                        "loss_rate": seed / 10,
                        "mean_logprob_delta": -seed,
                        "median_logprob_delta": -seed,
                        "p10_logprob_delta": -seed,
                    },
                    "ppl": 10 + seed,
                    "ppl_delta": seed / 2,
                }
            ],
        }
        for seed, margin in ((1, 1.0), (2, 2.0), (3, 3.0))
    ]

    metrics = aggregate_seed_results(results, 1)["arms"]["replay"]["metrics"]

    assert metrics["corrected_margin"][0]["values"] == [1.0, 2.0, 3.0]
    assert metrics["exact_match"][0]["mean"] == 1.0
    assert metrics["battery_loss_rate"][0]["values"] == [0.1, 0.2, 0.3]
    assert metrics["ppl_delta"][0]["values"] == [0.5, 1.0, 1.5]
    assert "ppl" not in metrics


def test_rehearsal_retention_joins_earlier_fact_at_later_sleep():
    from experiments.adaptive.analysis import aggregate_seed_results

    results = [
        {
            "seed": seed,
            "records": [
                {
                    "arm": "replay",
                    "wake": 1,
                    "facts": {"old": {"fact_wave": 1, "floor_corrected_margin": 1.0}},
                    "treatment": {"bound_rehearsals": {"old": 0}},
                },
                {
                    "arm": "replay",
                    "wake": 2,
                    "facts": {
                        "old": {"fact_wave": 1, "floor_corrected_margin": margin},
                        "new": {"fact_wave": 2, "floor_corrected_margin": 2.0},
                    },
                    "treatment": {"bound_rehearsals": {"old": count, "new": 1}},
                },
            ],
        }
        for seed, count, margin in ((1, 2, 3.0), (2, 2, 5.0), (3, 0, 1.0))
    ]

    analysis = aggregate_seed_results(results, 2)["rehearsal_retention"]

    assert analysis["observations"][0] == {
        "seed": 1,
        "sleep_wake": 2,
        "fact": "old",
        "learning_wake": 1,
        "rehearsals": 2,
        "floor_corrected_margin": 3.0,
    }
    assert analysis["by_rehearsal_count"]["2"] == {
        "mean": 4.0,
        "stderr": pytest.approx(1.0),
        "seed_values": {"1": 3.0, "2": 5.0},
        "values": [3.0, 5.0],
    }


def test_rehearsal_retention_uses_one_mean_per_seed_and_count():
    from experiments.adaptive.analysis import rehearsal_retention_analysis

    results = [
        {
            "seed": 1,
            "records": [
                {
                    "arm": "replay",
                    "wake": 2,
                    "facts": {
                        "a": {"fact_wave": 1, "floor_corrected_margin": 2.0},
                        "b": {"fact_wave": 1, "floor_corrected_margin": 6.0},
                    },
                    "treatment": {"bound_rehearsals": {"a": 3, "b": 3}},
                }
            ],
        },
        {
            "seed": 2,
            "records": [
                {
                    "arm": "replay",
                    "wake": 2,
                    "facts": {"c": {"fact_wave": 1, "floor_corrected_margin": 8.0}},
                    "treatment": {"bound_rehearsals": {"c": 3}},
                }
            ],
        },
    ]

    grouped = rehearsal_retention_analysis(results)["by_rehearsal_count"]["3"]

    assert grouped["seed_values"] == {"1": 4.0, "2": 8.0}
    assert grouped["values"] == [4.0, 8.0]
    assert grouped["mean"] == 6.0


def test_dream_cache_identity_binds_current_teacher_tokens_state_facts_and_settings():
    from experiments.adaptive.backend import dream_cache_identity

    base = dream_cache_identity(
        "teacher-1",
        "tokens",
        "state",
        [("e", "12345")],
        {"dream_count": 300, "dream_tokens": 64},
        7,
        2,
    )
    changed = dream_cache_identity(
        "teacher-2",
        "tokens",
        "state",
        [("e", "12345")],
        {"dream_count": 300, "dream_tokens": 64},
        7,
        2,
    )

    assert base["teacher_sha256"] == "teacher-1"
    assert base["identity_sha256"] != changed["identity_sha256"]


def test_battery_collision_terms_cover_registered_wake_inputs(tmp_path):
    from experiments.adaptive.backend import battery_collision_terms
    from experiments.adaptive.manifest import load_experiment_manifest

    value = {
        "arms": ["replay", "nosleep", "sft-ref"],
        "seeds": [1, 2, 3],
        "generator": {"command": ["fake"], "provider": "fake", "model": "fake", "version": "1"},
        "output_root": "out",
        "batch_sizes": {"dream": 1, "probe": 1, "battery": 1},
        "runtime": {
            "model": "fake",
            "warm_start": "x",
            "warm_start_sha256": "a" * 64,
            "lora_rank": 1,
            "lora_alpha": 1,
            "learning_rate": 0.1,
            "chunk_len": 1,
            "dream_count": 300,
            "dream_tokens": 1,
            "dream_temperature": 1,
            "reply_tokens": 1,
            "reply_temperature": 0,
            "probe_tokens": 1,
            "kl_temperature": 1,
            "battery": "b",
        },
        "wakes": [
            {
                "scenario": f"scenario {wake}",
                "facts": [
                    {"entity": f"entity{wake}{fact}", "code": f"{wake}{fact}000"}
                    for fact in range(4)
                ],
                "turn_count": 4,
                "injection_turns": [1, 2, 3, 4],
                "turn_goals": GOALS,
            }
            for wake in range(6)
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))

    terms = battery_collision_terms(load_experiment_manifest(path), "[U]", "[A]")

    assert "scenario 0" in terms and GOALS[0] in terms
    assert "entity00" in terms and "00000" in terms
    assert "[U] What is the code for the entity00?[A] The code for the entity00 is" in terms


def test_teacher_hash_tracks_only_current_trainable_weights():
    import torch
    from experiments.adaptive.backend import DreamSleepBackend

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    model[1].weight.requires_grad_(False)
    model[1].bias.requires_grad_(False)
    backend = DreamSleepBackend.__new__(DreamSleepBackend)
    backend.torch, backend.initial_full_sha = torch, "anchor"

    first = backend._teacher_hash(model)
    assert backend._teacher_hash(model) == first
    with torch.no_grad():
        model[0].weight.add_(1)
    assert backend._teacher_hash(model) != first
    stable = backend._teacher_hash(model)
    with torch.no_grad():
        model[1].weight.add_(1)
    assert backend._teacher_hash(model) == stable


def test_later_dream_rehearsal_counts_include_earlier_facts():
    from types import SimpleNamespace
    from experiments.adaptive.backend import bound_rehearsal_counts
    from experiments.facts import Fact

    facts = [Fact("older", "entity", "1 2 3 4 5"), Fact("newer", "entity", "5 4 3 2 1")]
    dreams = [
        SimpleNamespace(token_texts=["The older has code 1 2 3 4 5."]),
        SimpleNamespace(token_texts=["The newer has code 5 4 3 2 1."]),
    ]

    assert bound_rehearsal_counts(dreams, facts) == {"older": 1, "newer": 1}
