"""Live adaptive wakes with an explicit, frozen turn plan."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


class AdaptiveWakeError(RuntimeError):
    pass


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class WakePlan:
    turn_count: int
    injection_turns: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "WakePlan":
        if "turn_count" not in value:
            raise AdaptiveWakeError("wake plan requires turn_count")
        if "injection_turns" not in value:
            raise AdaptiveWakeError("wake plan requires injection_turns")
        turn_count, turns = value["turn_count"], value["injection_turns"]
        if not isinstance(turn_count, int) or turn_count < 4:
            raise AdaptiveWakeError("turn_count must be an integer of at least four")
        if not isinstance(turns, list) or len(turns) != 4:
            raise AdaptiveWakeError("wake plan requires four injection turns")
        if (any(not isinstance(turn, int) for turn in turns) or sorted(turns) != turns
                or len(set(turns)) != len(turns) or turns[0] < 1 or turns[-1] > turn_count):
            raise AdaptiveWakeError("injection turns must be distinct, sorted, and inside turn_count")
        return cls(turn_count, tuple(turns))


@dataclass(frozen=True)
class ExperimentConfig:
    arms: tuple[str, ...]
    wakes: int
    seeds: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ExperimentConfig":
        arms, wakes, seeds = value.get("arms"), value.get("wakes"), value.get("seeds")
        if arms != ["replay", "nosleep", "sft-ref"]:
            raise AdaptiveWakeError("registered arms are replay, nosleep, and sft-ref")
        if wakes != 6:
            raise AdaptiveWakeError("registered experiment requires six wakes")
        if not isinstance(seeds, list) or len(seeds) != 3 or any(not isinstance(seed, int) for seed in seeds):
            raise AdaptiveWakeError("registered experiment requires three integer seeds")
        return cls(tuple(arms), wakes, tuple(seeds))


@dataclass(frozen=True)
class WakeSpec:
    scenario: str
    facts: tuple[tuple[str, str], ...]
    plan: WakePlan


@dataclass(frozen=True)
class ExperimentManifest:
    config: ExperimentConfig
    wakes: tuple[WakeSpec, ...]
    generator: dict[str, object]
    output_root: str
    batch_sizes: dict[str, int]


def load_wake_plan(path: str | Path) -> WakePlan:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveWakeError(f"cannot read wake plan {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdaptiveWakeError("wake plan must be a JSON object")
    return WakePlan.from_dict(value)


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveWakeError(f"cannot read experiment manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdaptiveWakeError("experiment manifest must be a JSON object")
    config = ExperimentConfig.from_dict({**value, "wakes": 6})
    wake_values = value.get("wakes")
    if not isinstance(wake_values, list) or len(wake_values) != config.wakes:
        raise AdaptiveWakeError("experiment manifest requires six wakes")
    wakes: list[WakeSpec] = []
    for index, wake in enumerate(wake_values, start=1):
        if not isinstance(wake, dict):
            raise AdaptiveWakeError(f"wake {index} must be a JSON object")
        scenario, facts = wake.get("scenario"), wake.get("facts")
        if not isinstance(scenario, str) or not scenario:
            raise AdaptiveWakeError(f"wake {index} requires a scenario")
        if not isinstance(facts, list) or len(facts) != 4:
            raise AdaptiveWakeError(f"wake {index} requires four facts")
        parsed: list[tuple[str, str]] = []
        for fact in facts:
            if not isinstance(fact, dict) or not isinstance(fact.get("entity"), str) or not isinstance(fact.get("code"), str):
                raise AdaptiveWakeError(f"wake {index} facts require entity and code strings")
            parsed.append((fact["entity"], fact["code"]))
        wakes.append(WakeSpec(scenario, tuple(parsed), WakePlan.from_dict(wake)))
    generator = value.get("generator")
    if not isinstance(generator, dict):
        raise AdaptiveWakeError("experiment manifest requires generator")
    command = generator.get("command")
    required = ("provider", "model", "version")
    if (not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command)
            or any(not isinstance(generator.get(key), str) or not generator[key] for key in required)):
        raise AdaptiveWakeError("generator requires command, provider, model, and version")
    output_root = value.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise AdaptiveWakeError("experiment manifest requires output_root")
    batch_sizes = value.get("batch_sizes")
    required_batches = ("dream", "probe", "battery")
    if (not isinstance(batch_sizes, dict) or any(not isinstance(batch_sizes.get(key), int) or batch_sizes[key] < 1
                                                 for key in required_batches)):
        raise AdaptiveWakeError("batch_sizes requires positive dream, probe, and battery values")
    return ExperimentManifest(config, tuple(wakes), dict(generator), output_root, dict(batch_sizes))


class CommandUserGenerator:
    """Provider-neutral JSON command adapter; a resume never starts afresh."""

    def __init__(self, command: Sequence[str], timeout: float = 120.0,
                 provenance: dict[str, str] | None = None):
        if not command:
            raise AdaptiveWakeError("user-generator command cannot be empty")
        self.command = tuple(command)
        self.timeout = timeout
        self.provenance = dict(provenance or {})

    def next_user(self, request: dict[str, object]) -> dict[str, object]:
        try:
            completed = subprocess.run(self.command, input=_canonical(request), text=True,
                                       capture_output=True, timeout=self.timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AdaptiveWakeError(f"user-generator command failed: {exc}") from exc
        if completed.returncode:
            context = "resume failed" if request.get("session_id") is not None else "start failed"
            raise AdaptiveWakeError(f"user-generator {context} with exit {completed.returncode}")
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise AdaptiveWakeError("user-generator command returned invalid JSON") from exc
        if not isinstance(result, dict) or not isinstance(result.get("message"), str):
            raise AdaptiveWakeError("user-generator result requires a string message")
        prior = request.get("session_id")
        if prior is not None and (result.get("resume_status") != "resumed" or result.get("session_id") != prior):
            raise AdaptiveWakeError("user-generator resume failed; refusing to start a new session")
        if prior is None and not isinstance(result.get("session_id"), str):
            raise AdaptiveWakeError("user-generator start returned no session_id")
        return result


class LiveWakeHarness:
    """Runs one arm's wake in order and stores one immutable realized artifact."""

    def __init__(self, plan: WakePlan, generator: CommandUserGenerator, artifact_dir: str | Path):
        self.plan, self.generator, self.artifact_dir = plan, generator, Path(artifact_dir)

    def run(self, arm: str, wake: int, facts: Sequence[tuple[str, str]],
            local_reply: Callable[[str], str], scenario: str, session_id: str | None,
            *, state_metadata: dict[str, object] | Callable[[], dict[str, object]] | None = None) -> dict[str, object]:
        if len(facts) != 4:
            raise AdaptiveWakeError("each wake requires exactly four facts")
        session, latest_reply = session_id, ""
        fact_by_turn = dict(zip(self.plan.injection_turns, facts, strict=True))
        turns: list[dict[str, object]] = []
        for turn in range(1, self.plan.turn_count + 1):
            fact = fact_by_turn.get(turn)
            goal = (f"State that {fact[0]} has code {fact[1]}." if fact else
                    "Continue the scenario without introducing a target fact.")
            request = {"latest_assistant_reply": latest_reply, "goal": goal,
                       "scenario": scenario, "turn": turn}
            if session is not None:
                request["session_id"] = session
            result = self.generator.next_user(request)
            user, session = str(result["message"]), str(result["session_id"])
            latest_reply = local_reply(user)
            turns.append({"turn": turn, "goal": goal, "user": user, "assistant": latest_reply,
                          "request_sha256": _sha(request), "response_sha256": _sha(result),
                          "resume_status": result.get("resume_status"),
                          "token_usage": result.get("token_usage")})
        metadata = state_metadata() if callable(state_metadata) else state_metadata
        metadata = dict(metadata or {})
        artifact: dict[str, object] = {
            "version": 1, "arm": arm, "wake": wake, "scenario": scenario,
            "plan": {"turn_count": self.plan.turn_count, "injection_turns": list(self.plan.injection_turns)},
            "facts": [{"entity": entity, "code": code} for entity, code in facts],
            "generator": {"command": list(self.generator.command), "session_id": session,
                          **self.generator.provenance},
            "turns": turns, "state_metadata": metadata, "state_sha256": _sha(metadata),
            "transcript_sha256": _sha(turns),
        }
        artifact["artifact_sha256"] = _sha(artifact)
        self.store(artifact)
        return artifact

    def store(self, artifact: dict[str, object]) -> None:
        path = self.artifact_dir / f"{artifact['arm']}_w{artifact['wake']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = _canonical(artifact) + "\n"
        if path.exists() and path.read_text() != serialized:
            raise AdaptiveWakeError(f"wake artifact already exists with different content: {path}")
        if not path.exists():
            path.write_text(serialized)


class MultiSleepCoordinator:
    """Own three arm states and preserve wake/sleep ordering."""

    def __init__(
        self,
        arms: Sequence[str],
        wake: Callable[[str, int, object, Sequence[tuple[str, str]], str], tuple[dict[str, object], object]],
        fork_state: Callable[[object], object],
        initial_state: object,
        store_artifact: Callable[[dict[str, object]], None],
    ):
        if tuple(arms) != ("replay", "nosleep", "sft-ref"):
            raise AdaptiveWakeError("registered arms are replay, nosleep, and sft-ref")
        self.arms, self.wake, self.fork_state, self.initial_state = tuple(arms), wake, fork_state, initial_state
        self.store_artifact = store_artifact

    def run(
        self,
        *,
        facts_for_wake: Callable[[int], Sequence[tuple[str, str]]],
        scenario_for_wake: Callable[[int], str],
        sleep: Callable[[str, int, object, dict[str, object]], object],
        probe: Callable[[str, int, object, dict[str, object]], None],
        wakes: int = 6,
    ) -> dict[tuple[str, int], dict[str, object]]:
        shared, shared_state = self.wake("shared", 1, self.initial_state, facts_for_wake(1), scenario_for_wake(1))
        artifacts: dict[tuple[str, int], dict[str, object]] = {}
        states = {arm: self.fork_state(shared_state) for arm in self.arms}
        for arm in self.arms:
            artifact = dict(shared)
            artifact["arm"] = arm
            artifact["artifact_sha256"] = _sha({key: value for key, value in artifact.items() if key != "artifact_sha256"})
            self.store_artifact(artifact)
            artifacts[arm, 1] = artifact
        for wake in range(1, wakes + 1):
            for arm in self.arms:
                if wake > 1:
                    artifacts[arm, wake], states[arm] = self.wake(
                        arm, wake, states[arm], facts_for_wake(wake), scenario_for_wake(wake)
                    )
                states[arm] = sleep(arm, wake, states[arm], artifacts[arm, wake])
                probe(arm, wake, states[arm], artifacts[arm, wake])
        return artifacts


class ExperimentRuntime:
    """Manifest-driven production boundary around the ordered coordinator."""

    def __init__(self, manifest: ExperimentManifest, *, initial_state: object,
                 fork_state: Callable[[object], object],
                 wake: Callable[[str, int, object, Sequence[tuple[str, str]], str], tuple[dict[str, object], object]],
                 sleep: Callable[[str, int, object, dict[str, object]], object],
                 probe: Callable[[str, int, object, dict[str, object]], dict[str, object]]):
        self.manifest = manifest
        self.initial_state, self.fork_state = initial_state, fork_state
        self.wake, self.sleep, self.probe = wake, sleep, probe

    def run(self, *, seed: int) -> dict[str, object]:
        if seed not in self.manifest.config.seeds:
            raise AdaptiveWakeError(f"seed {seed} is not in the manifest")
        records: list[dict[str, object]] = []

        def probe(arm: str, wake: int, state: object, artifact: dict[str, object]) -> None:
            records.append({"seed": seed, "arm": arm, "wake": wake, "artifact_sha256": artifact.get("artifact_sha256"),
                            **self.probe(arm, wake, state, artifact)})

        coordinator = MultiSleepCoordinator(self.manifest.config.arms, self.wake, self.fork_state,
                                             self.initial_state, lambda _: None)
        coordinator.run(
            facts_for_wake=lambda wake: self.manifest.wakes[wake - 1].facts,
            scenario_for_wake=lambda wake: self.manifest.wakes[wake - 1].scenario,
            sleep=self.sleep, probe=probe,
        )
        return {"records": records, "execution": {"batch_sizes": self.manifest.batch_sizes,
                "peak_vram_bytes": None, "throughput": None, "retries": 0}}
