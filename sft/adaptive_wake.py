"""Live adaptive wakes with an explicit, frozen turn plan."""

from __future__ import annotations

import hashlib
import json
import re
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


class AdaptiveWakeError(RuntimeError):
    pass


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def token_sha(ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(int(token)) for token in ids).encode()).hexdigest()


@dataclass(frozen=True)
class WakePlan:
    turn_count: int
    injection_turns: tuple[int, ...]
    turn_goals: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "WakePlan":
        if "turn_count" not in value:
            raise AdaptiveWakeError("wake plan requires turn_count")
        if "injection_turns" not in value:
            raise AdaptiveWakeError("wake plan requires injection_turns")
        turn_count, turns, goals = value["turn_count"], value["injection_turns"], value.get("turn_goals")
        if not isinstance(turn_count, int) or turn_count < 4:
            raise AdaptiveWakeError("turn_count must be an integer of at least four")
        if not isinstance(turns, list) or len(turns) != 4:
            raise AdaptiveWakeError("wake plan requires four injection turns")
        if (any(not isinstance(turn, int) for turn in turns) or sorted(turns) != turns
                or len(set(turns)) != len(turns) or turns[0] < 1 or turns[-1] > turn_count):
            raise AdaptiveWakeError("injection turns must be distinct, sorted, and inside turn_count")
        if (not isinstance(goals, list) or len(goals) != turn_count
                or any(not isinstance(goal, str) or not goal.strip() for goal in goals)):
            raise AdaptiveWakeError("turn_goals must contain one non-empty goal per turn")
        return cls(turn_count, tuple(turns), tuple(goals))


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
        if len(set(seeds)) != len(seeds):
            raise AdaptiveWakeError("registered experiment seeds must be distinct")
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
    runtime: dict[str, object]


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
    entities: set[str] = set()
    codes: set[str] = set()
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
            if fact["entity"] in entities:
                raise AdaptiveWakeError(f"wake {index} repeats entity {fact['entity']}")
            canonical_code = fact["code"].replace(" ", "")
            if canonical_code in codes:
                raise AdaptiveWakeError(f"wake {index} repeats fact code {fact['code']}")
            if re.fullmatch(r"\d(?: ?\d){4}", fact["code"]) is None:
                raise AdaptiveWakeError(f"wake {index} fact codes must contain five digits")
            entities.add(fact["entity"])
            codes.add(canonical_code)
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
    runtime = value.get("runtime")
    required_runtime = {
        "model": str, "warm_start": str, "warm_start_sha256": str,
        "lora_rank": int, "lora_alpha": (int, float), "learning_rate": (int, float),
        "chunk_len": int, "dream_count": int, "dream_tokens": int,
        "dream_temperature": (int, float), "reply_tokens": int,
        "reply_temperature": (int, float), "probe_tokens": int,
        "kl_temperature": (int, float), "battery": str,
    }
    if not isinstance(runtime, dict):
        raise AdaptiveWakeError("experiment manifest requires runtime")
    for key, kind in required_runtime.items():
        if not isinstance(runtime.get(key), kind):
            raise AdaptiveWakeError(f"runtime requires {key}")
    if runtime["dream_count"] != 300:
        raise AdaptiveWakeError("runtime dream_count must be the registered 300")
    for key in ("lora_rank", "chunk_len", "dream_tokens", "reply_tokens", "probe_tokens"):
        if int(runtime[key]) < 1:
            raise AdaptiveWakeError(f"runtime {key} must be positive")
    if len(str(runtime["warm_start_sha256"])) != 64:
        raise AdaptiveWakeError("runtime warm_start_sha256 must be a full SHA-256")
    for key in ("lora_alpha", "learning_rate", "dream_temperature", "kl_temperature"):
        if float(runtime[key]) <= 0:
            raise AdaptiveWakeError(f"runtime {key} must be positive")
    if float(runtime["reply_temperature"]) < 0:
        raise AdaptiveWakeError("runtime reply_temperature cannot be negative")
    return ExperimentManifest(config, tuple(wakes), dict(generator), output_root, dict(batch_sizes), dict(runtime))


def counterbalanced_facts(facts: Sequence[tuple[str, str]], seed_index: int,
                          wake: int) -> list[tuple[str, str]]:
    ordered = list(facts)
    offset = (seed_index + wake - 1) % len(ordered)
    return ordered[offset:] + ordered[:offset]


def floor_correct_records(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Subtract each seed/wake/fact's matched no-sleep margin."""
    floors: dict[tuple[object, object, str], float] = {}
    for record in records:
        if record.get("arm") != "nosleep" or not isinstance(record.get("facts"), dict):
            continue
        for fact, stats in record["facts"].items():
            if isinstance(fact, str) and isinstance(stats, dict) and isinstance(stats.get("margin"), (int, float)):
                floors[record.get("seed"), record.get("wake"), fact] = float(stats["margin"])

    corrected: list[dict[str, object]] = []
    for record in records:
        result = dict(record)
        facts = record.get("facts")
        if isinstance(facts, dict):
            fact_results: dict[str, object] = {}
            for fact, stats in facts.items():
                if not isinstance(fact, str) or not isinstance(stats, dict):
                    continue
                values = dict(stats)
                margin, floor = values.get("margin"), floors.get((record.get("seed"), record.get("wake"), fact))
                if (record.get("arm") != "nosleep" and isinstance(margin, (int, float))
                        and floor is None):
                    raise AdaptiveWakeError(
                        f"missing matched no-sleep floor for seed {record.get('seed')} "
                        f"wake {record.get('wake')} fact {fact}"
                    )
                if isinstance(margin, (int, float)) and floor is not None:
                    delta = float(margin) - floor
                    values["nosleep_margin"] = floor
                    values["floor_corrected_margin"] = delta
                    values["installed"] = delta >= 1.0
                fact_results[fact] = values
            result["facts"] = fact_results
        corrected.append(result)
    return corrected


def retention_summary(records: Sequence[dict[str, object]], wakes: int) -> dict[str, dict[str, object]]:
    """Build per-arm raw and no-sleep-corrected retention matrices."""
    summaries: dict[str, dict[str, object]] = {}
    for arm in ("replay", "nosleep", "sft-ref"):
        raw: list[list[float | None]] = [[None] * wakes for _ in range(wakes)]
        corrected: list[list[float | None]] = [[None] * wakes for _ in range(wakes)]
        installed = [0] * wakes
        own: dict[int, float] = {}
        final: dict[int, float] = {}
        for record in records:
            if record.get("arm") != arm or not isinstance(record.get("wake"), int):
                continue
            evaluation = int(record["wake"])
            facts = record.get("facts")
            if not isinstance(facts, dict):
                continue
            by_learning: dict[int, list[dict[str, object]]] = {}
            for stats in facts.values():
                if isinstance(stats, dict) and isinstance(stats.get("fact_wave"), int):
                    by_learning.setdefault(int(stats["fact_wave"]), []).append(stats)
                    installed[evaluation - 1] += bool(stats.get("installed"))
            for learning, values in by_learning.items():
                margins = [float(value["margin"]) for value in values if isinstance(value.get("margin"), (int, float))]
                deltas = [float(value["floor_corrected_margin"]) for value in values
                          if isinstance(value.get("floor_corrected_margin"), (int, float))]
                if margins:
                    raw[evaluation - 1][learning - 1] = sum(margins) / len(margins)
                if deltas:
                    mean = sum(deltas) / len(deltas)
                    corrected[evaluation - 1][learning - 1] = mean
                    if evaluation == learning:
                        own[learning] = mean
                    if evaluation == wakes:
                        final[learning] = mean
        changes = [final[learning] - own[learning] for learning in own.keys() & final.keys() if learning < wakes]
        summaries[arm] = {
            "r_matrix": raw, "floor_corrected_r_matrix": corrected,
            "cumulative_installed": installed,
            "backward_transfer": sum(changes) / len(changes) if changes else None,
        }
    return summaries


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
        for key, expected in self.provenance.items():
            if result.get(key) is not None and result[key] != expected:
                raise AdaptiveWakeError(f"user-generator {key} provenance changed during the run")
        return result


class LiveWakeHarness:
    """Runs one arm's wake in order and stores one immutable realized artifact."""

    def __init__(self, plan: WakePlan, generator: CommandUserGenerator, artifact_dir: str | Path):
        self.plan, self.generator, self.artifact_dir = plan, generator, Path(artifact_dir)

    def run(self, arm: str, wake: int, facts: Sequence[tuple[str, str]],
            local_reply: Callable[[str, int], str], scenario: str, session_id: str | None,
            *, state_metadata: dict[str, object] | Callable[[], dict[str, object]] | None = None,
            fact_distances: Callable[[Sequence[dict[str, object]]], dict[str, int]] | None = None,
            replay_turn: Callable[[dict[str, object]], None] | None = None,
            ) -> dict[str, object]:
        if len(facts) != 4:
            raise AdaptiveWakeError("each wake requires exactly four facts")
        final_path = self.artifact_dir / f"{arm}_w{wake}.json"
        partial_path = self.artifact_dir / f"{arm}_w{wake}.partial.json"
        if final_path.exists():
            artifact = json.loads(final_path.read_text())
            expected_plan = {"turn_count": self.plan.turn_count,
                             "injection_turns": list(self.plan.injection_turns),
                             "turn_goals": list(self.plan.turn_goals)}
            if (artifact.get("scenario") != scenario or artifact.get("plan") != expected_plan
                    or artifact.get("facts") != [{"entity": entity, "code": code} for entity, code in facts]):
                raise AdaptiveWakeError(f"completed wake does not match requested inputs: {final_path}")
            if replay_turn:
                for stored in artifact["turns"]:
                    replay_turn(stored)
            return artifact
        session, latest_reply = session_id, ""
        fact_by_turn = dict(zip(self.plan.injection_turns, facts, strict=True))
        turns: list[dict[str, object]] = []
        pending: dict[str, object] | None = None
        if partial_path.exists():
            partial = json.loads(partial_path.read_text())
            if (partial.get("scenario") != scenario or partial.get("facts") != [list(fact) for fact in facts]
                    or partial.get("turn_goals") != list(self.plan.turn_goals)
                    or partial.get("injection_turns") != list(self.plan.injection_turns)):
                raise AdaptiveWakeError(f"incomplete wake does not match requested inputs: {partial_path}")
            turns = list(partial.get("turns", []))
            pending = partial.get("pending") if isinstance(partial.get("pending"), dict) else None
            session = partial.get("session_id") if isinstance(partial.get("session_id"), str) else session
            latest_reply = str(turns[-1]["assistant"]) if turns else ""
            if replay_turn:
                for stored in turns:
                    replay_turn(stored)

        def persist() -> None:
            value = {"version": 1, "arm": arm, "wake": wake, "scenario": scenario,
                     "facts": [list(fact) for fact in facts], "turn_goals": list(self.plan.turn_goals),
                     "injection_turns": list(self.plan.injection_turns),
                     "session_id": session, "turns": turns, "pending": pending}
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = partial_path.with_suffix(".tmp")
            temporary.write_text(_canonical(value) + "\n")
            os.replace(temporary, partial_path)

        for turn in range(len(turns) + 1, self.plan.turn_count + 1):
            fact = fact_by_turn.get(turn)
            intent = self.plan.turn_goals[turn - 1]
            goal = (f"{intent}\nCommunicate that {fact[0]} has code {fact[1]} exactly once."
                    if fact else intent)
            request = {"latest_assistant_reply": latest_reply, "goal": goal,
                       "scenario": scenario, "turn": turn}
            if session is not None:
                request["session_id"] = session
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] {arm} wake {wake} turn {turn}/{self.plan.turn_count}: requesting user", flush=True)
            result = pending["result"] if pending and pending.get("turn") == turn else self.generator.next_user(request)
            user, session = str(result["message"]), str(result["session_id"])
            if fact and (fact[0].casefold() not in user.casefold() or user.count(fact[1]) != 1):
                raise AdaptiveWakeError(
                    f"injection turn {turn} must communicate entity {fact[0]!r} and code {fact[1]!r} once"
                )
            if fact and len(re.findall(r"[A-Za-z]+", user)) < 3:
                raise AdaptiveWakeError(f"injection turn {turn} must use a natural-language user message")
            pending = {"turn": turn, "request": request, "result": result}
            persist()
            reply = local_reply(user, turn)
            token_fields: dict[str, object] = {}
            if isinstance(reply, dict):
                latest_reply = reply.get("text")
                prompt_ids, assistant_ids = reply.get("prompt_token_ids"), reply.get("assistant_token_ids")
                if (not isinstance(latest_reply, str) or not isinstance(prompt_ids, list)
                        or not isinstance(assistant_ids, list) or not assistant_ids
                        or any(not isinstance(token, int) for token in prompt_ids + assistant_ids)):
                    raise AdaptiveWakeError("local reply requires text and consumed prompt/assistant token IDs")
                token_fields = {"prompt_token_ids": prompt_ids, "assistant_token_ids": assistant_ids}
                if isinstance(reply.get("stop_reason"), str):
                    token_fields["assistant_stop_reason"] = reply["stop_reason"]
            elif isinstance(reply, str):
                latest_reply = reply
            else:
                raise AdaptiveWakeError("local reply must be text or a tokenized reply object")
            print(f"[{time.strftime('%H:%M:%S')}] {arm} wake {wake} turn {turn} user: {user!r}", flush=True)
            print(f"[{time.strftime('%H:%M:%S')}] {arm} wake {wake} turn {turn} assistant: "
                  f"{latest_reply!r}", flush=True)
            turns.append({"turn": turn, "goal": goal, "user": user, "assistant": latest_reply,
                          "request_sha256": _sha(request), "response_sha256": _sha(result),
                          "resume_status": result.get("resume_status"),
                          "token_usage": result.get("token_usage"),
                          "provider_metadata": {key: result.get(key) for key in
                                                ("provider", "model", "version", "work_dir")},
                          **token_fields})
            pending = None
            persist()
        metadata = state_metadata() if callable(state_metadata) else state_metadata
        metadata = dict(metadata or {})
        tokenized = all("prompt_token_ids" in turn and "assistant_token_ids" in turn for turn in turns)
        transcript_ids = ([token for turn in turns for key in ("prompt_token_ids", "assistant_token_ids")
                           for token in turn[key]] if tokenized else None)
        artifact: dict[str, object] = {
            "version": 1, "arm": arm, "wake": wake, "scenario": scenario,
            "plan": {"turn_count": self.plan.turn_count, "injection_turns": list(self.plan.injection_turns),
                     "turn_goals": list(self.plan.turn_goals)},
            "facts": [{"entity": entity, "code": code} for entity, code in facts],
            "generator": {"command": list(self.generator.command), "session_id": session,
                          **self.generator.provenance},
            "turns": turns, "state_metadata": metadata,
            "state_sha256": (metadata["sha256"] if isinstance(metadata.get("sha256"), str)
                             else _sha(metadata)),
            "fact_token_distances": fact_distances(turns) if fact_distances else None,
            "transcript_sha256": _sha(turns),
            "transcript_token_ids": transcript_ids,
            "transcript_token_sha256": token_sha(transcript_ids) if transcript_ids is not None else None,
        }
        artifact["artifact_sha256"] = _sha(artifact)
        self.store(artifact)
        partial_path.unlink()
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
                            "transcript_token_sha256": artifact.get("transcript_token_sha256"),
                            "state_sha256": artifact.get("state_sha256"),
                            **self.probe(arm, wake, state, artifact)})

        coordinator = MultiSleepCoordinator(self.manifest.config.arms, self.wake, self.fork_state,
                                             self.initial_state, lambda _: None)
        coordinator.run(
            facts_for_wake=lambda wake: counterbalanced_facts(
                self.manifest.wakes[wake - 1].facts, self.manifest.config.seeds.index(seed), wake,
            ),
            scenario_for_wake=lambda wake: self.manifest.wakes[wake - 1].scenario,
            sleep=self.sleep, probe=probe,
        )
        return {"records": records, "execution": {"batch_sizes": self.manifest.batch_sizes,
                "peak_vram_bytes": None, "throughput": None, "retries": 0}}
