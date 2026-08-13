"""Manifest schemas, validation, identity, and counterbalancing."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


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


__all__ = [
    "AdaptiveWakeError",
    "ExperimentConfig",
    "ExperimentManifest",
    "WakePlan",
    "WakeSpec",
    "counterbalanced_facts",
    "load_experiment_manifest",
    "load_wake_plan",
    "token_sha",
]
