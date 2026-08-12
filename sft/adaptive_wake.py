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


def load_wake_plan(path: str | Path) -> WakePlan:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptiveWakeError(f"cannot read wake plan {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdaptiveWakeError("wake plan must be a JSON object")
    return WakePlan.from_dict(value)


class CommandUserGenerator:
    """Provider-neutral JSON command adapter; a resume never starts afresh."""

    def __init__(self, command: Sequence[str], timeout: float = 120.0):
        if not command:
            raise AdaptiveWakeError("user-generator command cannot be empty")
        self.command = tuple(command)
        self.timeout = timeout

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
            *, state_metadata: dict[str, object] | None = None) -> dict[str, object]:
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
                          "resume_status": result.get("resume_status")})
        artifact: dict[str, object] = {
            "version": 1, "arm": arm, "wake": wake, "scenario": scenario,
            "plan": {"turn_count": self.plan.turn_count, "injection_turns": list(self.plan.injection_turns)},
            "facts": [{"entity": entity, "code": code} for entity, code in facts],
            "generator": {"command": list(self.generator.command), "session_id": session},
            "turns": turns, "state_metadata": state_metadata or {}, "transcript_sha256": _sha(turns),
        }
        artifact["artifact_sha256"] = _sha(artifact)
        path = self.artifact_dir / f"{arm}_w{wake}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = _canonical(artifact) + "\n"
        if path.exists() and path.read_text() != serialized:
            raise AdaptiveWakeError(f"wake artifact already exists with different content: {path}")
        if not path.exists():
            path.write_text(serialized)
        return artifact
