"""Live adaptive wake realization and provider control."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence

from experiments.adaptive.manifest import AdaptiveWakeError, WakePlan, _canonical, _sha, token_sha


class CommandUserGenerator:
    """Provider-neutral JSON command adapter; a resume never starts afresh."""

    def __init__(
        self,
        command: Sequence[str],
        timeout: float = 120.0,
        provenance: dict[str, str] | None = None,
    ):
        if not command:
            raise AdaptiveWakeError("user-generator command cannot be empty")
        self.command = tuple(command)
        self.timeout = timeout
        self.provenance = dict(provenance or {})

    def next_user(self, request: dict[str, object]) -> dict[str, object]:
        try:
            completed = subprocess.run(
                self.command,
                input=_canonical(request),
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
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
        if prior is not None and (
            result.get("resume_status") != "resumed" or result.get("session_id") != prior
        ):
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

    def run(
        self,
        arm: str,
        wake: int,
        facts: Sequence[tuple[str, str]],
        local_reply: Callable[[str, int], str],
        scenario: str,
        session_id: str | None,
        *,
        state_metadata: dict[str, object] | Callable[[], dict[str, object]] | None = None,
        fact_distances: Callable[[Sequence[dict[str, object]]], dict[str, int]] | None = None,
        replay_turn: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        if len(facts) != 4:
            raise AdaptiveWakeError("each wake requires exactly four facts")
        final_path = self.artifact_dir / f"{arm}_w{wake}.json"
        partial_path = self.artifact_dir / f"{arm}_w{wake}.partial.json"
        if final_path.exists():
            artifact = json.loads(final_path.read_text())
            expected_plan = {
                "turn_count": self.plan.turn_count,
                "injection_turns": list(self.plan.injection_turns),
                "turn_goals": list(self.plan.turn_goals),
            }
            if (
                artifact.get("scenario") != scenario
                or artifact.get("plan") != expected_plan
                or artifact.get("facts")
                != [{"entity": entity, "code": code} for entity, code in facts]
            ):
                raise AdaptiveWakeError(
                    f"completed wake does not match requested inputs: {final_path}"
                )
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
            if (
                partial.get("scenario") != scenario
                or partial.get("facts") != [list(fact) for fact in facts]
                or partial.get("turn_goals") != list(self.plan.turn_goals)
                or partial.get("injection_turns") != list(self.plan.injection_turns)
            ):
                raise AdaptiveWakeError(
                    f"incomplete wake does not match requested inputs: {partial_path}"
                )
            turns = list(partial.get("turns", []))
            pending = partial.get("pending") if isinstance(partial.get("pending"), dict) else None
            session = (
                partial.get("session_id") if isinstance(partial.get("session_id"), str) else session
            )
            latest_reply = str(turns[-1]["assistant"]) if turns else ""
            if replay_turn:
                for stored in turns:
                    replay_turn(stored)

        def persist() -> None:
            value = {
                "version": 1,
                "arm": arm,
                "wake": wake,
                "scenario": scenario,
                "facts": [list(fact) for fact in facts],
                "turn_goals": list(self.plan.turn_goals),
                "injection_turns": list(self.plan.injection_turns),
                "session_id": session,
                "turns": turns,
                "pending": pending,
            }
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = partial_path.with_suffix(".tmp")
            temporary.write_text(_canonical(value) + "\n")
            os.replace(temporary, partial_path)

        for turn in range(len(turns) + 1, self.plan.turn_count + 1):
            fact = fact_by_turn.get(turn)
            intent = self.plan.turn_goals[turn - 1]
            goal = (
                f"{intent}\nCommunicate that {fact[0]} has code {fact[1]} exactly once."
                if fact
                else intent
            )
            request = {
                "latest_assistant_reply": latest_reply,
                "goal": goal,
                "scenario": scenario,
                "turn": turn,
            }
            if session is not None:
                request["session_id"] = session
            stamp = time.strftime("%H:%M:%S")
            print(
                f"[{stamp}] {arm} wake {wake} turn {turn}/{self.plan.turn_count}: requesting user",
                flush=True,
            )
            result = (
                pending["result"]
                if pending and pending.get("turn") == turn
                else self.generator.next_user(request)
            )
            user, session = str(result["message"]), str(result["session_id"])
            if fact and (fact[0].casefold() not in user.casefold() or user.count(fact[1]) != 1):
                raise AdaptiveWakeError(
                    f"injection turn {turn} must communicate entity {fact[0]!r} and code {fact[1]!r} once"
                )
            if fact and len(re.findall(r"[A-Za-z]+", user)) < 3:
                raise AdaptiveWakeError(
                    f"injection turn {turn} must use a natural-language user message"
                )
            pending = {"turn": turn, "request": request, "result": result}
            persist()
            reply = local_reply(user, turn)
            token_fields: dict[str, object] = {}
            if isinstance(reply, dict):
                latest_reply = reply.get("text")
                prompt_ids, assistant_ids = (
                    reply.get("prompt_token_ids"),
                    reply.get("assistant_token_ids"),
                )
                if (
                    not isinstance(latest_reply, str)
                    or not isinstance(prompt_ids, list)
                    or not isinstance(assistant_ids, list)
                    or not assistant_ids
                    or any(not isinstance(token, int) for token in prompt_ids + assistant_ids)
                ):
                    raise AdaptiveWakeError(
                        "local reply requires text and consumed prompt/assistant token IDs"
                    )
                token_fields = {
                    "prompt_token_ids": prompt_ids,
                    "assistant_token_ids": assistant_ids,
                }
                if isinstance(reply.get("stop_reason"), str):
                    token_fields["assistant_stop_reason"] = reply["stop_reason"]
            elif isinstance(reply, str):
                latest_reply = reply
            else:
                raise AdaptiveWakeError("local reply must be text or a tokenized reply object")
            print(
                f"[{time.strftime('%H:%M:%S')}] {arm} wake {wake} turn {turn} user: {user!r}",
                flush=True,
            )
            print(
                f"[{time.strftime('%H:%M:%S')}] {arm} wake {wake} turn {turn} assistant: "
                f"{latest_reply!r}",
                flush=True,
            )
            turns.append(
                {
                    "turn": turn,
                    "goal": goal,
                    "user": user,
                    "assistant": latest_reply,
                    "request_sha256": _sha(request),
                    "response_sha256": _sha(result),
                    "resume_status": result.get("resume_status"),
                    "token_usage": result.get("token_usage"),
                    "provider_metadata": {
                        key: result.get(key) for key in ("provider", "model", "version", "work_dir")
                    },
                    **token_fields,
                }
            )
            pending = None
            persist()
        metadata = state_metadata() if callable(state_metadata) else state_metadata
        metadata = dict(metadata or {})
        tokenized = all(
            "prompt_token_ids" in turn and "assistant_token_ids" in turn for turn in turns
        )
        transcript_ids = (
            [
                token
                for turn in turns
                for key in ("prompt_token_ids", "assistant_token_ids")
                for token in turn[key]
            ]
            if tokenized
            else None
        )
        artifact: dict[str, object] = {
            "version": 1,
            "arm": arm,
            "wake": wake,
            "scenario": scenario,
            "plan": {
                "turn_count": self.plan.turn_count,
                "injection_turns": list(self.plan.injection_turns),
                "turn_goals": list(self.plan.turn_goals),
            },
            "facts": [{"entity": entity, "code": code} for entity, code in facts],
            "generator": {
                "command": list(self.generator.command),
                "session_id": session,
                **self.generator.provenance,
            },
            "turns": turns,
            "state_metadata": metadata,
            "state_sha256": (
                metadata["sha256"] if isinstance(metadata.get("sha256"), str) else _sha(metadata)
            ),
            "fact_token_distances": fact_distances(turns) if fact_distances else None,
            "transcript_sha256": _sha(turns),
            "transcript_token_ids": transcript_ids,
            "transcript_token_sha256": token_sha(transcript_ids)
            if transcript_ids is not None
            else None,
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
