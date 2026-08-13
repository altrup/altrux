"""Live adaptive wakes with an explicit, frozen turn plan."""

from __future__ import annotations

from typing import Callable, Sequence

from experiments.adaptive.manifest import (
    AdaptiveWakeError,
    ExperimentConfig,
    ExperimentManifest,
    WakePlan,
    WakeSpec,
    _canonical,
    _sha,
    counterbalanced_facts,
    load_experiment_manifest,
    load_wake_plan,
    token_sha,
)
from experiments.adaptive.wake import CommandUserGenerator, LiveWakeHarness

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
