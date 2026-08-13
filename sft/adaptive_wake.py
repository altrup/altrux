"""Live adaptive wakes with an explicit, frozen turn plan."""

from __future__ import annotations

from typing import Sequence

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
from experiments.adaptive.coordinator import ExperimentRuntime, MultiSleepCoordinator

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
