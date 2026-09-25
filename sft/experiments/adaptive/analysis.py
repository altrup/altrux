"""Experiment: state-erasure

Aggregation and retention analysis for adaptive experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

from experiments.adaptive.manifest import AdaptiveWakeError


def floor_correct_records(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Subtract each seed/wake/fact's matched no-sleep margin."""
    floors: dict[tuple[object, object, str], float] = {}
    for record in records:
        if record.get("arm") != "nosleep" or not isinstance(record.get("facts"), dict):
            continue
        for fact, stats in record["facts"].items():
            if (
                isinstance(fact, str)
                and isinstance(stats, dict)
                and isinstance(stats.get("margin"), (int, float))
            ):
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
                margin, floor = (
                    values.get("margin"),
                    floors.get((record.get("seed"), record.get("wake"), fact)),
                )
                if (
                    record.get("arm") != "nosleep"
                    and isinstance(margin, (int, float))
                    and floor is None
                ):
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


def retention_summary(
    records: Sequence[dict[str, object]], wakes: int
) -> dict[str, dict[str, object]]:
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
                margins = [
                    float(value["margin"])
                    for value in values
                    if isinstance(value.get("margin"), (int, float))
                ]
                deltas = [
                    float(value["floor_corrected_margin"])
                    for value in values
                    if isinstance(value.get("floor_corrected_margin"), (int, float))
                ]
                if margins:
                    raw[evaluation - 1][learning - 1] = sum(margins) / len(margins)
                if deltas:
                    mean = sum(deltas) / len(deltas)
                    corrected[evaluation - 1][learning - 1] = mean
                    if evaluation == learning:
                        own[learning] = mean
                    if evaluation == wakes:
                        final[learning] = mean
        changes = [
            final[learning] - own[learning]
            for learning in own.keys() & final.keys()
            if learning < wakes
        ]
        summaries[arm] = {
            "r_matrix": raw,
            "floor_corrected_r_matrix": corrected,
            "cumulative_installed": installed,
            "backward_transfer": sum(changes) / len(changes) if changes else None,
        }
    return summaries


def _uncertainty(values: Sequence[object]) -> dict[str, object]:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if len(numbers) != len(values) or not numbers:
        return {"mean": None, "stderr": None, "values": list(values)}
    mean = sum(numbers) / len(numbers)
    stderr = (
        math.sqrt(sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1))
        / math.sqrt(len(numbers))
        if len(numbers) > 1
        else 0.0
    )
    return {"mean": mean, "stderr": stderr, "values": list(values)}


def rehearsal_retention_analysis(results: Sequence[dict[str, object]]) -> dict[str, object]:
    observations: list[dict[str, object]] = []
    for result in results:
        for record in result.get("records", []):
            if (
                not isinstance(record, dict)
                or record.get("arm") != "replay"
                or not isinstance(record.get("wake"), int)
            ):
                continue
            facts, treatment = record.get("facts"), record.get("treatment")
            counts = treatment.get("bound_rehearsals") if isinstance(treatment, dict) else None
            if not isinstance(facts, dict) or not isinstance(counts, dict):
                continue
            for fact, count in counts.items():
                stats = facts.get(fact)
                if (
                    not isinstance(fact, str)
                    or not isinstance(count, int)
                    or not isinstance(stats, dict)
                    or not isinstance(stats.get("fact_wave"), int)
                    or int(stats["fact_wave"]) >= int(record["wake"])
                    or not isinstance(stats.get("floor_corrected_margin"), (int, float))
                ):
                    continue
                observations.append(
                    {
                        "seed": result.get("seed"),
                        "sleep_wake": record["wake"],
                        "fact": fact,
                        "learning_wake": stats["fact_wave"],
                        "rehearsals": count,
                        "floor_corrected_margin": float(stats["floor_corrected_margin"]),
                    }
                )
    by_count: dict[str, object] = {}
    for count in sorted({int(item["rehearsals"]) for item in observations}):
        grouped: dict[object, list[float]] = {}
        for item in observations:
            if item["rehearsals"] == count:
                grouped.setdefault(item["seed"], []).append(float(item["floor_corrected_margin"]))
        seed_values = {str(seed): sum(values) / len(values) for seed, values in grouped.items()}
        summary = _uncertainty(list(seed_values.values()))
        summary["seed_values"] = seed_values
        by_count[str(count)] = summary
    return {"observations": observations, "by_rehearsal_count": by_count}


def aggregate_seed_results(results: Sequence[dict[str, object]], wakes: int) -> dict[str, object]:
    arms: dict[str, object] = {}
    arm_names = sorted(
        {
            arm
            for result in results
            if isinstance(result.get("retention"), dict)
            for arm in result["retention"]
        }
    )
    for arm in arm_names:
        summaries = [
            result["retention"].get(arm, {})
            for result in results
            if isinstance(result.get("retention"), dict)
        ]
        matrix = [
            [
                _uncertainty(
                    [
                        summary.get(
                            "floor_corrected_r_matrix", [[None] * wakes for _ in range(wakes)]
                        )[row][column]
                        if isinstance(summary, dict)
                        else None
                        for summary in summaries
                    ]
                )
                for column in range(wakes)
            ]
            for row in range(wakes)
        ]
        cumulative = [
            _uncertainty(
                [
                    summary.get("cumulative_installed", [None] * wakes)[wake]
                    if isinstance(summary, dict)
                    else None
                    for summary in summaries
                ]
            )
            for wake in range(wakes)
        ]
        arms[arm] = {
            "floor_corrected_r_matrix": matrix,
            "cumulative_installed": cumulative,
            "backward_transfer": _uncertainty(
                [
                    summary.get("backward_transfer") if isinstance(summary, dict) else None
                    for summary in summaries
                ]
            ),
        }
        records_by_seed = [
            {
                int(record["wake"]): record
                for record in result.get("records", [])
                if isinstance(record, dict)
                and record.get("arm") == arm
                and isinstance(record.get("wake"), int)
            }
            for result in results
        ]

        def cell(record: dict[str, object], metric: str) -> float | int | None:
            facts = record.get("facts")
            if metric in (
                "corrected_margin",
                "cumulative_margin",
                "exact_match",
                "paraphrase_rate",
            ):
                if not isinstance(facts, dict):
                    return None
                key = (
                    "floor_corrected_margin"
                    if metric in ("corrected_margin", "cumulative_margin")
                    else metric
                )
                values = [
                    float(stats[key])
                    for stats in facts.values()
                    if isinstance(stats, dict) and isinstance(stats.get(key), (int, float, bool))
                ]
                if not values:
                    return None
                return sum(values) if metric == "cumulative_margin" else sum(values) / len(values)
            if metric == "ppl_delta":
                return (
                    record.get("ppl_delta")
                    if isinstance(record.get("ppl_delta"), (int, float))
                    else None
                )
            battery = record.get("battery")
            key = metric.removeprefix("battery_")
            return (
                battery.get(key)
                if isinstance(battery, dict) and isinstance(battery.get(key), (int, float))
                else None
            )

        metric_names = (
            "corrected_margin",
            "cumulative_margin",
            "exact_match",
            "paraphrase_rate",
            "battery_lost",
            "battery_loss_rate",
            "battery_mean_logprob_delta",
            "battery_median_logprob_delta",
            "battery_p10_logprob_delta",
            "ppl_delta",
        )
        arms[arm]["metrics"] = {
            metric: [
                _uncertainty(
                    [cell(seed_records.get(wake, {}), metric) for seed_records in records_by_seed]
                )
                for wake in range(1, wakes + 1)
            ]
            for metric in metric_names
        }
    return {
        "seeds": [result.get("seed") for result in results],
        "arms": arms,
        "rehearsal_retention": rehearsal_retention_analysis(results),
    }


__all__ = [
    "floor_correct_records",
    "retention_summary",
    "rehearsal_retention_analysis",
    "aggregate_seed_results",
]
