"""Run the registered adaptive six-wake experiment from one frozen manifest."""

from __future__ import annotations

from experiments.adaptive.manifest import (
    ExperimentManifest,
    load_experiment_manifest,
    token_sha,
)
from experiments.adaptive.wake import (
    CommandUserGenerator,
    LiveWakeHarness,
)
from experiments.adaptive.coordinator import ExperimentRuntime
from experiments.adaptive.backend import (
    DreamSleepBackend,
    RuntimeBackend,
    artifact_transcript_ids,
    battery_collision_terms,
    bound_rehearsal_counts,
    dream_cache_identity,
    manifest_payload,
    manifest_sha,
)
from experiments.adaptive.analysis import (
    aggregate_seed_results,
    floor_correct_records,
    rehearsal_retention_analysis,
    retention_summary,
)
from experiments.adaptive.runner import run_registered_experiment
from experiments.adaptive.cli import main

__all__ = [
    "CommandUserGenerator",
    "DreamSleepBackend",
    "ExperimentManifest",
    "ExperimentRuntime",
    "LiveWakeHarness",
    "RuntimeBackend",
    "aggregate_seed_results",
    "artifact_transcript_ids",
    "battery_collision_terms",
    "bound_rehearsal_counts",
    "dream_cache_identity",
    "floor_correct_records",
    "load_experiment_manifest",
    "main",
    "manifest_payload",
    "manifest_sha",
    "rehearsal_retention_analysis",
    "retention_summary",
    "run_registered_experiment",
    "token_sha",
]


if __name__ == "__main__":
    main()
