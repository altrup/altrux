"""Run the registered adaptive six-wake experiment from one frozen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_wake import (
    CommandUserGenerator,
    ExperimentManifest,
    ExperimentRuntime,
    LiveWakeHarness,
    floor_correct_records,
    load_experiment_manifest,
    retention_summary,
    token_sha,
)
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
from experiments.adaptive.analysis import aggregate_seed_results, rehearsal_retention_analysis


def run_registered_experiment(manifest: ExperimentManifest, seed: int,
                              backend: RuntimeBackend) -> dict[str, object]:
    output_root = Path(manifest.output_root)
    stream_path = output_root / f"seed-{seed}.jsonl"
    final_path = Path(manifest.output_root) / f"seed-{seed}.json"
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        loaded = json.loads(final_path.read_text())
        execution = loaded.get("execution") if isinstance(loaded, dict) else None
        if (isinstance(loaded, dict) and loaded.get("seed") == seed and isinstance(execution, dict)
                and execution.get("manifest_sha256") == manifest_sha(manifest)):
            return loaded
        raise RuntimeError(f"completed seed result is invalid: {final_path}")
    if stream_path.exists():
        resume = 1
        while (output_root / f"seed-{seed}.resume-{resume}.jsonl").exists():
            resume += 1
        stream_path = output_root / f"seed-{seed}.resume-{resume}.jsonl"
    stream = stream_path.open("x")

    def probe(arm: str, wake: int, state: object, artifact: dict[str, object]) -> dict[str, object]:
        details = backend.probe(arm, wake, state, artifact)
        stream.write(json.dumps({"seed": seed, "arm": arm, "wake": wake,
                                 "artifact_sha256": artifact.get("artifact_sha256"),
                                 "transcript_token_sha256": artifact.get("transcript_token_sha256"),
                                 "state_sha256": artifact.get("state_sha256"), **details}) + "\n")
        stream.flush()
        return details

    runtime = ExperimentRuntime(
        manifest, initial_state=backend.initial_state, fork_state=backend.fork_state,
        wake=backend.wake, sleep=backend.sleep, probe=probe,
    )
    try:
        result = runtime.run(seed=seed)
    finally:
        stream.close()
    records = result.get("records")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise RuntimeError("experiment backend returned malformed records")
    wake_one = [record for record in records if record.get("wake") == 1]
    if (len(wake_one) != 3
            or len({record.get("transcript_token_sha256") for record in wake_one}) != 1
            or len({record.get("state_sha256") for record in wake_one}) != 1
            or any(record.get("transcript_token_sha256") is None or record.get("state_sha256") is None
                   for record in wake_one)):
        raise RuntimeError("Wake 1 transcript tokens and state must be identical across all arms")
    corrected = floor_correct_records(records)
    metadata = backend.execution_metadata()
    expected_manifest = manifest_sha(manifest)
    if metadata.get("manifest_sha256") not in (None, expected_manifest):
        raise RuntimeError("backend manifest SHA does not match the registered manifest")
    result = {"seed": seed, "records": corrected,
              "retention": retention_summary(corrected, manifest.config.wakes),
              "execution": {"batch_sizes": manifest.batch_sizes, **metadata,
                            "manifest_sha256": expected_manifest}}
    path = final_path
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=1, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != serialized:
        raise RuntimeError(f"seed result already exists with different content: {path}")
    if not path.exists():
        path.write_text(serialized)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, help="Run one registered seed; omit to run all three")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Aggregate completed seed JSON files without loading a model")
    args = parser.parse_args()
    manifest = load_experiment_manifest(args.manifest)
    seeds = [args.seed] if args.seed is not None else list(manifest.config.seeds)
    if args.aggregate_only:
        if args.seed is not None:
            raise SystemExit("--aggregate-only cannot be combined with --seed")
        paths = [Path(manifest.output_root) / f"seed-{seed}.json" for seed in seeds]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise SystemExit(f"cannot aggregate missing seed results: {missing}")
        results = [json.loads(path.read_text()) for path in paths]
        if any(not isinstance(result.get("execution"), dict)
               or result["execution"].get("manifest_sha256") != manifest_sha(manifest) for result in results):
            raise SystemExit("cannot aggregate seed results from a different manifest")
    else:
        results = []
        for seed in seeds:
            completed = Path(manifest.output_root) / f"seed-{seed}.json"
            if completed.exists():
                loaded = json.loads(completed.read_text())
                if (not isinstance(loaded.get("execution"), dict)
                        or loaded["execution"].get("manifest_sha256") != manifest_sha(manifest)):
                    raise SystemExit(f"completed seed {seed} belongs to a different manifest")
                results.append(loaded)
            else:
                results.append(run_registered_experiment(manifest, seed, DreamSleepBackend(manifest, seed)))
    if args.seed is None:
        aggregate = aggregate_seed_results(results, manifest.config.wakes)
        DreamSleepBackend._store_json(Path(manifest.output_root) / "aggregate.json", aggregate)


if __name__ == "__main__":
    main()
