"""Registered adaptive experiment persistence and execution."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.adaptive.analysis import floor_correct_records, retention_summary
from experiments.adaptive.backend import RuntimeBackend, manifest_sha
from experiments.adaptive.coordinator import ExperimentRuntime
from experiments.adaptive.manifest import ExperimentManifest


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

