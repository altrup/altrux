"""Run the registered adaptive six-wake experiment from one frozen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.adaptive.analysis import aggregate_seed_results
from experiments.adaptive.backend import DreamSleepBackend, manifest_sha
from experiments.adaptive.manifest import load_experiment_manifest

PARSER_OPTIONS = ("--manifest", "--seed", "--aggregate-only")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, help="Run one registered seed; omit to run all three")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Aggregate completed seed JSON files without loading a model",
    )
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
        if any(
            not isinstance(result.get("execution"), dict)
            or result["execution"].get("manifest_sha256") != manifest_sha(manifest)
            for result in results
        ):
            raise SystemExit("cannot aggregate seed results from a different manifest")
    else:
        from experiments.adaptive.runner import run_registered_experiment

        results = []
        for seed in seeds:
            completed = Path(manifest.output_root) / f"seed-{seed}.json"
            if completed.exists():
                loaded = json.loads(completed.read_text())
                if not isinstance(loaded.get("execution"), dict) or loaded["execution"].get(
                    "manifest_sha256"
                ) != manifest_sha(manifest):
                    raise SystemExit(f"completed seed {seed} belongs to a different manifest")
                results.append(loaded)
            else:
                results.append(
                    run_registered_experiment(manifest, seed, DreamSleepBackend(manifest, seed))
                )
    if args.seed is None:
        aggregate = aggregate_seed_results(results, manifest.config.wakes)
        DreamSleepBackend._store_json(Path(manifest.output_root) / "aggregate.json", aggregate)


__all__ = ["PARSER_OPTIONS", "main"]


if __name__ == "__main__":
    main()
