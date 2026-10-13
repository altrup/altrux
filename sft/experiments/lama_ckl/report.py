"""Aggregate completed LAMA-CKL arms without selecting checkpoints by dream quality."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from experiments.lama_ckl.runner import ARMS, curve_summary
from progress import ts

SEEDS = (42, 43, 44)
TRAINED_ARMS = ("lora", "mix-review", "altrux")
SCORES = ("to_learn_accuracy", "not_to_forget_accuracy")
CYCLE_METRICS = (*SCORES, "role_marker_closes", "forced_closes", "loss")
CYCLE_DIAGNOSTICS = ("subjects_mentioned", "mean_copy_fraction")


def _stats(values: Sequence[float]) -> dict[str, object]:
    mean = sum(values) / len(values)
    stderr = (
        math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    return {"mean": mean, "stderr": stderr, "values": list(values)}


def scientific_runs(runs: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [run for run in runs if not bool(run["settings"].get("engineering_only"))]


def _load_run(path: Path) -> dict[str, object]:
    run = json.loads(path.read_text())
    run["persistent_artifact_bytes"] = sum(
        artifact.stat().st_size
        for artifact in path.parent.rglob("*")
        if artifact.is_file() and artifact.relative_to(path.parent).parts[0] != "work"
    )
    return run


def aggregate_runs(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not runs:
        raise ValueError("no completed runs")
    if any(bool(run["settings"].get("engineering_only")) for run in runs):
        raise ValueError("engineering-only smoke results cannot enter the scientific report")
    split_hashes = {run["settings"].get("split_manifest_sha256") for run in runs}
    if len(split_hashes) != 1:
        raise ValueError("runs use different split artifacts")
    core_keys = (
        "train_batch_size",
        "learning_rate",
        "evidence_tokens",
        "reply_tokens",
        "reply_temperature",
        "eval_batch_size",
        "requested_dream_batch_size",
        "model_name",
        "warmstart_sha256",
        "split_manifest_sha256",
        "lora_rank",
        "lora_alpha",
        "total_parameters",
        "source_document_tokens",
        "review_document_tokens",
    )
    for key in core_keys:
        if len({run["settings"].get(key) for run in runs}) != 1:
            raise ValueError(f"runs use different {key} settings")
    for key, arms in (
        ("docs_per_wake", ("altrux",)),
        ("dreams_per_cycle", ("altrux",)),
        *(("epochs", (arm,)) for arm in TRAINED_ARMS),
    ):
        if len({run["settings"].get(key) for run in runs if run["arm"] in arms}) > 1:
            raise ValueError(f"runs use different {key} settings")
    epochs = {str(run["arm"]): int(run["settings"]["epochs"]) for run in runs}
    trained_epochs = [epochs[arm] for arm in TRAINED_ARMS if arm in epochs]
    matched = min(trained_epochs) if trained_epochs else None
    matched_scores: dict[str, dict[str, list[float]]] = {}
    frozen_scores: dict[int, dict[str, list[float]]] = {}
    cycle_cells: dict[int, dict[str, list[float]]] = {}

    raw: dict[str, dict[str, list[float]]] = {}
    per_run: list[dict[str, object]] = []
    curves: dict[str, dict[int, dict[str, list[float]]]] = {}
    seen: set[tuple[str, int]] = set()
    for run in runs:
        arm, seed = str(run["arm"]), int(run["seed"])
        if arm in TRAINED_ARMS and (arm, seed) in seen:
            raise ValueError(f"duplicate run for {arm} seed {seed}")
        seen.add((arm, seed))
        curve = run["curve"]
        expected_epochs = list(range(int(run["settings"]["epochs"]) + 1))
        if [int(row["epoch"]) for row in curve] != expected_epochs:
            raise ValueError(f"{arm} seed {seed} has an incomplete epoch curve")
        summary = curve_summary(curve)
        initial, final = curve[0], curve[-1]
        wall_seconds = sum(float(row.get("epoch_seconds", 0.0)) for row in curve)
        metrics = {
            "top_accuracy": float(summary["top_accuracy"]),
            "peak_epoch": float(summary["epoch"]),
            "retention_at_peak": float(summary["not_to_forget_accuracy"]),
            "total_knowledge": float(summary["total_knowledge"]),
            "final_accuracy": float(final["to_learn_accuracy"]),
            "final_retention": float(final["not_to_forget_accuracy"]),
            "forgetting_at_peak": (
                float(initial["not_to_forget_accuracy"]) - float(summary["not_to_forget_accuracy"])
            ),
            "final_forgetting": (
                float(initial["not_to_forget_accuracy"]) - float(final["not_to_forget_accuracy"])
            ),
            "wall_seconds": wall_seconds,
            "gpu_hours": wall_seconds * int(run["settings"].get("gpu_count", 1)) / 3600,
            "artifact_bytes": float(
                run.get(
                    "persistent_artifact_bytes",
                    sum(float(row.get("artifact_bytes", 0.0)) for row in curve),
                )
            ),
            "optimizer_steps": sum(
                float(row.get("treatment", {}).get("optimizer_steps", 0.0)) for row in curve
            ),
            "token_gradients": sum(
                float(row.get("treatment", {}).get("token_gradients", 0.0)) for row in curve
            ),
            "generated_tokens": sum(
                float(row.get("treatment", {}).get("generated_tokens", 0.0)) for row in curve
            ),
            "source_tokens": sum(float(row.get("source_tokens", 0.0)) for row in curve),
            "wake_tokens": sum(float(row.get("wake_tokens") or 0.0) for row in curve),
            "review_tokens": sum(
                float(row.get("treatment", {}).get("review_tokens", 0.0)) for row in curve
            ),
            "total_parameters": float(run["settings"].get("total_parameters", 0.0)),
            "optimizer_parameters": float(run["settings"].get("optimizer_parameters", 0.0)),
            "peak_vram_bytes": max(float(row.get("peak_vram_bytes", 0.0)) for row in curve),
        }
        if arm == "frozen":
            for key in SCORES:
                frozen_scores.setdefault(seed, {}).setdefault(key, []).append(float(initial[key]))
        matched_row = (
            initial
            if arm == "frozen"
            else next((row for row in curve if int(row["epoch"]) == matched), None)
        )
        if matched_row is not None:
            for key in SCORES:
                matched_scores.setdefault(arm, {}).setdefault(key, []).append(
                    float(matched_row[key])
                )
        # Cycle numbers restart every epoch; altrux's comparison epoch is epoch 1.
        scored = [
            record
            for row in curve
            if int(row["epoch"]) == 1
            for record in row.get("treatment", {}).get("cycles", [])
            if "to_learn_accuracy" in record
        ]
        for record in scored:
            cell = cycle_cells.setdefault(int(record["cycle"]), {})
            for key in CYCLE_METRICS:
                cell.setdefault(key, []).append(float(record[key]))
            for key in CYCLE_DIAGNOSTICS:
                cell.setdefault(key, []).append(float(record["diagnostics"][key]))
        if scored:
            peak = max(scored, key=lambda record: float(record["to_learn_accuracy"]))
            metrics["peak_cycle_accuracy"] = float(peak["to_learn_accuracy"])
            metrics["peak_cycle"] = float(peak["cycle"])
            metrics["retention_at_peak_cycle"] = float(peak["not_to_forget_accuracy"])
        for name, value in metrics.items():
            raw.setdefault(arm, {}).setdefault(name, []).append(value)
        per_run.append(
            {
                "arm": arm,
                "seed": seed,
                "gpu_model": run["settings"].get("gpu_model"),
                "gpu_count": run["settings"].get("gpu_count"),
                "visible_gpu_count": run["settings"].get("visible_gpu_count"),
                "lora_rank": run["settings"].get("lora_rank"),
                "lora_alpha": run["settings"].get("lora_alpha"),
                **metrics,
            }
        )
        for row in curve:
            epoch = int(row["epoch"])
            cell = curves.setdefault(arm, {}).setdefault(
                epoch,
                {
                    "to_learn_accuracy": [],
                    "not_to_forget_accuracy": [],
                },
            )
            cell["to_learn_accuracy"].append(float(row["to_learn_accuracy"]))
            cell["not_to_forget_accuracy"].append(float(row["not_to_forget_accuracy"]))
    spreads = [
        {key: max(values[key]) - min(values[key]) for key in SCORES}
        for values in frozen_scores.values()
        if len(values["to_learn_accuracy"]) > 1
    ]
    return {
        "split_manifest_sha256": next(iter(split_hashes)),
        "epochs": epochs,
        "matched_epoch": {
            "epoch": matched,
            "arms": {
                arm: {key: _stats(values) for key, values in scores.items()}
                for arm, scores in matched_scores.items()
            },
        },
        "evaluation_resolution": {
            key: max((spread[key] for spread in spreads), default=None) for key in SCORES
        },
        "cycle_curves": (
            {
                "altrux": {
                    str(cycle): {key: _stats(values) for key, values in cell.items()}
                    for cycle, cell in sorted(cycle_cells.items())
                }
            }
            if cycle_cells
            else {}
        ),
        "runs": per_run,
        "arms": {
            arm: {name: _stats(values) for name, values in metrics.items()}
            for arm, metrics in raw.items()
        },
        "curves": {
            arm: {
                str(epoch): {name: _stats(values) for name, values in metrics.items()}
                for epoch, metrics in cells.items()
            }
            for arm, cells in curves.items()
        },
    }


def build_report(runs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    report = aggregate_runs(runs)
    present = {(str(run["arm"]), int(run["seed"])) for run in runs}
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    report["present_cells"] = [list(cell) for cell in sorted(present)]
    report["missing_cells"] = [list(cell) for cell in sorted(expected - present)]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("../.cache/lama_ckl/runs"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.glob("*/summary.json"))
    runs = scientific_runs([_load_run(path) for path in paths])
    if not runs:
        raise SystemExit(f"no scientific runs under {args.root}; nothing to report")
    report = build_report(runs)
    output = args.output or args.root / "report.json"
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    for arm, metrics in report["arms"].items():
        print(
            f"[{ts()}] {arm}: top={metrics['top_accuracy']['mean']:.6f} "
            f"epoch={metrics['peak_epoch']['mean']:.2f} "
            f"retained={metrics['retention_at_peak']['mean']:.6f} "
            f"final_forgetting={metrics['final_forgetting']['mean']:.6f}",
            flush=True,
        )
    matched = report["matched_epoch"]
    for arm, scores in matched["arms"].items():
        epoch = 0 if arm == "frozen" else matched["epoch"]
        print(
            f"[{ts()}] {arm} epoch {epoch}: to-learn "
            f"{scores['to_learn_accuracy']['mean']:.6f} retained "
            f"{scores['not_to_forget_accuracy']['mean']:.6f}",
            flush=True,
        )
    spread = {
        key: "n/a (fewer than 2 frozen replicates)" if value is None else f"{value:.6f}"
        for key, value in report["evaluation_resolution"].items()
    }
    print(
        f"[{ts()}] evaluation resolution (frozen replicate spread): "
        f"to-learn {spread['to_learn_accuracy']} retained {spread['not_to_forget_accuracy']}",
        flush=True,
    )
    altrux = report["arms"].get("altrux", {})
    if "peak_cycle" in altrux:
        print(
            f"[{ts()}] altrux peak cycle {altrux['peak_cycle']['mean']:.2f}: to-learn "
            f"{altrux['peak_cycle_accuracy']['mean']:.6f} retained "
            f"{altrux['retention_at_peak_cycle']['mean']:.6f} over "
            f"{len(report['cycle_curves']['altrux'])} cycles",
            flush=True,
        )
    missing = report["missing_cells"]
    print(
        f"[{ts()}] missing cells: {', '.join(f'{arm}/{seed}' for arm, seed in missing)}"
        if missing
        else f"[{ts()}] all {len(ARMS) * len(SEEDS)} registered cells present",
        flush=True,
    )
    print(f"[{ts()}] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
