import json
from pathlib import Path

import pytest

from experiments.lama_ckl.report import _load_run, aggregate_runs, scientific_runs


def test_aggregate_runs_reports_acquisition_and_forgetting_by_arm():
    settings = {
        "epochs": 1,
        "engineering_only": False,
        "eval_batch_size": 16,
        "requested_dream_batch_size": 8,
        "split_manifest_sha256": "x",
        "gpu_model": "NVIDIA GH200",
        "gpu_count": 1,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "total_parameters": 2_700_000_000,
        "optimizer_parameters": 1_000_000,
    }
    runs = [
        {
            "arm": "frozen",
            "seed": 42,
            "settings": settings | {"optimizer_parameters": 0},
            "curve": [
                {"epoch": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
                {"epoch": 1, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
            ],
        },
        {
            "arm": "altrux",
            "seed": 42,
            "settings": settings,
            "curve": [
                {"epoch": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
                {
                    "epoch": 1,
                    "to_learn_accuracy": 0.2,
                    "not_to_forget_accuracy": 0.9,
                    "source_tokens": 400,
                    "wake_tokens": 500,
                    "treatment": {
                        "review_tokens": 100,
                        "token_gradients": 300,
                        "generated_tokens": 200,
                    },
                },
            ],
        },
    ]

    report = aggregate_runs(runs)

    assert report["arms"]["altrux"]["top_accuracy"]["mean"] == 0.2
    assert report["arms"]["altrux"]["final_forgetting"]["mean"] == pytest.approx(0.1)
    assert report["arms"]["altrux"]["source_tokens"]["mean"] == 400
    assert report["arms"]["altrux"]["wake_tokens"]["mean"] == 500
    assert report["arms"]["altrux"]["review_tokens"]["mean"] == 100
    assert report["runs"][1]["gpu_model"] == "NVIDIA GH200"
    assert report["runs"][1]["gpu_count"] == 1
    assert report["runs"][1]["lora_rank"] == 8
    assert report["runs"][1]["total_parameters"] == 2_700_000_000
    assert report["arms"]["frozen"]["final_forgetting"]["mean"] == 0.0


def test_aggregate_runs_rejects_smokes_and_mixed_splits():
    run = {
        "arm": "frozen",
        "seed": 42,
        "curve": [
            {"epoch": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
        ],
    }

    with pytest.raises(ValueError, match="engineering-only"):
        aggregate_runs(
            [
                {
                    **run,
                    "settings": {
                        "epochs": 0,
                        "engineering_only": True,
                        "eval_batch_size": 16,
                        "requested_dream_batch_size": 8,
                        "split_manifest_sha256": "x",
                    },
                }
            ]
        )
    with pytest.raises(ValueError, match="split"):
        aggregate_runs(
            [
                {
                    **run,
                    "settings": {
                        "epochs": 0,
                        "engineering_only": False,
                        "eval_batch_size": 16,
                        "requested_dream_batch_size": 8,
                        "split_manifest_sha256": "x",
                    },
                },
                {
                    **run,
                    "seed": 43,
                    "settings": {
                        "epochs": 0,
                        "engineering_only": False,
                        "eval_batch_size": 16,
                        "requested_dream_batch_size": 8,
                        "split_manifest_sha256": "y",
                    },
                },
            ]
        )


@pytest.mark.parametrize("epochs", [[0, 1], [0, 2]])
def test_aggregate_runs_rejects_incomplete_epoch_curves(epochs: list[int]):
    run = {
        "arm": "frozen",
        "seed": 42,
        "settings": {
            "epochs": 2,
            "engineering_only": False,
            "eval_batch_size": 16,
            "requested_dream_batch_size": 8,
            "split_manifest_sha256": "x",
        },
        "curve": [{"epoch": epoch} for epoch in epochs],
    }

    with pytest.raises(ValueError, match="epoch curve"):
        aggregate_runs([run])


@pytest.mark.parametrize("key", ["eval_batch_size", "requested_dream_batch_size"])
def test_aggregate_runs_rejects_mixed_batch_settings(key: str):
    settings = {
        "epochs": 0,
        "engineering_only": False,
        "eval_batch_size": 16,
        "requested_dream_batch_size": 8,
        "split_manifest_sha256": "x",
    }
    runs = [
        {"arm": "frozen", "seed": 42, "settings": settings, "curve": [{"epoch": 0}]},
        {
            "arm": "lora",
            "seed": 42,
            "settings": settings | {key: settings[key] * 2},
            "curve": [{"epoch": 0}],
        },
    ]

    with pytest.raises(ValueError, match=key):
        aggregate_runs(runs)


def test_scientific_runs_ignores_adjacent_smoke_directories():
    full = {"settings": {"engineering_only": False}}
    smoke = {"settings": {"engineering_only": True}}

    assert scientific_runs([smoke, full]) == [full]


def test_load_run_counts_persistent_files_but_not_work_files(tmp_path: Path):
    run = tmp_path / "arm-seed-42"
    work = run / "work"
    work.mkdir(parents=True)
    summary = run / "summary.json"
    summary.write_text(json.dumps({"settings": {"engineering_only": False}}))
    (run / "run.json").write_text("settings")
    (work / "partial.pt").write_bytes(b"unfinished")

    loaded = _load_run(summary)

    assert loaded["persistent_artifact_bytes"] == (
        summary.stat().st_size + (run / "run.json").stat().st_size
    )


def test_aggregate_runs_accepts_a_one_point_frozen_curve_beside_trained_arms():
    settings = {"split_manifest_sha256": "x", "epochs": 1, "docs_per_wake": 10}
    point = {"to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0}
    runs = [
        {
            "arm": "frozen",
            "seed": 42,
            "settings": settings | {"epochs": 0, "docs_per_wake": None},
            "curve": [{"epoch": 0, "wake_tokens": None, **point}],
        },
        {
            "arm": "altrux",
            "seed": 42,
            "settings": settings,
            "curve": [{"epoch": 0, **point}, {"epoch": 1, "wake_tokens": 30, **point}],
        },
    ]

    report = aggregate_runs(runs)

    assert report["arms"]["frozen"]["final_forgetting"]["mean"] == 0.0
    assert report["arms"]["altrux"]["wake_tokens"]["mean"] == 30
