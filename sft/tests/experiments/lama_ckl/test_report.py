import json
from pathlib import Path

import pytest

from experiments.lama_ckl.report import _load_run, aggregate_runs, build_report, scientific_runs


def _cycle(cycle: int, learned: float, retained: float, subjects: int) -> dict[str, object]:
    return {
        "cycle": cycle,
        "to_learn_accuracy": learned,
        "not_to_forget_accuracy": retained,
        "loss": 0.5,
        "role_marker_closes": 0,
        "forced_closes": 1,
        "diagnostics": {"subjects_mentioned": subjects, "mean_copy_fraction": 0.25},
    }


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
            "settings": settings | {"epochs": 0, "optimizer_parameters": 0},
            "curve": [{"epoch": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0}],
        },
        {
            "arm": "dream",
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
                        "cycles": [
                            _cycle(1, 0.1, 0.95, 2),
                            _cycle(2, 0.3, 0.85, 4),
                            _cycle(3, 0.3, 0.8, 0),
                        ],
                    },
                },
            ],
        },
        {
            "arm": "lora",
            "seed": 42,
            "settings": settings | {"epochs": 2},
            "curve": [
                {"epoch": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
                {"epoch": 1, "to_learn_accuracy": 0.4, "not_to_forget_accuracy": 0.7},
                {"epoch": 2, "to_learn_accuracy": 0.6, "not_to_forget_accuracy": 0.5},
            ],
        },
    ]

    report = aggregate_runs(runs)

    assert report["arms"]["dream"]["top_accuracy"]["mean"] == 0.2
    assert report["arms"]["dream"]["final_forgetting"]["mean"] == pytest.approx(0.1)
    assert report["arms"]["dream"]["source_tokens"]["mean"] == 400
    assert report["arms"]["dream"]["wake_tokens"]["mean"] == 500
    assert report["arms"]["dream"]["review_tokens"]["mean"] == 100
    assert report["runs"][1]["gpu_model"] == "NVIDIA GH200"
    assert report["runs"][1]["gpu_count"] == 1
    assert report["runs"][1]["lora_rank"] == 8
    assert report["runs"][1]["total_parameters"] == 2_700_000_000
    assert report["arms"]["frozen"]["final_forgetting"]["mean"] == 0.0
    assert report["epochs"] == {"frozen": 0, "dream": 1, "lora": 2}
    matched = report["matched_epoch"]
    assert matched["epoch"] == 1
    assert set(matched["arms"]) == {"frozen", "dream", "lora"}
    assert matched["arms"]["lora"]["to_learn_accuracy"]["mean"] == 0.4
    assert matched["arms"]["dream"]["not_to_forget_accuracy"]["mean"] == 0.9
    assert matched["arms"]["frozen"]["not_to_forget_accuracy"]["mean"] == 1.0
    assert report["evaluation_resolution"] == {
        "to_learn_accuracy": None,
        "not_to_forget_accuracy": None,
    }
    cycles = report["cycle_curves"]["dream"]
    assert set(cycles) == {"1", "2", "3"}
    assert cycles["2"]["to_learn_accuracy"]["mean"] == 0.3
    assert cycles["2"]["subjects_mentioned"]["mean"] == 4
    assert cycles["3"]["mean_copy_fraction"]["mean"] == 0.25
    assert cycles["1"]["forced_closes"]["mean"] == 1
    assert cycles["1"]["loss"]["mean"] == 0.5
    dream_arm = report["arms"]["dream"]
    assert dream_arm["peak_cycle_accuracy"]["mean"] == 0.3
    assert dream_arm["peak_cycle"]["mean"] == 2
    assert dream_arm["retention_at_peak_cycle"]["mean"] == 0.85
    assert "peak_cycle" not in report["arms"]["lora"]


def _frozen(seed: int, learned: float, retained: float) -> dict[str, object]:
    return {
        "arm": "frozen",
        "seed": seed,
        "settings": {"split_manifest_sha256": "x", "epochs": 0},
        "curve": [{"epoch": 0, "to_learn_accuracy": learned, "not_to_forget_accuracy": retained}],
    }


def test_frozen_replicates_aggregate_and_their_spread_is_the_resolution():
    report = aggregate_runs(
        [_frozen(42, 0.01, 0.90), _frozen(42, 0.03, 0.88), _frozen(42, 0.02, 0.91)]
    )

    assert report["arms"]["frozen"]["final_accuracy"]["values"] == [0.01, 0.03, 0.02]
    assert report["evaluation_resolution"]["to_learn_accuracy"] == pytest.approx(0.02)
    assert report["evaluation_resolution"]["not_to_forget_accuracy"] == pytest.approx(0.03)
    assert report["matched_epoch"]["arms"]["frozen"]["to_learn_accuracy"]["mean"] == (
        pytest.approx(0.02)
    )


def test_duplicate_trained_seeds_still_refuse():
    lora = {**_frozen(42, 0.0, 1.0), "arm": "lora"}

    with pytest.raises(ValueError, match="duplicate"):
        aggregate_runs([lora, lora])


def test_acceptance_cycle_records_without_accuracies_are_skipped():
    point = {"to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0}
    acceptance = {"cycle": 1, "loss": 0.5, "diagnostics": {"subjects_mentioned": 1}}
    run = {
        "arm": "dream",
        "seed": 42,
        "settings": {"split_manifest_sha256": "x", "epochs": 1},
        "curve": [
            {"epoch": 0, **point},
            {"epoch": 1, **point, "treatment": {"cycles": [acceptance]}},
        ],
    }

    report = aggregate_runs([run])

    assert report["cycle_curves"] == {}
    assert "peak_cycle" not in report["arms"]["dream"]


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


def test_reply_temperature_is_a_dream_setting_not_a_cross_arm_one():
    settings = {
        "epochs": 0,
        "engineering_only": False,
        "eval_batch_size": 16,
        "requested_dream_batch_size": 10,
        "split_manifest_sha256": "x",
    }
    runs = [
        {
            "arm": "frozen",
            "seed": 42,
            "settings": settings | {"reply_temperature": None},
            "curve": [{"epoch": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0}],
        },
        {
            "arm": "dream",
            "seed": 42,
            "settings": settings | {"reply_temperature": 0.7},
            "curve": [{"epoch": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0}],
        },
    ]

    assert set(aggregate_runs(runs)["arms"]) == {"frozen", "dream"}


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
            "arm": "dream",
            "seed": 42,
            "settings": settings,
            "curve": [{"epoch": 0, **point}, {"epoch": 1, "wake_tokens": 30, **point}],
        },
    ]

    report = aggregate_runs(runs)

    assert report["arms"]["frozen"]["final_forgetting"]["mean"] == 0.0
    assert report["arms"]["dream"]["wake_tokens"]["mean"] == 30


def _cell(arm: str, seed: int) -> dict[str, object]:
    point = {"to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0}
    return {
        "arm": arm,
        "seed": seed,
        "settings": {"split_manifest_sha256": "x", "epochs": 0},
        "curve": [{"epoch": 0, **point}],
    }


def test_build_report_aggregates_the_seed_42_cells_and_lists_the_rest_as_missing():
    arms = ("frozen", "lora", "mix-review", "dream")

    report = build_report([_cell(arm, 42) for arm in arms])

    assert set(report["arms"]) == set(arms)
    assert sorted(map(tuple, report["present_cells"])) == sorted((arm, 42) for arm in arms)
    assert sorted(map(tuple, report["missing_cells"])) == sorted(
        (arm, seed) for arm in arms for seed in (43, 44)
    )


def test_build_report_accepts_a_two_arm_subset():
    report = build_report([_cell("frozen", 42), _cell("lora", 42)])

    assert set(report["arms"]) == {"frozen", "lora"}
    assert len(report["missing_cells"]) == 10


def test_build_report_refuses_an_empty_set():
    with pytest.raises(ValueError):
        build_report([])
