import pytest

from experiments.lama_ckl.report import aggregate_runs, scientific_runs


def test_aggregate_runs_reports_acquisition_and_forgetting_by_arm():
    settings = {
        "cycles": 1,
        "engineering_only": False,
        "eval_batch_size": 16,
        "requested_dream_batch_size": 8,
        "split_manifest_sha256": "x",
    }
    runs = [
        {"arm": "frozen", "seed": 42,
         "settings": settings,
         "curve": [
             {"cycle": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
             {"cycle": 1, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
         ]},
        {"arm": "altrux", "seed": 42,
         "settings": settings,
         "curve": [
             {"cycle": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
             {"cycle": 1, "to_learn_accuracy": 0.2, "not_to_forget_accuracy": 0.9},
         ]},
    ]

    report = aggregate_runs(runs)

    assert report["arms"]["altrux"]["top_accuracy"]["mean"] == 0.2
    assert report["arms"]["altrux"]["final_forgetting"]["mean"] == pytest.approx(0.1)
    assert report["arms"]["frozen"]["final_forgetting"]["mean"] == 0.0


def test_aggregate_runs_rejects_smokes_and_mixed_splits():
    run = {"arm": "frozen", "seed": 42, "curve": [
        {"cycle": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
    ]}

    with pytest.raises(ValueError, match="engineering-only"):
        aggregate_runs([{**run, "settings": {"cycles": 0, "engineering_only": True,
                                               "eval_batch_size": 16,
                                               "requested_dream_batch_size": 8,
                                               "split_manifest_sha256": "x"}}])
    with pytest.raises(ValueError, match="split"):
        aggregate_runs([
            {**run, "settings": {"cycles": 0, "engineering_only": False,
                                   "eval_batch_size": 16,
                                   "requested_dream_batch_size": 8,
                                   "split_manifest_sha256": "x"}},
            {**run, "seed": 43,
             "settings": {"cycles": 0, "engineering_only": False,
                           "eval_batch_size": 16,
                           "requested_dream_batch_size": 8,
                           "split_manifest_sha256": "y"}},
        ])


@pytest.mark.parametrize("cycles", [[0, 1], [0, 2]])
def test_aggregate_runs_rejects_incomplete_cycle_curves(cycles: list[int]):
    run = {
        "arm": "frozen",
        "seed": 42,
        "settings": {
            "cycles": 2,
            "engineering_only": False,
            "eval_batch_size": 16,
            "requested_dream_batch_size": 8,
            "split_manifest_sha256": "x",
        },
        "curve": [{"cycle": cycle} for cycle in cycles],
    }

    with pytest.raises(ValueError, match="cycle curve"):
        aggregate_runs([run])


@pytest.mark.parametrize("key", ["eval_batch_size", "requested_dream_batch_size"])
def test_aggregate_runs_rejects_mixed_batch_settings(key: str):
    settings = {
        "cycles": 0,
        "engineering_only": False,
        "eval_batch_size": 16,
        "requested_dream_batch_size": 8,
        "split_manifest_sha256": "x",
    }
    runs = [
        {"arm": "frozen", "seed": 42, "settings": settings, "curve": [{"cycle": 0}]},
        {"arm": "lora", "seed": 42, "settings": settings | {key: settings[key] * 2},
         "curve": [{"cycle": 0}]},
    ]

    with pytest.raises(ValueError, match=key):
        aggregate_runs(runs)


def test_scientific_runs_ignores_adjacent_smoke_directories():
    full = {"settings": {"engineering_only": False}}
    smoke = {"settings": {"engineering_only": True}}

    assert scientific_runs([smoke, full]) == [full]
