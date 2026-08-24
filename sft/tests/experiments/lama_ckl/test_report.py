import pytest

from experiments.lama_ckl.report import aggregate_runs


def test_aggregate_runs_reports_acquisition_and_forgetting_by_arm():
    runs = [
        {"arm": "frozen", "seed": 42,
         "settings": {"engineering_only": False, "split_manifest_sha256": "x"},
         "curve": [
             {"cycle": 0, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
             {"cycle": 1, "to_learn_accuracy": 0.0, "not_to_forget_accuracy": 1.0},
         ]},
        {"arm": "altrux", "seed": 42,
         "settings": {"engineering_only": False, "split_manifest_sha256": "x"},
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
        aggregate_runs([{**run, "settings": {"engineering_only": True,
                                               "split_manifest_sha256": "x"}}])
    with pytest.raises(ValueError, match="split"):
        aggregate_runs([
            {**run, "settings": {"engineering_only": False, "split_manifest_sha256": "x"}},
            {**run, "seed": 43,
             "settings": {"engineering_only": False, "split_manifest_sha256": "y"}},
        ])
