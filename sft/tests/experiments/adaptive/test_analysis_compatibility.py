import importlib


PUBLIC = ("rehearsal_retention_analysis", "aggregate_seed_results")


def test_adaptive_multisleep_reexports_analysis_helpers():
    legacy = importlib.import_module("adaptive_multisleep")
    domain = importlib.import_module("experiments.adaptive.analysis")

    assert tuple(domain.__all__) == PUBLIC
    for name in PUBLIC:
        assert getattr(legacy, name) is getattr(domain, name)
