import importlib


def test_top_level_interference_wrapper_reexports_public_api():
    implementation = importlib.import_module("preparation.interference")
    legacy = importlib.import_module("prepare_interference")

    expected = ["LABEL_POOL", "LABEL_SKIP", "COLORS", "WEEKDAYS", "NAMES", "FACT_KINDS", "main"]
    assert legacy.__all__ == expected
    for name in expected:
        assert getattr(legacy, name) is getattr(implementation, name)
