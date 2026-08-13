import importlib


def test_top_level_eligibility_wrapper_reexports_main():
    implementation = importlib.import_module("preparation.eligibility")
    legacy = importlib.import_module("eligibility_check")

    assert legacy.__all__ == ["main"]
    assert legacy.main is implementation.main
