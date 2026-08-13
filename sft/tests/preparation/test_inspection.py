import importlib


def test_top_level_sanity_wrapper_reexports_inspection_api():
    implementation = importlib.import_module("preparation.inspection")
    legacy = importlib.import_module("sanity_sample")

    assert legacy.__all__ == ["pick_windows", "main"]
    assert legacy.pick_windows is implementation.pick_windows
    assert legacy.main is implementation.main
