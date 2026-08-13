import importlib


def test_top_level_babilong_wrapper_reexports_main():
    implementation = importlib.import_module("preparation.babilong")
    legacy = importlib.import_module("prepare_babilong")

    assert legacy.__all__ == ["main"]
    assert legacy.main is implementation.main
