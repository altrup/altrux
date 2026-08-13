import importlib


def test_top_level_filter_wrapper_reexports_filtering_api():
    implementation = importlib.import_module("preparation.filtering")
    legacy = importlib.import_module("filter_items")

    assert legacy.__all__ == implementation.__all__
    for name in implementation.__all__:
        assert getattr(legacy, name) is getattr(implementation, name)
