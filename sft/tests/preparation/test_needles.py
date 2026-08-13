import importlib


def test_top_level_needles_wrapper_reexports_public_api():
    implementation = importlib.import_module("preparation.needles")
    legacy = importlib.import_module("prepare_needles")

    expected = [
        "add_block_args",
        "babilong_items",
        "build_blocks",
        "emit",
        "load_babilong",
        "load_wikipedia_passages",
        "split_articles",
        "main",
    ]
    assert legacy.__all__ == expected
    for name in expected:
        assert getattr(legacy, name) is getattr(implementation, name)
