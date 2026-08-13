import importlib


def test_top_level_cram_wrapper_reexports_public_api():
    implementation = importlib.import_module("preparation.cram")
    legacy = importlib.import_module("prepare_cram")

    expected = [
        "add_block_args",
        "build_blocks",
        "emit",
        "find_subsequence",
        "load_wikipedia_passages",
        "make_items",
        "split_articles",
        "split_sentences",
        "validate_blocks",
        "_find",
        "main",
    ]
    assert legacy.__all__ == expected
    for name in expected:
        assert getattr(legacy, name) is getattr(implementation, name)
