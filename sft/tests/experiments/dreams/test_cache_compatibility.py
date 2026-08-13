"""Characterization tests for the dream-cache extraction boundary."""

import importlib


def test_cache_functions_move_without_removing_legacy_exports():
    cache = importlib.import_module("experiments.dreams.cache")
    legacy = importlib.import_module("dream_sleep")

    names = (
        "default_cache_path",
        "dream_sidecar_text",
        "load_dream_cache",
        "merge_dream_sets",
        "pilot_path",
        "rebase_dream_set",
        "save_dream_cache",
        "sidecar_path",
        "write_dream_set_sidecar",
        "write_dream_sidecar",
    )
    assert all(getattr(legacy, name) is getattr(cache, name) for name in names)

