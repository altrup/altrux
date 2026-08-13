"""Characterization tests for the dream runner extraction boundary."""

import importlib


def test_runner_helpers_have_a_domain_owner_and_legacy_exports():
    runner = importlib.import_module("experiments.dreams.runner")
    legacy = importlib.import_module("dream_sleep")

    names = (
        "build_cache",
        "build_cues",
        "build_dream_set",
        "cl_summary",
        "commit_erase",
        "generate_wave_dream",
        "load_live_wake_scenarios",
        "r_matrix_row",
        "r_matrix_rows",
        "render_live_wake_transcript",
        "run_dream_set_sleep",
        "run_sleep",
        "validate_live_wake_args",
        "validate_wave_args",
    )
    assert all(getattr(legacy, name) is getattr(runner, name) for name in names)

