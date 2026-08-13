"""Characterization tests for the dream-generation extraction boundary."""

import importlib


def test_generation_helpers_have_a_domain_owner_and_legacy_exports():
    generation = importlib.import_module("experiments.dreams.generation")
    legacy = importlib.import_module("dream_sleep")

    names = (
        "copy_state",
        "dream_generation_seed",
        "dream_seed_text",
        "frozen_teacher",
        "generate_replay_dreams",
        "rehearsal_fraction",
        "sample_next",
    )
    assert all(getattr(legacy, name) is getattr(generation, name) for name in names)
    assert callable(generation.teacher_dream)
    assert callable(legacy.teacher_dream)

