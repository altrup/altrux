"""Characterization tests for the dream distillation extraction boundary."""

import importlib


def test_distillation_helpers_have_a_domain_owner_and_legacy_exports():
    distillation = importlib.import_module("experiments.dreams.distillation")
    legacy = importlib.import_module("dream_sleep")

    names = (
        "distill_counterfactual",
        "distill_dream_set",
        "distill_fused",
        "distill_live",
        "distill_replay",
        "distill_sft",
        "erased_start",
        "erased_start_scaled",
        "fused_pass",
        "sft_steps",
        "spine_states",
    )
    assert all(getattr(legacy, name) is getattr(distillation, name) for name in names)

