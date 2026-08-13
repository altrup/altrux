"""Characterization tests for the dream probe extraction boundary."""

import importlib


def test_dream_probe_helpers_have_a_domain_owner_and_legacy_exports():
    probes = importlib.import_module("experiments.dreams.probes")
    legacy = importlib.import_module("dream_sleep")

    names = (
        "basis_overlap",
        "battery_read_queries",
        "blank_state_logits",
        "dream_is_degenerate",
        "longest_verbatim_run",
        "probe_leakage",
        "report_dream",
        "report_dream_set",
    )
    assert all(getattr(legacy, name) is getattr(probes, name) for name in names)

