"""Compatibility tests for the shared erasure tensor operators."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[3]))


def test_legacy_modules_reexport_the_shared_erasure_operators():
    import b4
    import erase_probe
    from experiments.erasure import operators

    for module, names in (
        (b4, ("deflate", "state_top_dirs", "erase_subspace", "sigma_gammas", "erase_subspace_scaled")),
        (erase_probe, ("rank1_erase", "deflate", "state_top_dirs", "clear_state_top_dirs_cache")),
    ):
        for name in names:
            assert getattr(module, name) is getattr(operators, name)


def test_b4_reexports_the_shared_gating_and_basis_primitives():
    import b4
    from experiments.erasure import gating

    for name in (
        "ADDRESS_BUDGET_FRACTION",
        "MEDIAN_C",
        "RANK_RULES",
        "VARIANTS",
        "DEFLATE_K",
        "state_divergence",
        "gated_positions",
        "address_budget",
        "rank_ratio_gap",
        "rank_median",
        "aggregate_basis",
        "orthonormalize",
        "variant_basis",
    ):
        assert getattr(b4, name) is getattr(gating, name)
