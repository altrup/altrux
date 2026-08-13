"""CPU tests for B4's mechanism (DISCUSSION-20260808 sec 2.7, 2.9.2, 2.10.3):
the state-dependency gate, the per-layer SVD aggregate with both rank rules,
the three variant bases, and the once-per-dream subspace projection.

The state here is h*p = 4 rows over an 8-dimensional address space, which is
the smallest shape where deflation and a rank cap are not degenerate.
"""

import pytest
import torch

from b4 import (
    ADDRESS_BUDGET_FRACTION,
    VARIANTS,
    address_budget,
    aggregate_basis,
    erase_state_subspace,
    erase_subspace,
    gated_positions,
    rank_median,
    rank_ratio_gap,
    state_divergence,
    variant_basis,
)


def _queries(m: int = 3, n: int = 8, seed: int = 0) -> list[torch.Tensor]:
    torch.manual_seed(seed)
    return [torch.randn(1, n) for _ in range(m)]


def _ssm(seed: int = 0, n: int = 8) -> torch.Tensor:
    """(b=1, h=2, p=2, n) -- h*p = 4 rows of address space."""
    torch.manual_seed(seed)
    return torch.randn(1, 2, 2, n)


def read(ssm_state: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhpn,bn->bhp", ssm_state.float(), c.float())


# ---- the state-dependency gate (sec 2.9.2) --------------------------------


def test_divergence_is_zero_where_the_state_changed_nothing():
    logits = torch.randn(5, 7)
    torch.testing.assert_close(state_divergence(logits, logits.clone()),
                               torch.zeros(5), atol=1e-6, rtol=0)


def test_divergence_is_larger_where_the_state_moved_the_prediction_further():
    blank = torch.zeros(2, 4)
    with_state = torch.tensor([[0.0, 0.5, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0]])

    d = state_divergence(with_state, blank)

    assert d[1] > d[0] > 0


def test_the_gate_keeps_only_positions_above_the_threshold():
    d = torch.tensor([0.1, 2.0, 0.4, 3.0])
    assert gated_positions(d, threshold=1.0) == [1, 3]


def test_the_gate_never_captures_a_prefix_or_a_cue_position():
    """Prefix tokens influence through state only (sec 2.10.8) and spliced cue
    text is not a read the dream performed -- neither may enter a basis."""
    d = torch.tensor([9.0, 9.0, 9.0, 9.0])

    assert gated_positions(d, 1.0, prefix_len=2) == [2, 3]
    assert gated_positions(d, 1.0, exclude=[False, False, True, False]) == [0, 1, 3]


def test_the_gate_uses_no_fact_knowledge():
    """Sec 2.9.1: the mechanism path sees a divergence vector and a threshold.
    Anything else would have to arrive through a parameter that does not exist."""
    assert set(gated_positions.__code__.co_varnames[:gated_positions.__code__.co_argcount]) == {
        "divergence", "threshold", "prefix_len", "exclude"}


# ---- rank rules and the address budget (sec 2.7) --------------------------


def test_the_address_budget_is_a_fraction_of_the_state_never_a_fact_count():
    assert address_budget(128) == 128 // ADDRESS_BUDGET_FRACTION
    assert address_budget(8) >= 1


def test_the_ratio_gap_rule_cuts_at_the_largest_drop():
    sigma = torch.tensor([10.0, 9.0, 8.0, 0.2, 0.1])
    assert rank_ratio_gap(sigma, budget=5) == 3


def test_the_median_rule_keeps_what_stands_out_of_the_noise_tail():
    # median is 0.1; c=4 keeps everything above 0.4.
    sigma = torch.tensor([10.0, 9.0, 0.1, 0.1, 0.1])
    assert rank_median(sigma, budget=5, c=4.0) == 2


@pytest.mark.parametrize("rule", [rank_ratio_gap, rank_median])
def test_both_rank_rules_respect_the_address_budget(rule):
    sigma = torch.tensor([10.0, 9.0, 8.0, 7.0, 0.1])
    assert rule(sigma, budget=2) <= 2


# ---- the per-layer SVD aggregate and its variants -------------------------


def test_the_aggregate_basis_is_orthonormal_and_ordered():
    v, sigma = aggregate_basis(_queries())

    torch.testing.assert_close(v @ v.T, torch.eye(v.shape[0]), atol=1e-5, rtol=0)
    assert list(sigma) == sorted(sigma, reverse=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_basis_is_orthonormal(variant):
    """Sec 2.7's harness assertion, in this module's row convention: the basis
    rows are orthonormal, so V V^T = I and the projection below is exact."""
    v, _ = aggregate_basis(_queries(m=4))

    basis = variant_basis(v, rank=2, variant=variant, ssm_state=_ssm())

    assert basis.shape[0] >= 1
    torch.testing.assert_close(basis @ basis.T, torch.eye(basis.shape[0]), atol=1e-5, rtol=0)


def test_the_qcm_variant_drops_the_query_consensus_direction():
    v, _ = aggregate_basis(_queries(m=4))
    qcm = variant_basis(v, rank=2, variant="qcm", ssm_state=_ssm())

    assert abs(float(qcm[0] @ v[0])) < 1e-5


def test_the_deflated_variant_leaves_the_state_s_top_direction_alone():
    """The faithful aggregate port of the operator that won the picker: the
    state's own top singular direction survives the erase."""
    ssm = _ssm()
    v, _ = aggregate_basis(_queries(m=3))
    basis = variant_basis(v, rank=3, variant="deflated", ssm_state=ssm)

    _, _, vh = torch.linalg.svd(ssm.reshape(1, -1, ssm.shape[-1]).float(), full_matrices=False)
    top = vh[0, :1]
    before, after = read(ssm, top), read(erase_subspace(ssm, basis), top)

    torch.testing.assert_close(after, before, atol=1e-4, rtol=1e-3)


# ---- the erase (sec 2.7 iii) ----------------------------------------------


def test_the_raw_eraser_zeroes_the_readout_along_every_captured_query():
    queries = _queries(m=3)
    ssm = _ssm()
    v, _ = aggregate_basis(queries)
    basis = variant_basis(v, rank=len(queries), variant="raw", ssm_state=ssm)

    erased = erase_subspace(ssm, basis)

    for c in queries:
        assert read(erased, c).abs().max().item() < 1e-4


def test_the_eraser_is_a_projection_so_applying_it_twice_is_applying_it_once():
    """Sec 2.7: idempotence is what makes the damage independent of how many
    dreams re-apply the eraser. A sigma-scaled cut would compound as (1-g)^N."""
    ssm = _ssm()
    v, _ = aggregate_basis(_queries(m=3))
    basis = variant_basis(v, rank=2, variant="raw", ssm_state=ssm)

    once = erase_subspace(ssm, basis)
    twice = erase_subspace(once, basis)

    torch.testing.assert_close(twice, once, atol=1e-5, rtol=0)


def test_erase_state_subspace_applies_each_layer_s_own_basis():
    class FakeState:
        def __init__(self, ssm_states):
            self.ssm_states = ssm_states

    state = FakeState([_ssm(seed=i) for i in range(3)])
    bases = [variant_basis(aggregate_basis(_queries(seed=i))[0], 2, "raw") for i in range(3)]

    erase_state_subspace(state, bases)

    for ssm, basis in zip(state.ssm_states, bases, strict=True):
        assert read(ssm, basis[:1]).abs().max().item() < 1e-4


def test_an_empty_basis_slice_stays_empty_rather_than_raising():
    """A single gated query leaves qcm nothing to keep -- the caller turns that
    into a stop-and-think failure, so the primitive must not blow up first."""
    v, _ = aggregate_basis(_queries(m=1))
    assert variant_basis(v, rank=1, variant="qcm", ssm_state=_ssm()).shape[0] == 0


def test_weighting_pulls_the_basis_toward_the_heavier_queries():
    """The pilot's divergence-weighted schemes (sec 2.10.7) enter here: a
    position's influence on the basis is its weight, and a zero weight is a
    position the basis never saw."""
    q = [torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]]), torch.tensor([[0.0, 1, 0, 0, 0, 0, 0, 0]])]

    v, sigma = aggregate_basis(q, weights=[1.0, 0.0])

    assert abs(abs(float(v[0, 0])) - 1.0) < 1e-5 and float(sigma[1]) < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to cross devices")
def test_variant_basis_deflates_cpu_rows_against_a_gpu_state():
    # Captured queries come off the cache on CPU while the wake state lives on
    # the GPU; the deflated variant must not assume one device (sec 1.8: the
    # all-CPU fake backbone can never catch this).
    v_full = torch.linalg.svd(torch.randn(6, 16), full_matrices=False).Vh
    state = torch.randn(1, 2, 4, 16, device="cuda")
    basis = variant_basis(v_full, 3, "deflated", state)
    assert basis.device.type == "cpu"
    assert torch.allclose(basis @ basis.T, torch.eye(len(basis)), atol=1e-5)


def test_sigma_scaled_erase_is_partial_and_ordered_by_singular_value():
    """DISCUSSION sec 5 rejected sigma-scaling on theory (partial cuts leave
    re-amplifiable residue, and (1-gamma)^N compounds across re-applications).
    Both objections are about REPEATED application; in single-sleep the eraser
    fires once per dream, so the question is empirical. This is the operator:
    each direction is removed in proportion to its own singular value, the
    strongest fully and the weakest barely.
    """
    import torch

    from b4 import erase_subspace, erase_subspace_scaled

    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(8, 2))[0].T.contiguous()
    state = torch.randn(1, 2, 3, 8)

    full = erase_subspace(state, basis)
    scaled = erase_subspace_scaled(state, basis, [1.0, 0.0])
    none = erase_subspace_scaled(state, basis, [0.0, 0.0])

    # gamma all-zero removes nothing; gamma all-one is the full projection
    assert torch.allclose(none, state, atol=1e-5)
    assert torch.allclose(erase_subspace_scaled(state, basis, [1.0, 1.0]), full, atol=1e-5)
    # a partial cut sits strictly between
    assert not torch.allclose(scaled, state, atol=1e-4)
    assert not torch.allclose(scaled, full, atol=1e-4)
    assert (scaled - state).norm() < (full - state).norm()


def test_sigma_gammas_come_from_the_spectrum_normalised_to_its_top():
    from b4 import sigma_gammas

    assert sigma_gammas([4.0, 2.0, 1.0], rank=3) == pytest.approx([1.0, 0.5, 0.25])
    assert sigma_gammas([4.0, 2.0, 1.0], rank=2) == pytest.approx([1.0, 0.5])
    assert sigma_gammas([], rank=2) == []
    assert sigma_gammas([0.0, 0.0], rank=2) == [0.0, 0.0]
