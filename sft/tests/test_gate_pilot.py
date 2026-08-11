"""CPU tests for the offline gate-pilot scorer (DISCUSSION-20260808 sec 2.10.6,
2.10.7), on synthetic captures with a planted fact subspace.

The planted geometry is the point: fact reads query one direction of the state's
address space and everything else queries its orthogonal complement, so a gate
that finds the fact positions removes the fact readout and nothing else. A
scheme that gates everything cannot -- which is exactly the target-vs-collateral
plane sec 2.10.6 decides on.
"""

import pytest
import torch

from consolidation_null import Fact
from gate_pilot import (
    MIN_AUC,
    PilotCapture,
    PilotDream,
    auc,
    main,
    recommend,
    score_scheme,
    separability,
)

FACTS = [Fact("osprey", "bird", "5 9 7 9 7")]
N = 32
LAYERS = 2
# " 5 9 7 9 7" is the only token the binding scan will land on, twice per dream.
TEXTS = ["The", " code", " for", " the", " osprey", " is", " 5 9 7 9 7", ".",
         "The", " code", " for", " the", " osprey", " is", " 5 9 7 9 7", ".",
         " Then", " something", " else", " entirely", "."]
FACT_AT = (6, 14)


class FakeState:
    def __init__(self, ssm_states):
        self.ssm_states = ssm_states


def _dream(seed: int, separable: bool = True) -> PilotDream:
    """Fact positions read direction 0; every other position reads the
    complement. `separable=False` makes D_t carry no signal at all."""
    g = torch.Generator().manual_seed(seed)
    queries, divergence = [], []
    for t in range(len(TEXTS)):
        fact = t in FACT_AT
        if fact:
            c = torch.zeros(1, N)
            c[0, 0] = 1.0
        else:
            c = torch.zeros(1, N)
            c[0, 2:] = torch.randn(1, N - 2, generator=g)
            c = c / c.norm()
        queries.append([c.half() for _ in range(LAYERS)])
        # Non-separable: the state moved every prediction the same amount, so
        # D_t knows nothing about which position was a memory read.
        divergence.append((5.0 if fact else 0.1) if separable else 1.0)
    return PilotDream(dream_sha=f"sha{seed}", token_texts=list(TEXTS), divergence=divergence,
                      queries=queries, cue_flags=[False] * len(TEXTS), prefix_len=1,
                      stop_reason="eoc")


def _capture(separable: bool = True, battery: bool = True) -> PilotCapture:
    torch.manual_seed(0)
    state = FakeState([torch.randn(1, 2, 2, N) for _ in range(LAYERS)])
    batt = torch.zeros(1, N)
    batt[0, 1] = 1.0
    return PilotCapture(
        seed=1234, facts=[(f.entity, f.category, f.code) for f in FACTS], wake_state=state,
        dreams=[_dream(1, separable), _dream(2, separable)],
        battery_queries={"who wrote it?": [[batt.half() for _ in range(LAYERS)]]} if battery else {},
        gate_threshold=1.0, rank_rule="ratio-gap", set_sha="set",
    )


# ---- test 1: separability (sec 2.10.7) ------------------------------------


def test_auc_is_one_when_the_positive_scores_all_exceed_the_negatives():
    assert auc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert auc([0.0, 1.0], [2.0, 3.0]) == 0.0
    assert auc([1.0, 2.0], [1.0, 2.0]) == 0.5


def test_fact_reads_separate_from_everything_else_on_divergence():
    result = separability(_capture())

    assert result["pooled"] == 1.0
    assert len(result["per_dream"]) == 2 and all(a == 1.0 for a in result["per_dream"])


def test_the_kill_condition_fires_when_divergence_carries_no_signal(tmp_path):
    """Sec 2.10.7: no separation means the gate CONCEPT fails -- stop and
    rethink before any harness is built on it, not quietly score on."""
    path = tmp_path / "noise.pilot.pt"
    torch.save(_capture(separable=False), path)

    with pytest.raises(SystemExit) as raised:
        main(str(path))

    assert raised.value.code != 0
    assert "rethink" in str(raised.value).lower()


# ---- test 2: the bake-off's decision metric (sec 2.10.6) -------------------


def _score(tau: float, family: str = "hard") -> dict[str, object]:
    return score_scheme(_capture(), f"{family}@t{tau}", tau, family, "raw", "ratio-gap")


def test_a_gate_that_finds_the_fact_reads_removes_the_target_and_little_else():
    row = _score(tau=1.0)

    assert row["target_removed"] > 0.95
    assert row["collateral_removed"] < 0.2


def test_gating_every_position_costs_collateral_the_targeted_gate_does_not():
    """The decision metric, not the labels: both schemes are mechanically
    sane, and the plane is what separates them."""
    targeted, everything = _score(tau=1.0), _score(tau=0.0)

    assert targeted["target_removed"] > everything["target_removed"]
    assert targeted["collateral_removed"] < everything["collateral_removed"]


def test_a_scheme_reports_the_oracle_overlap_and_both_rank_rules():
    row = _score(tau=1.0)

    assert row["oracle_overlap"] > 0.9
    assert set(row["ranks"]) == {"ratio-gap", "median"}


def test_battery_queries_join_the_collateral_pool_when_the_capture_has_them():
    row = _score(tau=1.0)

    assert row["battery_removed"] is not None
    assert score_scheme(_capture(battery=False), "hard@t1.0", 1.0, "hard", "raw",
                        "ratio-gap")["battery_removed"] is None


def test_the_weighted_families_weight_the_queries_entering_the_svd():
    """A weighted scheme with the same gate as a hard one must not produce the
    same basis by accident -- the weights have to reach the SVD."""
    hard, weighted = _score(tau=0.0, family="hard"), _score(tau=0.0, family="weighted")

    assert weighted["target_removed"] > hard["target_removed"]


# ---- the deliverable (sec 2.10.7) -----------------------------------------


def _row(scheme, target, collateral, family="hard", ok=True):
    return {"scheme": scheme, "family": family, "target_removed": target,
            "collateral_removed": collateral, "ok": ok}


def test_a_pareto_front_with_no_dominator_falls_through_to_the_knee():
    """Sec 2.10.6(ii) is a scheme that dominates ALL the others, not merely one
    nothing beats -- a nonempty Pareto front always exists, so the weaker
    reading would answer from iteration order and never reach step (iii)."""
    front = [_row("mid", 0.80, 0.10), _row("greedy", 0.90, 0.20), _row("timid", 0.70, 0.05)]

    # knee = median collateral (0.10); the highest target under it is `mid`.
    assert recommend(front)["scheme"] == "mid"
    assert recommend(list(reversed(front)))["scheme"] == "mid"
    assert recommend([front[1], front[2], front[0]])["scheme"] == "mid"


def test_a_scheme_that_dominates_every_other_wins_at_step_two():
    front = [_row("mid", 0.80, 0.10), _row("greedy", 0.90, 0.20), _row("timid", 0.70, 0.05),
             _row("dominator", 0.95, 0.05)]

    assert recommend(front)["scheme"] == "dominator"


def test_the_recommendation_is_lexicographic_and_defers_the_freeze_to_the_team():
    rows = [
        {"scheme": "dominated", "target_removed": 0.5, "collateral_removed": 0.4, "ok": True},
        {"scheme": "dominating", "target_removed": 0.9, "collateral_removed": 0.1, "ok": True},
        {"scheme": "broken", "target_removed": 1.0, "collateral_removed": 0.0, "ok": False},
    ]

    assert recommend(rows)["scheme"] == "dominating"


def test_the_table_is_written_beside_the_capture_and_the_freeze_is_a_human_call(tmp_path, capsys):
    path = tmp_path / "dream_set_s1234.pilot.pt"
    torch.save(_capture(), path)

    main(str(path))

    table = (tmp_path / "dream_set_s1234.gate_pilot.txt").read_text()
    assert (tmp_path / "dream_set_s1234.gate_pilot.jsonl").exists()
    assert "hard@" in table and "weighted@" in table
    assert "RECOMMENDS" in table and "team" in table


def _factless(seed: int) -> PilotDream:
    """A dream that rehearses nothing -- the binding scan finds no fact read.
    15 of 20 dreams in the first real capture looked like this."""
    dream = _dream(seed)
    dream.token_texts = ["nothing "] * len(dream.token_texts)
    return dream


def test_a_factless_dream_does_not_fail_the_scheme(tmp_path):
    """Sparse coverage is the finding, not a scorer crash: dreams without
    fact reads still contribute collateral and stability; target and oracle
    pool over the dreams that have reads."""
    capture = _capture()
    capture.dreams.append(_factless(3))

    row = score_scheme(capture, "hard@q50", 0.5, "hard", "raw", "ratio-gap")

    assert row["ok"], row.get("why")
    assert row["dreams_with_reads"] == 2 and row["dreams_scored"] == 3


def test_a_capture_with_no_fact_reads_anywhere_fails_the_scheme():
    capture = _capture()
    capture.dreams = [_factless(1), _factless(2)]

    row = score_scheme(capture, "hard@q50", 0.5, "hard", "raw", "ratio-gap")

    assert not row["ok"] and "no fact read" in row["why"]


def test_readout_removals_batches_queries_without_changing_the_math():
    """The scoring path runs two einsums over the full ssm_state per query;
    at 64 layers that dominated the sweep's runtime. Batching over queries has
    to be arithmetically identical to the per-query loop it replaces."""
    import torch

    from b4 import erase_subspace
    from gate_pilot import readout_removals

    torch.manual_seed(0)
    state = torch.randn(1, 3, 4, 8)
    basis = torch.linalg.qr(torch.randn(8, 2))[0].T.contiguous()
    queries = [torch.randn(1, 8) for _ in range(5)]

    reference = []
    erased = erase_subspace(state.float(), basis)
    for c in queries:
        c = c.reshape(1, -1).float()
        before = torch.einsum("bhpn,bn->bhp", state.float(), c).norm()
        after = torch.einsum("bhpn,bn->bhp", erased, c).norm()
        if float(before) > 1e-9:
            reference.append(max(0.0, 1.0 - float(after / before)))

    got = readout_removals(state, basis, queries)

    assert len(got) == len(reference)
    for a, b in zip(got, reference, strict=True):
        assert abs(a - b) < 1e-5


def test_readout_removals_handles_an_empty_query_list():
    import torch

    from gate_pilot import readout_removals

    basis = torch.linalg.qr(torch.randn(8, 2))[0].T.contiguous()
    assert readout_removals(torch.randn(1, 3, 4, 8), basis, []) == []
