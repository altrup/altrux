"""Experiment: shared

The gate pilot (DISCUSSION-20260808 sec 2.10.6, 2.10.7): capture everything
once, then score every gating scheme offline.

`experiments/dreams/cli.py --build-dream-cache --dreams N --pilot-capture` writes the
capture this module reads. What is stored, and why:

  * **every position's per-layer read query**, fp16 -- not just the gated ones,
    since the point of the pilot is that a threshold sweep needs the positions
    a given threshold would have dropped.
  * **D_t at every position**, float -- the state-dependency divergence
    (`experiments.erasure.gating.state_divergence`), computed on the box from
    the with-state and blank-state logits.
  * the wake state, the facts, the dream texts, and the battery items' read
    queries (a forward pass, so it happens on the box).

Deliberately NOT stored: the (T, V) with-state and blank-state logit pairs
themselves. At the run's shape they are ~100 MB per dream against ~6 MB of
queries, and every scheme sec 2.10.7 names -- hard@any-tau, divergence-weighted
with a floor, weighted-capped -- is a function of D_t, which is stored in full.
The accepted limit: trying a DIFFERENT divergence measure needs a fresh capture
run.

Scoring (`python experiments/erasure/pilot.py <capture>`) runs sec 2.10.7's two tests:

  * **test 1, separability** -- D_t at the binding scan's fact positions against
    every other position, AUC per dream and pooled. Below `--min-auc` the gate
    CONCEPT has failed: the tool says so and exits nonzero rather than scoring
    schemes on a signal that is not there.
  * **test 2, the bake-off** -- {hard@tau, divergence-weighted, sqrt-capped,
    clip-capped} x a tau grid taken from the capture's own D_t quantiles. Each
    scheme builds every dream's per-layer V through the SAME primitives prod
    uses (`experiments.erasure.gating` + `experiments.dreams.cache.dream_bases`), and is
    scored on sec 2.10.6's DECISION METRIC: target removal (readout removed
    along oracle fact-read queries) against collateral removal (along context
    reads and battery-item queries), measured by applying V to the actual wake
    state.

Label precision/recall and the oracle-subspace overlap are printed as
diagnostics only -- sec 2.10.6 names label accuracy a NON-goal, and mini-training
outcomes are not scored here at all. The tool RECOMMENDS by sec 2.10.6's
lexicographic procedure; the freeze itself is a team decision.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))



@dataclass
class PilotDream:
    """One dream, unfiltered: the gate has not run on any of this."""

    dream_sha: str
    token_texts: list[str]
    divergence: list[float]
    queries: list[list[torch.Tensor]]  # T x n_layers, fp16
    cue_flags: list[bool]
    prefix_len: int
    stop_reason: str


@dataclass
class PilotCapture:
    seed: int
    facts: list[tuple[str, str, str]]  # entity, category, code
    wake_state: object
    dreams: list[PilotDream]
    # Battery prompt -> that prompt's per-position, per-layer read queries, run
    # from a copy of the wake state: sec 2.10.6's collateral pool.
    battery_queries: dict[str, list[list[torch.Tensor]]] = field(default_factory=dict)
    gate_threshold: float = 0.0
    rank_rule: str = ""
    set_sha: str = ""


# The tau grid: quantiles of the capture's OWN pooled D_t, so the sweep is
# scale-free and lands where the distribution actually is.
QUANTILES = (0.0, 0.5, 0.75, 0.9, 0.95, 0.99)
FAMILIES = ("hard", "weighted", "sqrt", "clip", "power2", "power3", "expmed")
# Below this the fact reads do not separate from context reads and sec 2.10.7's
# kill-condition fires.
MIN_AUC = 0.6
# The clip family's cap, as a quantile of the kept positions' own D_t.
CLIP_QUANTILE = 0.9


def auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """P(a random positive scores above a random negative), ties at half --
    the Mann-Whitney form, which needs no binning."""
    if not positive or not negative:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def eligible_positions(dream: PilotDream) -> list[int]:
    """Everything a gate could ever keep: the prefix conditions through state
    only and spliced cue text is not a read the dream performed."""
    return [t for t in range(len(dream.token_texts))
            if t >= dream.prefix_len and not (t < len(dream.cue_flags) and dream.cue_flags[t])]


def oracle_positions(dream: PilotDream, facts) -> tuple[dict[str, list[int]], list[int]]:
    """The binding scan's fact-read positions -- the VALIDATION overlay
    (sec 2.9.1), which is allowed on this side of the pilot and nowhere in the
    mechanism path."""
    from experiments.dreams.cache import fact_read_positions

    reads = fact_read_positions(dream.token_texts, facts)
    return reads, sorted({t for positions in reads.values() for t in positions})


def separability(capture: PilotCapture) -> dict[str, object]:
    """Sec 2.10.7 test 1: does D_t tell a fact read from everything else?"""
    from experiments.facts import Fact
    from progress import ts

    facts = [Fact(*f) for f in capture.facts]
    per_dream, pooled_pos, pooled_neg = [], [], []
    for i, dream in enumerate(capture.dreams):
        _, fact_pos = oracle_positions(dream, facts)
        keep = eligible_positions(dream)
        positive = [dream.divergence[t] for t in keep if t in set(fact_pos)]
        negative = [dream.divergence[t] for t in keep if t not in set(fact_pos)]
        pooled_pos += positive
        pooled_neg += negative
        per_dream.append(auc(positive, negative))
        print(f"[{ts()}]  dream {i}: {len(positive)} fact-read positions vs {len(negative)} others, "
              f"AUC {per_dream[-1]:.3f}", flush=True)
    result = {"per_dream": per_dream, "pooled": auc(pooled_pos, pooled_neg),
              "fact_positions": len(pooled_pos), "other_positions": len(pooled_neg)}
    print(f"[{ts()}] pooled AUC {result['pooled']:.3f} over {len(pooled_pos)} fact-read and "
          f"{len(pooled_neg)} other positions")
    return result


def scheme_weights(divergence: Sequence[float], family: str) -> list[float]:
    """A scheme's weight per KEPT position. The tau that selected them is the
    divergence-weighted family's floor (sec 2.10.7) -- one constant, not two."""
    d = [max(0.0, float(x)) for x in divergence]
    if family == "hard":
        return [1.0] * len(d)
    if family == "weighted":
        return d
    if family == "sqrt":
        return [math.sqrt(x) for x in d]
    if family == "clip":
        cap = quantile(d, CLIP_QUANTILE)
        return [min(x, cap) for x in d]
    # Super-linear: every other soft family is proportional (weighted) or
    # flattening (sqrt, clip), so none of them concentrates the basis on the
    # tail where fact reads sit. These do, without a binary cut.
    if family == "power2":
        return [x * x for x in d]
    if family == "power3":
        return [x * x * x for x in d]
    if family == "expmed":
        # exp(D / median D): the exponent is dimensionless, so unlike a bare
        # exp(D) -- which is exp(D/T) with T = 1 nat silently assumed -- the
        # weighting's shape does not move when the divergences are rescaled,
        # and its steepness adapts to each capture's own spread.
        positive = [x for x in d if x > 0]
        scale = statistics.median(positive) if positive else 0.0
        if scale <= 0:
            return [1.0] * len(d)
        # Subtracting the max is a global factor on every weight, so the SVD's
        # subspace is unchanged; without it a capture whose max/median is ~750
        # overflows outright.
        top = max(d)
        return [math.exp((x - top) / scale) for x in d]
    raise ValueError(f"unknown scheme family {family!r}; expected one of {FAMILIES}")


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0


def readout_removals(ssm_state, basis, queries: Sequence[torch.Tensor]) -> list[float]:
    """Fraction of the state's readout along each query that the eraser removes
    -- sec 2.10.6's proxy, measured on the actual wake state."""
    import torch

    from experiments.erasure.operators import erase_subspace

    if not queries:
        return []
    # One batched pass over all queries, on the accelerator when there is one:
    # the per-query loop ran two einsums over the whole state each time, which
    # at 64 layers is most of a sweep's runtime.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = ssm_state.float().to(device)
    erased = erase_subspace(state, basis.to(device))
    rows = torch.stack([c.reshape(-1).float() for c in queries]).to(device)
    before = torch.einsum("bhpn,qn->qbhp", state, rows).flatten(1).norm(dim=1)
    after = torch.einsum("bhpn,qn->qbhp", erased, rows).flatten(1).norm(dim=1)
    kept = before > 1e-9
    return (1.0 - after[kept] / before[kept]).clamp(min=0.0).cpu().tolist()


def _basis_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float((a.float() @ b.float().T).pow(2).sum() / min(a.shape[0], b.shape[0]))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_scheme(capture: PilotCapture, name: str, tau: float, family: str,
                 variant: str, rank_rule: str) -> dict[str, object]:
    """One row of sec 2.10.7's table: build every dream's per-layer V under this
    scheme, through the same primitives prod uses, and score the plane."""
    import torch

    from experiments.dreams.cache import dream_bases, gate_agreement
    from experiments.erasure.gating import RANK_RULES, gated_positions
    from experiments.facts import Fact

    facts = [Fact(*f) for f in capture.facts]
    battery = [per_layer for positions in capture.battery_queries.values() for per_layer in positions]
    target: list[float] = []
    context: list[float] = []
    batt: list[float] = []
    overlaps: list[float] = []
    ranks: dict[str, list[int]] = {rule: [] for rule in RANK_RULES}
    precision: list[float] = []
    recall: list[float] = []
    gated: list[int] = []
    per_dream_bases: list[list[torch.Tensor]] = []
    with_reads = 0
    for dream in capture.dreams:
        gate = gated_positions(torch.tensor(dream.divergence), tau, dream.prefix_len, dream.cue_flags)
        # An empty gate is an empty eraser, not a scheme failure -- the dream
        # contributes nothing at this tau (mirrors the builder's semantics).
        if not gate:
            continue
        # A dream without fact reads is sparse coverage, not a scheme failure
        # (15 of 20 dreams in the first real capture): it still contributes
        # collateral and stability; target and oracle pool over the rest.
        reads, fact_pos = oracle_positions(dream, facts)
        weights = scheme_weights([dream.divergence[t] for t in gate], family)
        try:
            _, chosen, bases = dream_bases(dream.queries, gate, capture.wake_state, rank_rule, weights)
            oracle = (dream_bases(dream.queries, fact_pos, capture.wake_state, rank_rule)[2]
                      if fact_pos else None)
        except SystemExit as failure:  # an empty or non-orthonormal basis is a scheme failure here
            return {"scheme": name, "family": family, "tau": tau, "ok": False,
                    "why": str(failure).split("--")[0].strip(),
                    "target_removed": 0.0, "collateral_removed": 0.0}
        basis = bases[variant]
        per_dream_bases.append(basis)
        with_reads += bool(fact_pos)
        others = [t for t in eligible_positions(dream) if t not in set(fact_pos)]
        for layer, v in enumerate(basis):
            state = capture.wake_state.ssm_states[layer]
            context += readout_removals(state, v, [dream.queries[t][layer] for t in others])
            batt += readout_removals(state, v, [per_layer[layer] for per_layer in battery])
            if fact_pos:
                target += readout_removals(state, v, [dream.queries[t][layer] for t in fact_pos])
                overlaps.append(_basis_overlap(v, oracle[variant][layer]))
        for rule in RANK_RULES:
            ranks[rule] += chosen[rule]
        if fact_pos:
            agreement = gate_agreement(gate, reads)
            precision.append(float(agreement["precision"]))
            recall.append(float(agreement["recall"]))
        gated.append(len(gate))
    if not per_dream_bases:
        return {"scheme": name, "family": family, "tau": tau, "ok": False,
                "why": "no gated positions at this tau anywhere in the capture",
                "target_removed": 0.0, "collateral_removed": 0.0}
    if not target:
        return {"scheme": name, "family": family, "tau": tau, "ok": False,
                "why": "no fact read anywhere in the capture -- the plane's target axis is unmeasurable",
                "target_removed": 0.0, "collateral_removed": 0.0}
    stability = [_basis_overlap(a[layer], b[layer])
                 for a, b in zip(per_dream_bases, per_dream_bases[1:], strict=False)
                 for layer in range(len(a))]
    return {
        "scheme": name, "family": family, "tau": tau, "ok": True, "why": "",
        "gated_positions": _mean(gated), "precision": _mean(precision), "recall": _mean(recall),
        "oracle_overlap": _mean(overlaps), "stability": _mean(stability),
        "dreams_scored": len(per_dream_bases), "dreams_with_reads": with_reads,
        "target_removed": _mean(target) or 0.0,
        "collateral_removed": _mean(context + batt) or 0.0,
        "context_removed": _mean(context), "battery_removed": _mean(batt),
        "ranks": {rule: [min(rs), max(rs)] for rule, rs in ranks.items()},
        "rank_rules_disagree": ranks[RANK_RULES[0]] != ranks[RANK_RULES[1]],
    }


def recommend(rows: Sequence[dict[str, object]]) -> dict[str, object] | None:
    """Sec 2.10.6's lexicographic procedure: (i) discard the mechanically
    unsound, (ii) a dominating scheme wins, (iii) else the highest target
    removal among schemes below the knee of this pilot's own collateral
    distribution -- taken as its median, a scale-free split of what was actually
    observed -- (iv) ties break toward simplicity, hard before weighted."""
    live = [r for r in rows if r["ok"]]
    if not live:
        return None
    # Step (ii) is a scheme that dominates EVERY other, not one that merely
    # nothing beats: a nonempty Pareto front always exists, so the weaker
    # reading would answer from iteration order and step (iii) would be dead.
    for row in live:
        if all(o["target_removed"] <= row["target_removed"]
               and o["collateral_removed"] >= row["collateral_removed"]
               for o in live if o is not row):
            return row
    knee = statistics.median(float(r["collateral_removed"]) for r in live)
    under = [r for r in live if float(r["collateral_removed"]) <= knee] or live
    return min(under, key=lambda r: (-round(float(r["target_removed"]), 3),
                                     FAMILIES.index(str(r["family"])),
                                     float(r["collateral_removed"])))


HEADER = (f"{'scheme':18} {'AUC':>6} {'prec':>5} {'rec':>5} {'oracle':>6} {'stab':>5} "
          f"{'target':>7} {'collat':>7} {'ctx':>6} {'batt':>6} {'gated':>6} {'rank':>10}")


def format_row(row: dict[str, object], pooled_auc: float) -> str:
    if not row["ok"]:
        return f"{str(row['scheme']):18} FAILED: {row['why']}"
    fmt = lambda v: "   n/a" if v is None else f"{float(v):6.3f}"  # noqa: E731
    ranks = row["ranks"]
    span = f"{ranks['ratio-gap'][0]}-{ranks['ratio-gap'][1]}"
    if row["rank_rules_disagree"]:
        span += f"/{ranks['median'][0]}-{ranks['median'][1]}"
    return (f"{str(row['scheme']):18} {pooled_auc:6.3f} {fmt(row['precision'])[1:]} "
            f"{fmt(row['recall'])[1:]} {fmt(row['oracle_overlap'])} {fmt(row['stability'])[1:]} "
            f"{float(row['target_removed']):7.3f} {float(row['collateral_removed']):7.3f} "
            f"{fmt(row['context_removed'])} {fmt(row['battery_removed'])} "
            f"{float(row['gated_positions']):6.1f} {span:>10}")


def main(capture_path: str, variant: str = "raw", rank_rule: str = "ratio-gap",
         min_auc: float = MIN_AUC, out: str | None = None,
         families: Sequence[str] = FAMILIES) -> None:
    import torch

    from progress import ts

    path = Path(capture_path)
    # The capture holds a MixerState, not just tensors -- this repo's own
    # artifact, written by the cache builder, exactly as load_dream_cache does.
    capture = torch.load(path, map_location="cpu", weights_only=False)
    stem = out or str(path.with_name(path.name.replace(".pilot.pt", "") + ".gate_pilot"))
    table_path, rows_path = Path(stem + ".txt"), Path(stem + ".jsonl")
    print(f"[{ts()}] === gate pilot: {path} -- {len(capture.dreams)} dreams, set_sha "
          f"{capture.set_sha[:12]}, variant {variant}, rank rule {rank_rule} ===")

    print(f"\n[{ts()}] --- test 1: does D_t separate fact reads from everything else? ---")
    sep = separability(capture)
    if not (sep["pooled"] >= min_auc):
        raise SystemExit(
            f"\nKILL CONDITION (sec 2.10.7): pooled AUC {sep['pooled']:.3f} < {min_auc} -- fact reads "
            f"do not separate from context reads on state divergence, so the gate CONCEPT fails. "
            f"STOP and rethink the gate before any harness is built on it; do not tune a threshold "
            f"against this capture."
        )

    print(f"\n[{ts()}] --- test 2: the bake-off (sec 2.10.6's plane; rows stream as they finish) ---")
    pooled = [d for dream in capture.dreams for t, d in enumerate(dream.divergence)
              if t in set(eligible_positions(dream))]
    print(HEADER, flush=True)
    rows = []
    for q in QUANTILES:
        tau = quantile(pooled, q)
        for family in families:
            row = score_scheme(capture, f"{family}@q{int(q * 100)}", tau, family, variant, rank_rule)
            row["quantile"] = q
            rows.append(row)
            print(f"[{ts()}] " + format_row(row, float(sep["pooled"])), flush=True)

    choice = recommend(rows)
    lines = [
        f"gate pilot: {path}  ({len(capture.dreams)} dreams, set_sha {capture.set_sha}, "
        f"variant {variant}, rank rule {rank_rule})",
        f"test 1 separability: pooled AUC {sep['pooled']:.3f} over {sep['fact_positions']} fact-read "
        f"and {sep['other_positions']} other positions; per dream "
        + ", ".join(f"{a:.3f}" for a in sep["per_dream"]),
        "",
        "test 2 bake-off. THE DECISION METRIC is target vs collateral removal (sec 2.10.6);",
        "precision/recall and oracle overlap are diagnostics -- label accuracy is a NON-goal,",
        "and no scheme here is scored by any downstream training outcome.",
        "rank column is the ratio-gap rule's per-layer span, /median where the rules disagree.",
        "",
        HEADER,
        *(format_row(r, float(sep["pooled"])) for r in rows),
        "",
        f"The tool RECOMMENDS {choice['scheme'] if choice else 'nothing -- every scheme failed'}"
        + (f" (target {float(choice['target_removed']):.3f}, collateral "
           f"{float(choice['collateral_removed']):.3f})" if choice else "")
        + " by sec 2.10.6's lexicographic procedure.",
        "The freeze is a HUMAN decision: the experimenter proposes, the team ratifies at a",
        "check-in, and only then is it frozen for the box.",
    ]
    table_path.write_text("\n".join(lines) + "\n")
    with rows_path.open("w") as handle:
        handle.write(json.dumps({"separability": sep, "capture": str(path),
                                 "variant": variant, "rank_rule": rank_rule}) + "\n")
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print("\n".join(lines[-3:]))
    print(f"\n[{ts()}] wrote {table_path} and {rows_path}")



def cli_main() -> None:
    from experiments.erasure.gating import RANK_RULES, VARIANTS

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("capture", help="a --pilot-capture artifact (data/dream_set_s<seed>.pilot.pt)")
    parser.add_argument("--variant", choices=VARIANTS, default="raw",
                        help="Which post-processing of the shared SVD each scheme's basis uses "
                             "(default: %(default)s)")
    parser.add_argument("--rank-rule", choices=RANK_RULES, default="ratio-gap",
                        help="Which rank rule truncates; both are reported (default: %(default)s)")
    parser.add_argument("--min-auc", type=float, default=MIN_AUC,
                        help="Sec 2.10.7's kill condition on test 1 (default: %(default)s)")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES),
                        help="Weighting families to score (default: all). A subset lets an "
                             "expensive sweep be split across processes.")
    parser.add_argument("--out", default=None,
                        help="Path stem for the table; .txt and .jsonl are appended (default: beside the capture)")
    args = parser.parse_args()
    main(args.capture, args.variant, args.rank_rule, args.min_auc, args.out, args.families)


if __name__ == "__main__":
    cli_main()


__all__ = ["PilotDream","PilotCapture","QUANTILES","FAMILIES","MIN_AUC","CLIP_QUANTILE","auc","eligible_positions","oracle_positions","separability","scheme_weights","quantile","readout_removals","_mean","score_scheme","recommend","HEADER","format_row","main"]
