#!/usr/bin/env python3
"""Experiment: state-erasure

Score the CL A/B grid: the install-vs-damage frontier per arm.

Installs are the distractor-code margin (DISCUSSION-20260806 sec 4), read at
the last probe point of each cell; greedy exact match is reported and never
gates. Rehearsal is binding-aware -- a code counts only beside its own entity,
and misbindings are their own column, because the substring coverage this
replaced counted "The code for the heron is <osprey's code>" as rehearsal.

Before anything is pooled, every cell of a seed must agree on the wave-1
transcript and dream hashes: one cached dream per seed, byte-identical across
arms, is a registered invariant and this is its machine check (sec 2).
"""

import argparse
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments.locality import MARGIN_INSTALL as INSTALL_NATS


def arm_and_seed(name: str) -> tuple[str, str]:
    """Split a cell's filename into its arm and its seed.

    The seed is the `_s<digits>` token, not "everything after the last _s":
    `lad_A_s1234_d3200` is arm `lad_A_d3200` at seed 1234, and reading its seed
    as `1234_d3200` matched no floor cell, which is why the ladder's Δ columns
    came out empty (and were computed by hand) in the 08-08 run. The rung stays
    in the arm so two rungs of one ladder never pool as one cell.
    """
    name = name[3:] if name.startswith("g2_") else name
    match = re.search(r"_s(\d+)(?=$|_)", name)
    if not match:
        arm, _, seed = name.rpartition("_s")
        return arm, seed
    return name[: match.start()] + name[match.end() :], match.group(1)


def aggregate_bound(dreams: list[dict[str, object]]) -> int:
    """Facts bound in at least one dream of the set (DISCUSSION-20260808
    sec 3): coverage is aggregate across dreams, and no dream is penalized for
    wandering off the facts."""
    bound: dict[str, int] = collections.defaultdict(int)
    for record in dreams:
        for entity, n in dict(record.get("bound_by_fact", {})).items():
            bound[entity] += int(n)
    return sum(n > 0 for n in bound.values())


def cell(path: str) -> dict[str, object] | None:
    rows = [json.loads(line) for line in open(path)]
    failed = [r for r in rows if r["phase"] == "equivalence" and not r.get("equivalent")]
    if failed:
        where = ", ".join(
            "sleep {} (abs_diff {:.3g})".format(r.get("wave"), r.get("abs_diff", 0.0))
            for r in failed
        )
        raise SystemExit(
            f"{path}: B3-fused's pass-1 equivalence to B2-fused-detached FAILED at {where} -- "
            f"the arm did not distil the spine it claims to. Nothing in this cell is scoreable."
        )
    ops = {r.get("erase_op") for r in rows}
    if len(ops) > 1:
        raise SystemExit(
            f"{path}: {len(ops)} distinct erase_op values ({', '.join(sorted(str(o) for o in ops))}) in one "
            f"cell -- the operator is a cell-level variable; rerun the cell under a single --erase-op."
        )
    loc = [r for r in rows if r["phase"] == "locality"]
    # A running cell already carries locality records from its periodic probes,
    # and its dream/sleep records are not written until sleep ends -- pooling one
    # reads as an arm that rehearsed nothing and took no gradients.
    if not loc or not any(r["phase"] == "done" for r in rows):
        return None
    dreams = [r for r in rows if r["phase"] == "dream"]
    dream = dreams[0] if dreams else None
    cache = next((r for r in rows if r["phase"] == "cache" and r.get("wave") == 1), {})
    sleep = next((r for r in rows if r["phase"] == "sleep"), {})
    ic = [r for r in rows if r["phase"] == "in_context"]
    probe = [r for r in rows if r["phase"] == "probe"]
    # Cells stream a full battery every --probe-every steps; the frontier is
    # read at the last point, the curve at all of them.
    last = max((r.get("step") or 0) for r in probe) if probe else 0
    final = [r for r in probe if (r.get("step") or 0) == last]
    steps: dict[tuple[int, int], dict[str, float]] = collections.defaultdict(dict)
    for r in probe:
        if r.get("margin") is not None:
            steps[(r.get("wave") or 1, r.get("step") or 0)][r["fact"]] = r["margin"]
    dppl_at = {(r.get("wave") or 1, r.get("step") or 0): r["ppl_delta"] for r in loc}
    arm, seed = arm_and_seed(os.path.basename(path).split(".jsonl")[0])
    return {
        "arm": arm,
        "seed": seed,
        "path": path,
        "erase_op": next(iter(ops), None),
        "steps": dict(steps),
        "dppl_at": dppl_at,
        "transcript_sha": cache.get("transcript_sha"),
        "dream_sha": cache.get("dream_sha"),
        "set_sha": cache.get("set_sha"),
        "dreams": cache.get("dreams"),
        "init_adapter": {r.get("init_adapter_sha256") for r in rows},
        "wave2_shas": [
            (r.get("wave"), r.get("dream_sha"))
            for r in rows
            if r["phase"] == "cache" and r.get("wave", 1) > 1
        ],
        "rehearse": dream.get("rehearsal_fraction") if dream else 0.0,
        "bound_cov": aggregate_bound(dreams)
        if cache.get("set_sha")
        else (dream.get("bound_cov", 0) if dream else 0),
        "misbound": (
            sum(sum(r.get("misbound_by_fact", {}).values()) for r in dreams)
            if cache.get("set_sha")
            else (dream.get("misbound", 0) if dream else 0)
        ),
        "free_tokens": cache.get("free_tokens", 0),
        "token_gradients": sleep.get("token_gradients", 0),
        "ic": sum(r["match"] for r in ic),
        "n": len(final),
        "margin_install": sum(bool(r.get("margin_install")) for r in final),
        "margin": sum(r.get("margin", 0.0) for r in final) / max(1, len(final)),
        "fact_margin": {r["fact"]: r.get("margin") for r in final if r.get("margin") is not None},
        "install": sum(r["match"] for r in final),
        "dlp": sum(r["logprob_delta"] for r in final if r.get("logprob_delta") is not None)
        / max(1, len(final)),
        "para": sum(r["paraphrase_rate"] for r in final) / max(1, len(final)),
        "lost": loc[-1]["lost"],
        "items": loc[-1]["items"],
        "dppl": loc[-1]["ppl_delta"],
    }


def load_cells(paths: list[str]) -> list[dict[str, object]]:
    return [c for c in (cell(p) for p in paths) if c]


def floor_deltas(cells, extract, fallback=None):
    """Baseline-correct every margin against the same fact's no-sleep margin.

    A raw margin is not a measure of learning. The distractor is one fixed
    random code, and nothing makes it as likely a priori as the real one: at
    the untrained floor the per-fact margins measured -1.6 to +4.1 nats, so the
    registered "installed iff margin >= 1.0" fires for two thirds of facts that
    were never trained on at all. The floor arm is per (seed, fact) constant, so
    subtracting it costs nothing and gives the metric a true zero.

    `extract` maps a cell to {group key: {fact: margin}} -- one group for the
    frontier's final margins, one per probe point for the curves. `fallback`
    supplies a per (seed, fact) floor for groups the no-sleep arm never reached.
    Returns the floor itself and one (cell, group key, deltas) row per group.
    """
    floor: dict[tuple[str, object, str], float] = {}
    for c in cells:
        if str(c["arm"]).replace("-", "_").split("_")[-1] == "nosleep":
            for key, facts in extract(c).items():
                for fact, m in facts.items():
                    floor[(str(c["seed"]), key, fact)] = m
    rows = []
    for c in cells:
        groups = extract(c)
        for key in sorted(groups):
            deltas = []
            for fact, m in groups[key].items():
                base = floor.get((str(c["seed"]), key, fact))
                if base is None and fallback is not None:
                    base = fallback.get((str(c["seed"]), fact))
                if base is not None:
                    deltas.append(m - base)
            rows.append((c, key, deltas))
    return floor, rows


def _final_margins(c: dict[str, object]) -> dict[str, dict[str, float]]:
    return {"": c["fact_margin"]}


def apply_floor(cells: list[dict[str, object]]) -> int:
    """Floor-correct each cell's final margins onto `dmargin`/`dinstall`/`dn`.

    Returns how many facts the floor itself would have scored installed --
    print it, because it is the evidence that the raw column cannot be read.
    """
    floor, rows = floor_deltas(cells, _final_margins)
    for c, _, deltas in rows:
        c["dmargin"] = sum(deltas) / len(deltas) if deltas else float("nan")
        c["dinstall"] = sum(d >= INSTALL_NATS for d in deltas)
        c["dn"] = len(deltas)
    return sum(m >= INSTALL_NATS for m in floor.values())


def curve_rows(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    """The probe curve of every cell: floor-corrected Δmargin and dPPL at each
    probe point (DISCUSSION-20260807 sec 3.4 rule 2 reads damage at matched
    Δmargin, which the endpoint alone cannot answer). The floor is the seed's
    no-sleep cell at the same probe step where it has one -- the no-sleep arm
    trains on nothing, so its final margin stands in everywhere else."""
    final_floor = {
        (seed, fact): m for (seed, _, fact), m in floor_deltas(cells, _final_margins)[0].items()
    }
    _, rows = floor_deltas(cells, lambda c: c["steps"], fallback=final_floor)
    return [
        {
            "arm": c["arm"],
            "seed": str(c["seed"]),
            "erase_op": c["erase_op"],
            "wave": key[0],
            "step": key[1],
            "n": len(deltas),
            "dmargin": sum(deltas) / len(deltas),
            "dinstall": sum(d >= INSTALL_NATS for d in deltas),
            "dppl": c["dppl_at"].get(key),
        }
        for c, key, deltas in rows
        if deltas
    ]


def print_curves(cells: list[dict[str, object]]) -> None:
    rows = curve_rows(cells)
    for seed in sorted({str(r["seed"]) for r in rows}):
        print(f"\nseed {seed} -- probe curves (Δmargin floor-corrected per step)")
        print(
            f"{'arm':22} {'erase':9} {'wave':>4} {'step':>6} {'dmarg':>7} {'dinst':>6} {'dPPL':>8}"
        )
        for r in sorted(rows, key=lambda r: (r["arm"], r["wave"], r["step"])):
            if str(r["seed"]) != seed:
                continue
            dppl = "     n/a" if r["dppl"] is None else f"{r['dppl']:+8.4f}"
            print(
                f"{r['arm']:22} {str(r['erase_op'] or '-'):9} {r['wave']:>4} {r['step']:>6} "
                f"{r['dmargin']:+7.2f} {r['dinstall']:>3}/{r['n']:<2} {dppl}"
            )


def check_hashes(cells: list[dict[str, object]]) -> None:
    """Every cell of a seed distilled the same wake transcript and the same
    dream, or nothing is pooled. Scoped to wave 1: in the multi-sleep grid a
    wave-2 dream legitimately differs per arm, so those are recorded, never
    asserted."""
    by_seed: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for c in cells:
        by_seed[str(c["seed"])].append(c)
    for seed, group in sorted(by_seed.items()):
        for field in ("transcript_sha", "dream_sha", "set_sha"):
            values = {str(c[field]) for c in group}
            if len(values) > 1:
                for c in group:
                    print(f"  {c['arm']:22} {field} {str(c[field])[:16]}  ({c['path']})")
                raise SystemExit(
                    f"seed {seed}: {len(values)} distinct wave-1 {field} across {len(group)} cells -- "
                    f"the arms did not share a dream, so nothing here is comparable. Rebuild the cache "
                    f"(--build-dream-cache) and rerun the seed."
                )


def check_init_adapter(cells: list[dict[str, object]]) -> str | None:
    """Every cell of the grid ran from the same warm-start adapter, or nothing
    is comparable (DISCUSSION-20260807 sec 3.1). All-absent is the
    pre-warm-start form (g2) and passes. Returns the grid's checkpoint hash."""
    shas = {s for c in cells for s in c["init_adapter"]}
    if len(shas) > 1:
        for c in cells:
            print(
                f"  {c['arm']:22} init_adapter {sorted(str(s)[:16] for s in c['init_adapter'])}  ({c['path']})"
            )
        raise SystemExit(
            f"{len(shas)} distinct warm starts across {len(cells)} cells -- the arms began from different "
            f"weights, so nothing here is comparable. Rerun the odd cells with the same --init-adapter."
        )
    return next(iter(shas), None)


def main(pattern: str, curves: bool = False) -> None:
    paths = sorted(glob.glob(pattern))
    cells = load_cells(paths)
    skipped = [p for p in paths if p not in {c["path"] for c in cells}]
    if skipped:
        print(
            f"skipping {len(skipped)} cell(s) with no done record (still running, or died): "
            f"{', '.join(os.path.basename(p) for p in skipped)}\n"
        )
    if not cells:
        raise SystemExit("no completed cells yet")
    check_hashes(cells)
    adapter = check_init_adapter(cells)
    seeds = sorted({str(c["seed"]) for c in cells})
    sets = {str(c["set_sha"])[:12] for c in cells if c["set_sha"]}
    print(
        f"{len(cells)} cells, seeds {', '.join(seeds)}; wave-1 dream hashes agree within every seed; "
        f"{f'warm start {adapter[:12]}' if adapter else 'no warm start'}"
        + (f"; dream set(s) {', '.join(sorted(sets))}" if sets else "")
    )
    false_positives = apply_floor(cells)
    floor_n = sum(
        c["dn"] for c in cells if str(c["arm"]).replace("-", "_").split("_")[-1] == "nosleep"
    )
    if floor_n:
        print(
            f"raw-margin floor: the untrained no-sleep arm clears the {INSTALL_NATS:.1f}-nat bar on "
            f"{false_positives}/{floor_n} facts -- read dmarg/dinst (floor-corrected), not marg/inst"
        )

    if curves:
        print_curves(cells)
        return

    print(
        f"\n{'arm':22} {'seed':5} {'erase':9} {'rehrs':>6} {'bound':>6} {'misb':>5} {'ic':>4} {'marg':>7} "
        f"{'inst':>5} {'dmarg':>7} {'dinst':>6} {'EM':>4} {'dlogp':>7} {'para':>5} {'tokgrad':>8} "
        f"{'lost':>6} {'dPPL':>8}"
    )
    for c in cells:
        # A dream set has no single-dream rehearsal fraction: its coverage is
        # aggregate across the set, which is what the bound column already says.
        rehearse = "     -" if c["rehearse"] is None else f"{c['rehearse']:6.3f}"
        print(
            f"{c['arm']:22} {c['seed']:5} {str(c['erase_op'] or '-'):9} {rehearse} {c['bound_cov']:>4}/4 {c['misbound']:>5} "
            f"{c['ic']:>2}/4 {c['margin']:+7.2f} {c['margin_install']:>2}/{c['n']:<2} "
            f"{c['dmargin']:+7.2f} {c['dinstall']:>3}/{c['dn']:<2} {c['install']:>2}/{c['n']:<1} "
            f"{c['dlp']:+7.3f} {c['para']:5.2f} {c['token_gradients']:>8} "
            f"{c['lost']:>2}/{c['items']:<3} {c['dppl']:+8.4f}"
        )

    print("\npooled by arm (all seeds):")
    pool: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for c in cells:
        pool[str(c["arm"])].append(c)
    print(
        f"{'arm':22} {'n':>2} {'install':>8} {'dmarg':>7} {'dinst':>8} {'EM':>7} {'dlogp':>7} "
        f"{'para':>5} {'tokgrad':>9} {'lost':>6} {'dPPL':>8}"
    )
    for arm, group in sorted(pool.items()):
        n = len(group)
        facts = sum(int(c["n"]) for c in group)
        dfacts = sum(int(c["dn"]) for c in group)
        print(
            f"{arm:22} {n:>2} {sum(c['margin_install'] for c in group):>3}/{facts:<4} "
            f"{sum(c['dmargin'] for c in group) / n:+7.2f} {sum(c['dinstall'] for c in group):>3}/{dfacts:<4} "
            f"{sum(c['install'] for c in group):>3}/{facts:<3} {sum(c['dlp'] for c in group) / n:+7.3f} "
            f"{sum(c['para'] for c in group) / n:5.2f} {sum(c['token_gradients'] for c in group):>9} "
            f"{sum(c['lost'] for c in group):>3}/{sum(c['items'] for c in group):<3} "
            f"{sum(c['dppl'] for c in group) / n:+8.4f}"
        )


def cli_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "pattern",
        nargs="?",
        default="logs/g2_*_s*.jsonl",
        help="glob of cell result jsonls (default: %(default)s)",
    )
    parser.add_argument(
        "--curves",
        action="store_true",
        help="one row per cell per probe step instead of the frontier tables",
    )
    args = parser.parse_args()
    main(args.pattern, args.curves)


if __name__ == "__main__":
    cli_main()
