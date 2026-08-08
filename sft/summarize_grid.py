#!/usr/bin/env python3
"""Score the CL A/B grid: the install-vs-damage frontier per arm.

Installs are the distractor-code margin (DISCUSSION-20260806 sec 4), read at
the last probe point of each cell; greedy exact match is reported and never
gates. Rehearsal is binding-aware -- a code counts only beside its own entity,
and misbindings are their own column, because the substring coverage this
replaced counted "The code for the heron is <osprey's code>" as rehearsal.

Before anything is pooled, every cell of a seed must agree on the wave-1
transcript and dream hashes: one cached dream per seed, byte-identical across
arms, is a registered invariant and this is its machine check (sec 2).
"""
import collections
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from probes_common import MARGIN_INSTALL as INSTALL_NATS


def cell(path: str) -> dict[str, object] | None:
    rows = [json.loads(line) for line in open(path)]
    loc = [r for r in rows if r["phase"] == "locality"]
    # A running cell already carries locality records from its periodic probes,
    # and its dream/sleep records are not written until sleep ends -- pooling one
    # reads as an arm that rehearsed nothing and took no gradients.
    if not loc or not any(r["phase"] == "done" for r in rows):
        return None
    dream = next((r for r in rows if r["phase"] == "dream"), None)
    cache = next((r for r in rows if r["phase"] == "cache" and r.get("wave") == 1), {})
    sleep = next((r for r in rows if r["phase"] == "sleep"), {})
    ic = [r for r in rows if r["phase"] == "in_context"]
    probe = [r for r in rows if r["phase"] == "probe"]
    # Cells stream a full battery every --probe-every steps; the frontier is
    # read at the last point, the curve at all of them.
    last = max((r.get("step") or 0) for r in probe) if probe else 0
    final = [r for r in probe if (r.get("step") or 0) == last]
    name = os.path.basename(path).split(".jsonl")[0]
    name = name[3:] if name.startswith("g2_") else name
    arm, _, seed = name.rpartition("_s")
    return {
        "arm": arm, "seed": seed, "path": path,
        "transcript_sha": cache.get("transcript_sha"), "dream_sha": cache.get("dream_sha"),
        "init_adapter": {r.get("init_adapter_sha256") for r in rows},
        "wave2_shas": [(r.get("wave"), r.get("dream_sha")) for r in rows
                       if r["phase"] == "cache" and r.get("wave", 1) > 1],
        "rehearse": dream["rehearsal_fraction"] if dream else 0.0,
        "bound_cov": dream.get("bound_cov", 0) if dream else 0,
        "misbound": dream.get("misbound", 0) if dream else 0,
        "free_tokens": cache.get("free_tokens", 0),
        "token_gradients": sleep.get("token_gradients", 0),
        "ic": sum(r["match"] for r in ic),
        "n": len(final),
        "margin_install": sum(bool(r.get("margin_install")) for r in final),
        "margin": sum(r.get("margin", 0.0) for r in final) / max(1, len(final)),
        "fact_margin": {r["fact"]: r.get("margin") for r in final if r.get("margin") is not None},
        "install": sum(r["match"] for r in final),
        "dlp": sum(r["logprob_delta"] for r in final if r.get("logprob_delta") is not None) / max(1, len(final)),
        "para": sum(r["paraphrase_rate"] for r in final) / max(1, len(final)),
        "lost": loc[-1]["lost"], "items": loc[-1]["items"], "dppl": loc[-1]["ppl_delta"],
    }


def load_cells(paths: list[str]) -> list[dict[str, object]]:
    return [c for c in (cell(p) for p in paths) if c]


def apply_floor(cells: list[dict[str, object]]) -> int:
    """Baseline-correct every margin against the same fact's no-sleep margin.

    A raw margin is not a measure of learning. The distractor is one fixed
    random code, and nothing makes it as likely a priori as the real one: at
    the untrained floor the per-fact margins measured -1.6 to +4.1 nats, so the
    registered "installed iff margin >= 1.0" fires for two thirds of facts that
    were never trained on at all. The floor arm is per (seed, fact) constant, so
    subtracting it costs nothing and gives the metric a true zero. Returns how
    many facts the floor itself would have scored installed -- print it, because
    it is the evidence that the raw column cannot be read.
    """
    floor: dict[tuple[str, str], float] = {}
    for c in cells:
        if c["arm"] == "nosleep":
            for fact, m in c["fact_margin"].items():
                floor[(str(c["seed"]), fact)] = m
    false_positives = sum(m >= INSTALL_NATS for m in floor.values())
    for c in cells:
        deltas = [m - floor[(str(c["seed"]), f)] for f, m in c["fact_margin"].items()
                  if (str(c["seed"]), f) in floor]
        c["dmargin"] = sum(deltas) / len(deltas) if deltas else float("nan")
        c["dinstall"] = sum(d >= INSTALL_NATS for d in deltas)
        c["dn"] = len(deltas)
    return false_positives


def check_hashes(cells: list[dict[str, object]]) -> None:
    """Every cell of a seed distilled the same wake transcript and the same
    dream, or nothing is pooled. Scoped to wave 1: in the multi-sleep grid a
    wave-2 dream legitimately differs per arm, so those are recorded, never
    asserted."""
    by_seed: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for c in cells:
        by_seed[str(c["seed"])].append(c)
    for seed, group in sorted(by_seed.items()):
        for field in ("transcript_sha", "dream_sha"):
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
            print(f"  {c['arm']:22} init_adapter {sorted(str(s)[:16] for s in c['init_adapter'])}  ({c['path']})")
        raise SystemExit(
            f"{len(shas)} distinct warm starts across {len(cells)} cells -- the arms began from different "
            f"weights, so nothing here is comparable. Rerun the odd cells with the same --init-adapter."
        )
    return next(iter(shas), None)


def main(pattern: str) -> None:
    paths = sorted(glob.glob(pattern))
    cells = load_cells(paths)
    skipped = [p for p in paths if p not in {c["path"] for c in cells}]
    if skipped:
        print(f"skipping {len(skipped)} cell(s) with no done record (still running, or died): "
              f"{', '.join(os.path.basename(p) for p in skipped)}\n")
    if not cells:
        raise SystemExit("no completed cells yet")
    check_hashes(cells)
    adapter = check_init_adapter(cells)
    seeds = sorted({str(c["seed"]) for c in cells})
    print(f"{len(cells)} cells, seeds {', '.join(seeds)}; wave-1 dream hashes agree within every seed; "
          f"{f'warm start {adapter[:12]}' if adapter else 'no warm start'}")
    false_positives = apply_floor(cells)
    floor_n = sum(c["dn"] for c in cells if c["arm"] == "nosleep")
    if floor_n:
        print(f"raw-margin floor: the untrained no-sleep arm clears the {INSTALL_NATS:.1f}-nat bar on "
              f"{false_positives}/{floor_n} facts -- read dmarg/dinst (floor-corrected), not marg/inst")

    print(f"\n{'arm':22} {'seed':5} {'rehrs':>6} {'bound':>6} {'misb':>5} {'ic':>4} {'marg':>7} "
          f"{'inst':>5} {'dmarg':>7} {'dinst':>6} {'EM':>4} {'dlogp':>7} {'para':>5} {'tokgrad':>8} "
          f"{'lost':>6} {'dPPL':>8}")
    for c in cells:
        print(f"{c['arm']:22} {c['seed']:5} {c['rehearse']:6.3f} {c['bound_cov']:>4}/4 {c['misbound']:>5} "
              f"{c['ic']:>2}/4 {c['margin']:+7.2f} {c['margin_install']:>2}/{c['n']:<2} "
              f"{c['dmargin']:+7.2f} {c['dinstall']:>3}/{c['dn']:<2} {c['install']:>2}/{c['n']:<1} "
              f"{c['dlp']:+7.3f} {c['para']:5.2f} {c['token_gradients']:>8} "
              f"{c['lost']:>2}/{c['items']:<3} {c['dppl']:+8.4f}")

    print("\npooled by arm (all seeds):")
    pool: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for c in cells:
        pool[str(c["arm"])].append(c)
    print(f"{'arm':22} {'n':>2} {'install':>8} {'dmarg':>7} {'dinst':>8} {'EM':>7} {'dlogp':>7} "
          f"{'para':>5} {'tokgrad':>9} {'lost':>6} {'dPPL':>8}")
    for arm, group in sorted(pool.items()):
        n = len(group)
        facts = sum(int(c["n"]) for c in group)
        dfacts = sum(int(c["dn"]) for c in group)
        print(f"{arm:22} {n:>2} {sum(c['margin_install'] for c in group):>3}/{facts:<4} "
              f"{sum(c['dmargin'] for c in group) / n:+7.2f} {sum(c['dinstall'] for c in group):>3}/{dfacts:<4} "
              f"{sum(c['install'] for c in group):>3}/{facts:<3} {sum(c['dlp'] for c in group) / n:+7.3f} "
              f"{sum(c['para'] for c in group) / n:5.2f} {sum(c['token_gradients'] for c in group):>9} "
              f"{sum(c['lost'] for c in group):>3}/{sum(c['items'] for c in group):<3} "
              f"{sum(c['dppl'] for c in group) / n:+8.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/g2_*_s*.jsonl")
