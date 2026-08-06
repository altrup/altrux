#!/usr/bin/env python3
"""Score the CL A/B grid: the install-vs-damage frontier per arm and budget.

Installs are conditioned on rehearsal the way DISCUSSION-20260805 sec 4 asks,
but on distinct-fact COVERAGE rather than rehearsal_fraction -- a dream that
loops one fact scores a high fraction while being able to install only that
one fact (see the run notes for the decoded example).
"""
import collections
import glob
import json
import os
import sys


def cell(path: str) -> dict[str, object] | None:
    rows = [json.loads(l) for l in open(path)]
    loc = [r for r in rows if r["phase"] == "locality"]
    if not loc:
        return None
    dream = next((r for r in rows if r["phase"] == "dream"), None)
    probe = [r for r in rows if r["phase"] == "probe"]
    ic = [r for r in rows if r["phase"] == "in_context"]
    counts = dream["needle_counts"] if dream else {}
    codes = {k: v for k, v in counts.items() if " " in k}
    rehearsed = {k for k, v in counts.items() if v > 0}
    # A fact is testable only if the dream mentioned its code at all.
    scored = [r for r in probe if r["code"] in rehearsed]
    name = os.path.basename(path)[len("dream_"):-len(".jsonl")]
    arm, _, seed = name.rpartition("_s")
    return {
        "arm": arm, "seed": seed,
        "rehearse": dream["rehearsal_fraction"] if dream else 0.0,
        "cov": sum(v > 0 for v in codes.values()),
        "ic": sum(r["match"] for r in ic),
        "install": sum(r["match"] for r in probe),
        "cond_n": len(scored), "cond_hit": sum(r["match"] for r in scored),
        "dlp": sum(r["logprob_delta"] for r in probe if r.get("logprob_delta") is not None) / max(1, len(probe)),
        "para": sum(r["paraphrase_rate"] for r in probe) / max(1, len(probe)),
        "lost": loc[-1]["lost"], "items": loc[-1]["items"], "dppl": loc[-1]["ppl_delta"],
    }


rows = [c for c in (cell(p) for p in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                                                      else "logs/dream_*_s*.jsonl"))) if c]
if not rows:
    raise SystemExit("no completed cells yet")

print(f"{'arm':22} {'seed':5} {'rehrs':>6} {'cov':>4} {'ic':>3} {'inst':>5} {'cond':>7} "
      f"{'dlogp':>7} {'para':>5} {'lost':>6} {'dPPL':>8}")
for r in rows:
    print(f"{r['arm']:22} {r['seed']:5} {r['rehearse']:6.3f} {r['cov']:>2}/4 {r['ic']:>2}/4 "
          f"{r['install']:>2}/4 {r['cond_hit']:>3}/{r['cond_n']:<3} {r['dlp']:+7.3f} {r['para']:5.2f} "
          f"{r['lost']:>2}/{r['items']:<3} {r['dppl']:+8.4f}")

print("\npooled by arm (all seeds):")
pool: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
for r in rows:
    pool[r["arm"]].append(r)
print(f"{'arm':22} {'n':>2} {'inst':>7} {'cond':>8} {'dlogp':>7} {'lost':>6} {'dPPL':>8}")
for arm, rs in sorted(pool.items()):
    n = len(rs)
    cn = sum(r["cond_n"] for r in rs)
    print(f"{arm:22} {n:>2} {sum(r['install'] for r in rs):>3}/{4 * n:<3} "
          f"{sum(r['cond_hit'] for r in rs):>3}/{cn:<4} {sum(r['dlp'] for r in rs) / n:+7.3f} "
          f"{sum(r['lost'] for r in rs):>3}/{sum(r['items'] for r in rs):<3} "
          f"{sum(r['dppl'] for r in rs) / n:+8.4f}")
