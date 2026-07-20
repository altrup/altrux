# Discussion notes — 2026-07-20, post-mortem of the episodic-chains run

Team discussion (altrup + Claude) after reading `EXPERIMENT_NOTES-20260720-023324.md`.
These are STANDING DIRECTION notes for the next experimenter run — read them
alongside the prior run notes; they say what the next rented-box session
should do and why.

## Reinterpretation of the run's conclusion

The run concluded "`M` cannot reliably store & retrieve specific facts across
an SSM wipe." We now think that conclusion is only established for **verbatim
exact-code recall** — and the probe demands exactly the thing `M` was not
designed for. `M` is a gist mechanism; the probe scores the log-prob of 5
exact digits, where any fuzzier trace contributes ~nothing. The step-180
+0.34–0.42 delta proves the write→wipe→read path *can* carry information;
what's unestablished is whether `M` carries **gist** that the exact-code probe
is blind to.

## Priority 1 for the next run: the cross-sleep GIST eval

Build and run a gist-shaped eval (the run notes' own "resolve a probe-design
question FIRST" item). Design agreed in the team discussion:

- Same harness as `probe_recall.py` (prefix → `sleep_slot` backbone wipe →
  ablation = fresh random `M`), implemented as a mode of that script, not a
  new one.
- Delete the engineered fact/query machinery for this mode. Instead: feed a
  real conversation/document prefix (held-out text, or chains data), wipe the
  backbone, then teacher-force the **actual continuation** (~1–2k tokens) and
  score mean per-token log-prob, memory intact vs ablated. No labels or
  summaries needed — the text's own continuation is the ground truth.
- **The wipe MUST land mid-conversation of one continuous text** — chains
  place most sleeps at between-EPISODE boundaries, and episodes are
  independent conversations, so a boundary-sleep continuation carries ~zero
  information about the pre-sleep text: nothing to measure, by construction.
  (The chains design's ~20% mid-conversation between-turn sleeps are the
  right shape; the eval should use long continuous conversations/documents
  and place the wipe at a between-turn point.)
- **Distance-grade it**: vary the wipe depth / check whether the delta
  survives when the continuation depends on EARLY pre-sleep content, not just
  the final turns before the wipe. This distinguishes "`M` holds episodic
  gist" from "`M` is a last-few-turns buffer" — the latter is still a real
  only-`M` contribution (SSM wiped, prior can't know this text) but is the
  bottom rung, and we want to see whether it plateaus there.
- Metric: intact − ablated mean log-prob over post-sleep continuation tokens
  (equivalently a perplexity delta). Any retained signal — topic, entities,
  style, facts — shows up; it is the most permissive detector of "`M` stored
  *something*."
- Caveat: the per-token delta is small and smeared — needs enough
  continuation tokens and probes to clear noise (SEM discipline as in the
  existing probe).
- Run it across the interesting checkpoints already on disk: step-180
  (baseline, exact-recall delta +0.4), 256/295 (collapsed), 332 (transient
  recovery), 396, 435 (freeze-lora). The trajectory of the *gist* delta
  across these is the payoff — does it track the exact-code delta, or was
  gist intact all along?

Decision rule:
- **Gist delta clearly positive** (esp. where exact-code delta was ~0) →
  `M` works at its designed job; the exact-code emphasis was the wrong ruler
  and plausibly the wrong training signal. Proceed to Priority 2.
- **Gist delta ≈ 0 too** → `M` stores nothing usable in any currency; the
  structural conclusion stands. Fall back to the run notes' structural items
  (freeze-at-step-180 preservation test, auxiliary local write supervision,
  capacity changes) — team discussion, likely stop rather than grind.

## Priority 2 (only if gist delta is positive): gist-shaped training

Shift training gradient from the task `M` demonstrably loses (verbatim codes,
`--recall-weight 8`) toward the task it can win (cross-sleep natural
continuation, which the parametric prior cannot fully absorb — the specifics
of *this* chain exist only in `M`). Two parts, both needed:

- Neutralize/remove the engineered exact-code queries (`--recall-weight 1`,
  or regenerate without them).
- **Raise the mid-conversation sleep fraction well above the current ~20%**
  (a `prepare_chains.py` change): between-EPISODE sleeps carry no natural
  cross-sleep signal (independent episodes), so dropping the queries without
  moving sleeps into conversations would leave almost no cross-sleep gradient
  at all. Keep `--head-weight` on post-sleep tokens.

Known risk, accepted: the gradient will first teach `M` to carry the most
RECENT pre-sleep context (recency dominates continuation prediction). That is
still an only-`M` contribution (SSM wiped, prior can't know the text) — the
bottom rung, fine as a start. Watch the distance-graded eval for plateauing
at short range; if it plateaus, that's when long-range gist-shaped engineered
demands (not verbatim codes) earn their way back.

Track the GIST delta (not the exact-code delta) across checkpoints as the
success criterion.

## Explicitly considered and rejected

- **chunk-len 48 → 64**: extends the BPTT credit-assignment horizon by 16
  tokens against a fact→query gap of thousands + a sleep — ~1000× too small
  to matter, and it costs peak VRAM (batch 12 already OOMed; 64 would force a
  batch cut). Don't spend time on it.

## Housekeeping since the run (already committed, aa2ae1e)

- `make probe-recall` now tees to `sft/logs/probe-<timestamp>.log` — probe
  output lands with the training logs and rides the existing rsync pull. Do
  NOT redirect probe output into `notes/` anymore.
- `lambda_launch.sh` starts setup detached and prints attach commands instead
  of attaching live.
