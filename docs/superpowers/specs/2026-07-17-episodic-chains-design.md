# Episodic chains — training the neural memory to occupy the episodic tier

2026-07-17. Applies to `models/mamba2_2_7b_memory` and `sft/`.

## Problem

Probes on the 2026-07-16 GH200 run (`notes/EXPERIMENT_NOTES.md`) showed the
neural memory `M` contributing ~nothing to recall: every recall demand in the
training data sits within the SSM's own capacity, so the optimizer routes the
task through the backbone and suppresses `M` (alpha climbing, o_t_norm
shrinking, ablation deltas ~0). Harder interference data narrowed but did not
close the gap — the SSM kept improving and re-absorbing the niche (probes #2,
#3). The niche must be made structural, not just hard.

## Goal

The three-tier hierarchy documented in `models/mamba2_2_7b_memory/README.md`
("Design goal — a three-tier memory hierarchy"): SSM = working memory
(current context), `M` = episodic memory (selected gist that survives context
turnover), parameters = skills/semantic. Deployment model: an always-on
assistant whose SSM state is occasionally wiped ("sleep") while `M` persists.
Training must mirror that regime: recall demands that cross an SSM wipe are
answerable only through `M` — no gradient path lets the backbone absorb them.

Out of scope: sleep-time consolidation (distilling `M` into parameters).
Sleep here is only a backbone-state wipe.

## Design

### 1. Data — `sft/prepare_chains.py` (evolves `prepare_interference.py`)

A **chain** is one long training example: a sampled sequence of episodes
(conversations) concatenated, with facts/queries spliced in and sleep points
recorded as metadata.

- **Pool**: the existing prepared data — normal-length examples and
  LongAlign long conversations, both.
- **Chain length**: sample a per-chain token budget, log-uniform 30k–130k;
  append sampled episodes until the budget is met, closing at the episode
  boundary that crosses it. Episode count is derived (~8–32 with the mixed
  pool). Short chains occur naturally in the mix; no separate staged
  curriculum.
- **Sleeps**: placed at randomly chosen *between-episode* boundaries so a
  wake spans 1–4 episodes; placement is unpredictable (no fixed cadence the
  model could cram against). Additionally ~20% of long conversations get one
  mid-conversation sleep at a *between-turn* boundary — the
  natural-continuation signal: every ordinary token after the wipe is
  predicted better iff `M` retained the gist of what preceded it. Sleeps
  never cut mid-turn.
- **Engineered queries at three distances**: within-episode,
  cross-episode-within-wake, and **cross-sleep** (the only-`M`-can-answer
  case). Heterogeneous fact types, revision/overwrite cases, and
  `recall_masks` (for `--recall-weight`) carry over from
  `prepare_interference.py`. Probe vocab slice `[0, 1024)` stays held out.
- **Output**: `sft/data/train_chains.pt` — ids, masks, `recall_masks`, and
  per-example `sleep_positions` (token offsets where the backbone resets).
  Deterministic given a seed; not in git/rsync.

### 2. Model — `models/mamba2_2_7b_memory/model.py`

- **Mid-sequence backbone reset**: a method that, for a given slot, zeroes
  the per-layer `ssm_state` and conv history while leaving the memory
  subsystem (`w1`/`w2`, momentum `S`) untouched. Sleep = backbone wipe only.
- **Alpha ceiling 0.1 → 1e-4** (the knob sigmoid multiplier). Rationale:
  BPTT truncation at `--chunk-len` means alpha's projection only ever
  receives short-horizon gradient ("decay now → cleaner reads now"); the
  compounded long-horizon cost of erosion is invisible to it, so the model
  cannot learn a retention-safe alpha — the ceiling is a prior we set.
  1e-4 makes the worst-case half-life ~7k writes (~55k tokens at
  memory-window 8). Alpha's remaining role is a stabilizer with a slow leak
  (bounds `w1`/`w2` drift, reclaims never-revisited content); targeted
  forgetting is the delta-overwrite path, whose credit assignment works
  within a chunk.

### 3. Training loop — `sft/train.py`

Slots keep the existing one-example-at-a-time machinery — a chain **is** the
example. The chunk driver consults `sleep_positions` and triggers the slot's
backbone reset at those offsets. `M` resets at chain end exactly as it resets
per-example today. Checkpoint format unchanged; a restarted mid-chain slot
replays its chain from the top, same as any example today.

### 4. Evaluation — `sft/probe_recall.py`

New cross-sleep condition: state N facts, wipe the backbone via the same
reset method, fill a gap, query. Success metric: with the SSM wiped, the
ablated condition sits at floor and intact sits above it — that gap is the
direct measure of `M` working. Existing no-sleep conditions remain for
continuity with the step-126/168/245/308 baselines.

### 5. Unchanged

Weighted-loss flags, memory-window batching, checkpoint cadence and format,
fused/manual forward dispatch. Whether the next run resumes step-350 weights
or starts fresh is a run-time call (resume cheaper; fresh cleaner if
suppression is sticky — the alpha recap de-fangs it either way).

### 6. Tests

- Model: mid-sequence reset zeroes backbone state and preserves `M`; alpha
  never exceeds the new ceiling.
- sft: chain generator places sleeps only at episode/turn boundaries, wakes
  span 1–4 episodes, queries exist at all three distances, masks align.
- Existing suites keep passing.

## Key decisions and their reasons (from the design discussion)

- **SSM wipes at sleeps, `M` persists** — makes the episodic niche
  structural: no backbone improvement can carry a fact across a wipe.
- **Wakes span multiple episodes (not per-episode wipes)** — mirrors the
  always-on deployment; the SSM legitimately handles cross-conversation
  recall within a wake, teaching *when* to fall back on `M`.
- **Natural continuations, not verbatim repeats** — repeating examples
  rewards token-for-token copying (the opposite of episodic gist);
  continuation-after-sleep rewards gist with a dense, template-free signal,
  guarding against overfitting to engineered query templates.
- **Unpredictable sleep placement** — a fixed cadence would train a
  pre-sleep cramming policy instead of continuous gist-writing.
- **Alpha capped, not learned-free** — see §2; the gradient is structurally
  blind to erosion's long-horizon cost.
