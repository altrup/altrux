# Discussion notes — 2026-08-05: dream-distillation sleep protocol, and the CL A/B

Team debrief (altrup + Claude) of the 2026-08-05 box session
(`EXPERIMENT_NOTES-20260805-040651.md`). Builds on
`DISCUSSION-20260804-binding-capacity-and-null-rerun.md`. **Supersedes the
consume-on-read design in `DISCUSSION-20260730-ssm-consume-on-read-and-m-necessity.md`
§1 and §5(a)** — see §3 below; the wake-time learned-gate version recorded
there was a drift from the intended design and is retired.

All numbers from the run notes were re-verified against the raw
`sft/logs/*.jsonl` during this debrief; every pooled figure reproduces
exactly.

## 1. What the run established (agreed)

1. **The fused SSD chunk-scan path is correct and changes the economics.**
   Oracle tests green on first execution; 780M ladder cells reproduce
   per-token-loop numbers (24×40 identical down to which fact survives). A
   null seed went ~30 min → 2m40s. 780M training/eval on rented CUDA is now
   cheap.
2. **Binding capacity is architectural, not a scale artifact.** 780M holds
   ~3–4 bindings, 2.7B holds ~5–8 at 3.5× the parameters;
   length-independent at both scales (4×800 ≡ 4×40). The capacity ladder is
   done as a question — no more cells.
3. **The pre-registered null FAIL is void; the corrected null tentatively
   passes.** The carried replay schedule only ever trained fact 0 under
   test-time conditions (6/6 installs at index 0, ~1/4096 under a uniform
   null). With `--fresh-state-replay`: 780M 2/9 → 4/9 installs, dlogp
   +1.81 → +3.74; 2.7B (chunk-len 48) 3/11 → 4/11, +2.27 → +3.22; first
   non-index-0 installs ever. Install-rate deltas alone are not significant
   (Fisher p≈0.3); the mechanism evidence (positional pattern, per-fact
   deltas, the flat ×4-steps control) is strong. Verdict we act on:
   **consolidation-by-distillation is mechanically alive; its role as a
   feasibility gate is discharged.**
4. **An untrained M is actively harmful and now quantified** (−1..−3
   bindings per cell, injection on vs off). This is the ablation bar a
   trained M must clear — parked with M (see §6).
5. **Cross-model null comparisons must pin `--chunk-len`** (2.7B default 7
   vs 780M 48 cost the 2.7B carried arm real signal: 1/11 → 3/11 at 48).

## 2. Direction pivot (agreed this debrief)

- **Focus is 780M.** M and the 2.7B arms are set aside until the CL A/B
  below has a result. The 2.7B fused port remains queued local work but
  does not gate anything on this docket.
- **No more null iterations.** The powered (10-seed) rerun discussed early
  in the debrief is dropped — the A/B's Arm B tests the real mechanism
  directly and subsumes the null's remaining evidential value.
- **The product of the next cycle is the CL A/B** (§4): dream-distillation
  sleep vs conventional fine-tuning, scored with established CL /
  knowledge-editing metrics including catastrophic forgetting.

## 3. The sleep protocol — signed off, supersedes 07-30

Design drift caught this debrief: 07-30 recorded consume-on-read as a
**wake-time** mechanism with a **learned** per-head gate γ_t, and staged a
gate-collapse check as the first experiment. That was never the intent.
The intended design, confirmed line-by-line by altrup today:

**Normal (wake) operation is exactly stock.** No gate, no erase, no
behavior change, no new parameters in the wake path.

**Sleep operation** — the model generates freely (EOS ignored, `[USER]`
token allowed, so it rehearses both sides of a session itself), and per
generated token t:

1. Forward from the current state → logits. Store as **teacher** signal.
2. Apply the state ablation: `S ← S(I − γ ĉ_⊥ĉ_⊥ᵀ)` where `ĉ_⊥` is the
   token's own read query **deflated against the state's top singular
   direction** (per layer, recomputed from the current `S`), and
   **γ = 1.0**. Both choices are probe-driven, registered in §3a below —
   hard-coded, no learned gate.
3. Gradient step: forward with the ablated state; train LoRA to match the
   stored teacher logits (KL). The gap is exactly the content step 2
   removed; only weights can close it.
4. Sample the emitted token from the teacher logits at `--temperature`;
   continue generation from the ablated state.

One state stream (teacher and student are one ablation increment apart; the
teacher degrades over the sleep — bootstrapped handoff to weights).
Two-stream (pristine teacher) is an **open question**, recorded in §7, not
part of v1.

Consequences: the gate-collapse check (07-30 §5a) is **cancelled** — there
is no learned gate to collapse. The every-token erase concern from this
debrief's discussion does not apply — the erase never runs during wake.

## 3a. Erase-efficacy probe results (run this debrief, local box)

`sft/erase_probe.py` (`make erase-probe`; committed `8469e37`, deflation
`5984d61`; logs `sft/logs/erase_probe*.jsonl`). Prime the 4×40 null
transcript, capture per-layer read queries while teacher-forcing each
fact's answer, apply the rank-1 erase, re-probe all facts. One seed
(1234), 780M. Findings:

1. **The facts' read queries share a ~0.9-|cos| interference cone.** The
   fact-discriminative content is a small orthogonal sliver per fact. This
   is also the mechanistic picture of the ~4-binding ceiling: readout is a
   mixture, capacity ends where sliver margins drown in cone mush.
2. **Sub-1 γ is a near-no-op** even applied at all 48 layers × 5 answer
   positions (raw γ=0.5: target −0.13 nats; deflated sweep 0.25/0.5/0.75/
   1.0 → −0.03/−0.09/−0.28/−0.74, supra-linear). A uniformly attenuated
   readout keeps its direction, and the mixer's gated RMSNorm renormalizes
   magnitude for free — the architecture has built-in automatic gain
   control, so partial erasure is undone at inference time.
3. **γ must be 1.0 for a second reason: compensation.** A γ<1 KL gap can
   be closed by amplifying the surviving signal, and amplifying the shared
   cone is a ~rank-1 change to `in_proj`'s C-slice — one of the cheapest
   directions available to the LoRA. At γ=1 the erased direction reads
   exactly zero and amplification has nothing to amplify. Residual escape
   (re-aiming C at a binding's orthogonal remnant) is rank-hungry and
   per-fact; the detector for all compensation flavors is fresh-state
   recall, which any state-reading strategy fails.
4. **Deflating the erase direction against the state's top singular
   direction makes the erase surgical, at no information cost.** The cone
   is where `S` concentrates its energy, so `S`'s top right-singular
   vector locates it from `(S, ĉ)` alone — streaming-computable in the
   dream loop. At γ=1: raw erase −1.05 target / −0.22 off-target; state-
   svd −0.72 / −0.066 (all targets still flip); oracle deflation (other
   facts' actual queries) −0.108 off-target — the self-contained estimate
   matches the oracle. Remaining collateral concentrates in the
   topaz–osprey pair, whose bindings are entangled *in the stored content*
   (topaz reads back osprey's code at baseline) — no erase direction can
   separate them.
5. **Erasure can un-mask interfered facts**: erasing clove recovered
   topaz's true code (a baseline miss) — direct evidence that freeing
   cone capacity restores drowned bindings, the capacity-freeing story the
   design is premised on.

Caveats: one seed, N=4, answer-span queries only. The SVD costs one
128-dim decomposition per layer per erased token — fine offline in sleep.

## 4. The CL A/B — design

Question: is dream-distillation sleep better than conventional CL
fine-tuning, and does it suffer the same catastrophic forgetting?

Model `mamba2_780m`, fused path, matched LoRA config and step budget across
arms, ≥3 seeds. Session material: the null's synthetic entity→code
transcripts, waves of **4 facts** (the measured capacity ceiling — sleep
cadence must be ≤ ~4 facts written, per the ladder).

**Structure (both arms):** wake 1 (4 facts) → sleep → wake 2 (4 new facts)
→ sleep → probe.

- **Arm A — conventional CL:** sleep = LoRA fine-tuning on the raw wake
  transcript (experience replay, the field default). State wiped at sleep.
- **Arm B — dream distillation:** sleep = the §3 protocol. State persists
  across sleep minus what the dream's reads consumed.
- **Control:** no-sleep (state carried, no training) — the ladder already
  gives its expected shape.

**Metrics (the established battery — knowledge-editing triad + CL
forgetting):**

- **Reliability:** fresh-state greedy exact match per fact (existing probe).
- **Generality:** paraphrased probes (~4 templates/fact, new) — installs
  knowledge vs memorizes a string.
- **Locality / catastrophic forgetting**, three layers:
  - **Pre-existing-knowledge battery (headline forgetting number):** a
    fixed battery of prompts the base 780M reliably answers correctly,
    built once before any training by self-calibration (run candidates
    through the base model greedy, keep only consistent hits — simple
    items are fine, they must be *its* knowledge). Re-probed after each
    sleep; forgotten = correct→incorrect flips, with per-item logprob
    drops as the sensitive measure. The frontier's forgetting axis is
    battery items lost + wave-1 items lost.
  - **Backward transfer:** wave-1 fact recall probed after sleep 2 (did
    learning wave 2 destroy wave 1), logprob drops alongside.
  - **ΔPPL** on a fixed held-out general-text slice, pre/post each sleep —
    the backstop for diffuse degradation no item battery catches.
- Per-fact logprob deltas throughout (the null taught us installs are too
  rare to carry significance alone).

**Arm-B-specific instrumentation (sanity-check-the-artifact rule):** the
dream transcript is decoded and printed live; log the fraction of dream
tokens that rehearse fact content. **Known risk:** free generation may
wander and never rehearse the facts, in which case reads never touch the
bindings and nothing distills. Decision rule: if the dream's fact-rehearsal
fraction is ~0 in the smoke run, seed the sleep generation with category
cues (a one-line prompt per wave) before concluding anything about the
mechanism.

**Hyperparameters:** γ = 1.0 with state-svd deflation, k=1 (probe-driven,
§3a — sub-1 γ is both ineffective and a compensation invitation; do not
re-tune γ downward without new evidence of a kind the probe couldn't see).
Dream length: fixed token budget per sleep, default 512, logged. Gradient
step per token (sleep is offline; if throughput is a problem, accumulate
over small windows — record the window size in the jsonl).

**Registered primary hypothesis (altrup, pre-data):** Arm B is expected to
be *slower* at installing facts; the bet is that its **ratio of learned to
catastrophically forgotten is better** — installation is more targeted.
Raw install rate at a matched budget is therefore NOT the headline and a
single-budget comparison would be misleading (B could sit lower on both
axes at one point while owning the better frontier). So each arm runs at
2–3 training budgets (Arm A: fine-tune steps; Arm B: dream length /
distill steps) and the primary outcome is the **install-vs-ΔPPL frontier**:
does B's curve dominate A's (more retained knowledge at equal damage, or
equal knowledge at less damage)? Backward transfer is read the same way —
wave-1 survival at matched wave-2 installation.

**Secondary readings:** B installs ~nothing at any budget → check the
dream's fact-rehearsal fraction before blaming the mechanism. Both arms
forget heavily at all budgets → the LoRA budget confounds both; report,
don't tune past it. B dominates the frontier → the mechanism's selling
point is confirmed even if A wins on speed.

## 5. Work queue

**Local (this machine, in order):**

1. ~~**Erase-efficacy probe**~~ **Done this debrief** — see §3a. Verdict:
   protocol viable; erase step registered as γ=1.0 + state-svd deflation.
2. **Build the sleep loop** — new `sft/dream_sleep.py` (generation loop
   with per-token teacher-capture → ablate (per §3 step 2, reusing
   `erase_probe.rank1_erase`/`deflate`/`state_top_dirs` and
   `Model.c_capture`) → distill-step → sample), plus the A/B driver, the
   Arm A transcript-SFT trainer, and the new probes (paraphrase templates,
   self-calibrated knowledge battery, fixed PPL slice). CPU-testable
   pieces get tests; TDD applies.
3. **2.7B fused port** — queued behind 1–2, does not gate the A/B.

**Box (next session, A10, in order):**

1. Smoke `dream_sleep.py` at 780M on the fused path: read the decoded dream
   and the rehearsal fraction before anything else.
2. The A/B per §4, 3 seeds.
3. Nothing else. No 2.7B, no M, no ladder, no null.

## 6. Explicitly considered and rejected

- **Powered 10-seed rerun of the corrected null.** Subsumed by the A/B
  (Arm B tests the real mechanism; the null was a proxy and its gate role
  is discharged). Revisit only if the A/B is unbuildable.
- **A/B on the null harness itself** (conventional-SFT arm inside
  `consolidation_null.py`). Rejected by altrup — the A/B belongs on the
  novel protocol, not the proxy.
- **Wake-time consume-on-read with a learned gate, and its gate-collapse
  check** (07-30 §5a). Superseded by §3 — design drift, not intent. Do not
  resurrect the wake-time version.
- **Hard-coding γ for a wake-time erase.** Moot once the erase is
  sleep-only; the sleep-only protocol is what makes hard-coding viable
  (fires only under programmatic control, no when-to-fire decision).
- **A third harness for high-N consolidation** (non-context-bound
  teacher). Still rejected; parked with M. The A/B runs at N=4 per wave,
  which the ladder validates.
- **More capacity-ladder cells at either scale.** The question is answered
  (§1.2).

## 7. Open questions

- **One vs two state streams in the sleep loop** (§3): one stream is v1 by
  decision; whether a pristine-state teacher materially changes what gets
  consolidated is untested. Revisit after the first A/B result.
- **Lit-search verdict (this debrief, web search): erase-on-read in a
  dense recurrent/SSM state appears unexplored.** Closest prior art: DNC
  free gates (read-triggered but slot memory, learned), stack-RNN pop
  (destructive read but stack semantics), DeltaNet-family erase (write-time
  only), KV-eviction (importance-based; some do the opposite). If the A/B
  favors Arm B, there is a claimable result; nothing to write up before
  then.
- Whether dream rehearsal needs seeding (§4 decision rule) — empirical,
  settled by the smoke run.

## 8. Housekeeping

- Run notes banked as `907c659`; four box commits (`197eb4e`, `d47e315`,
  `ac1d29c`, `ea4c37b`) were already on main at debrief start.
- Correction marker added to
  `DISCUSSION-20260730-ssm-consume-on-read-and-m-necessity.md` pointing
  here (wake-time consume-on-read superseded).
- Process rule adopted after a design-drift postmortem: **mechanism designs
  enter DISCUSSION files as math plus a numbered event sequence, and
  altrup signs off on that block specifically** before it becomes standing
  direction. Prose summaries drift; numbered sequences don't. (Recorded
  here only for now — not propagated into the command files, by decision.)
- Probe-session commits (same day, after the debrief proper): `7de6e00`
  (Mamba3 stub on ROCm + venv-rebuild gotchas in root CLAUDE.md — the
  local venv was rebuilt from empty; `UV_TORCH_BACKEND=auto` now resolves
  CPU torch, mamba-ssm needs `MAMBA_SKIP_CUDA_BUILD=TRUE` without hipcc),
  `8469e37` (erase probe + `Model.c_capture`), `5984d61` (`--deflate`).
