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
2. Apply the state ablation: `S ← S(I − γ ĉ_tĉ_tᵀ)` where `ĉ_t` is the
   token's own read query — targeted consume-what-you-just-read. γ is
   **hard-coded** (partial removal, γ < 1; no learned gate exists).
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
- **Locality / catastrophic forgetting:** ΔPPL on a fixed held-out
  general-text slice, measured pre/post each sleep, per arm. This is the
  direct A-vs-B forgetting comparison altrup asked for.
- **Backward transfer:** wave-1 fact recall probed after sleep 2 (did
  learning wave 2 destroy wave 1).
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

**Hyperparameters:** γ default 0.5 (one knob; sweep only if the smoke run
shows it saturating — dlogp flat and dream degenerating → try 0.25/0.9).
Dream length: fixed token budget per sleep, default 512, logged. Gradient
step per token (sleep is offline; if throughput is a problem, accumulate
over small windows — record the window size in the jsonl).

**Reading the result:** B ≥ A on reliability with smaller ΔPPL → the novel
mechanism wins where it's supposed to (on-policy KL to own logits should
drift less than off-policy SFT — that's the hypothesis being tested, not
assumed). B ≪ A on reliability → dream rehearsal isn't delivering the
facts to the reads; check the rehearsal fraction before blaming the
mechanism. Both heavily forgetting → the LoRA budget confounds both arms;
report, don't tune past it.

## 5. Work queue

**Local (this machine, in order):**

1. **Erase-efficacy probe** (pure inference, cheap): prime a 4×40
   transcript, apply the rank-1 erase with one fact's read query, probe all
   four facts. Expect: target fact degraded, others intact (the algebra
   note says non-overlapping keys are preserved). This is the physics of
   step 2 of the protocol and has never touched a real model. If targeted
   erase can't remove a binding, the protocol needs rework before any box
   time.
2. **Build the sleep loop** — new `sft/dream_sleep.py` (generation loop
   with per-token teacher-capture → ablate → distill-step → sample), plus
   the A/B driver and the new probes (paraphrase templates, fixed PPL
   slice). CPU-testable pieces get tests; TDD applies.
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
