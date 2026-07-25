# Discussion notes — 2026-07-25: CL sleep gradient analysis, filter test C, implementation kickoff

Team discussion (altrup + Claude), building on
`DISCUSSION-20260724-next-run-plan.md` (referenced below as "the plan") and
`RESEARCH-20260724-local-diagnostics.md`.

## 1. Does the CL sleep loop train the model away from M? (No — with two real risks)

Question raised: the plan's §3 sleep loop (dream with M intact storing logits
→ decay-on-read erase → teacher-forced pass with eroded M → KL(stored ‖ new)
into LoRA) trains the whole model after making M forget — does that gradient
push toward ignoring M and leaning on the SSM? Analysis says no:

- Target and student passes share tokens **and** SSM state, so the KL gap
  isolates exactly M's contribution. It cannot be closed by "read the SSM
  instead" (the SSM term cancels) — only by weights.
- The target *includes* M's contribution, so suppressing reads does not
  reduce the loss. This is the inverse of the training-run anneal dynamic
  (plan §2.2), where zeroing the read path did reduce loss.
- Decay-on-read leaves M(q) ≈ 0 — reads of an eroded M are no-ops, not
  noise. No pollution, no pressure to close the gate.
- beta/o_proj/M slow weights are outside the CL trainable set (plan §3
  distills into LoRA only) — the suppression knob isn't trainable at sleep.

Two real adjacent risks:

1. **Atrophy, not suppression.** Every CL gradient step conditions on an
   eroded M; nothing in the loop rehearses read-with-full-M, which only the
   offline run installs. LoRA drift can passively degrade M-integration over
   a long CL horizon — same end state as suppression, arrived at by neglect.
   No mitigation adopted; candidates recorded: periodic rehearsal passes
   exercising a populated M against wake-time data.
2. **Co-held content shredding** — a found hole, fixed by §2 below.

## 2. Design change (adopted): gate the erase on the consolidation signal

Sleep does not wipe the SSM (deliberate, prior note §4.2 — dreaming over a
zeroed state is the OOD condition that plausibly killed the B0 dream
result). So at sleep time, recent wake content sits in **both** M and the
SSM. For that co-held content the SSM masks M's incremental contribution →
KL gap ≈ 0 → **nothing consolidates** — while decay-on-read erases it from
M unconditionally. Sequence: erased from M, never committed to weights,
later displaced from the SSM by wake interference → destroyed by the sleep
that was supposed to preserve it. Older M-exclusive content is handled
correctly; recent content is silently shredded.

**Change:** only decay M at reads whose chunk (or position) showed a real
KL gap — *forget what you actually committed*. Co-held content then
survives in M through this sleep and consolidates at a later one, once
interference has pushed it out of the SSM and the gap opens. The per-chunk
KL is already computed for the loss; the gate is a threshold comparison.
Consistent with the existing unread-content-survives principle — extends
"unread" to "read but not needed."

## 3. beta/o_proj stay frozen during sleep (considered unfreezing; rejected)

- Under full erase, o_proj's input activations are ~0 → essentially no
  gradient. Unfreezing is a no-op.
- Under partial erase (the real case — decay is one gradient step), the
  student reads γ·content against a target built from the full-strength
  read; the cheapest KL reduction is inflating beta/o_proj by ~1/γ. That
  bias is systematic — every sleep chunk pushes the same direction — and
  the inflated gain then over-injects on a *full* M at next wake (the plan
  §2.2 pollution/instability regime reintroduced by CL itself).
- Distilling with M intact gives KL ≡ 0 (target and student are the same
  pass) — the sleep loss structurally cannot teach read-with-full-M, only
  compensate-for-absent-M.
- If the read path is ever CL-trained, the honest signal is a **wake-time**
  loss (continuous gradient accumulation, data-structure note §4.5),
  accepting that wake gradient reintroduces genuine suppression pressure —
  the same pressure the offline run must survive anyway.

## 4. M-necessity framing after a field check

- The field default for consolidation-style CL is transcript/context-sourced
  training data: SEAL generates fine-tuning data from a passage in context;
  Cartridges' self-study is the same shape; 2605.26099 runs offline passes
  over accumulated real context; SuRe replays buffered real examples. None
  use a memory module as the source. M-as-replay-source has essentially one
  precedent — the Hope sleep paper (2606.03979), our own lineage. (SEAL/
  Cartridges mechanics are snippet-level in the survey note — verify before
  leaning on them in a design doc.)
- The transcript-consolidation arm is therefore the **field-default null**,
  not a strawman. Elevated: run it locally before the box run — prefix
  transcript → brief LoRA distillation during "sleep" → behavioral recall
  with state wiped. If lossless-transcript distillation can't install
  recallable facts, M-replay can't either, and CL-by-consolidation needs a
  rethink before the training budget is spent. If it works, it's the
  harness and the bar for M-replay.
- M's only non-substitutable role is **wake-time working memory** beyond the
  SSM's interference capacity (measured: dead by ~192 tokens of dense
  interference, diagnostics §1). The planned dataset trains exactly that
  role, so it holds value even under an M-not-needed-for-CL outcome.

## 5. CL evaluation (settled shape)

Definitive readout per cycle: wake on session content → sleep → probe with
**M emptied and the SSM displaced by interference** — anything recalled now
can only be in the weights. Plus:

- **Behavioral tier**: generation-based exact recall, not teacher-forced
  logprob deltas (data-structure note §4.7; 2607.00368). Gist-delta is not
  the headline CL metric.
- **Arms**: no-sleep baseline / transcript-consolidation / M-replay.
- **Locality control** after every consolidation (general-knowledge probes
  before/after).
- **Multi-sleep durability** (north-star #3) — iterated self-distillation is
  where drift/collapse shows.
- **Process metrics** per sleep: read coverage and KL-gap mass, so
  "consolidated nothing" is distinguishable from "worked".

## 6. Filter: test C adopted (three tests)

- **A (well-posed):** backbone + source + cue → must answer.
- **B (memory-required):** backbone + interference + cue, source absent →
  must fail.
- **C (SSM-insufficient):** backbone + source + interference + cue,
  in-stream → must fail. Per-item guarantee that no SSM-solvable item gets
  ×16 weight as an anti-M lesson. Gate 1's result predicts a high pass
  rate; C converts that statistical sizing into a per-item property. Cost:
  a second cached-state sweep (states now include the source, so B's cache
  doesn't reuse) — roughly doubles filter compute, still offline, one-time.

## 7. High-discard-rate policy

Oversample candidates; do not raise the cram token share. Wikipedia is
effectively unbounded and generation is nearly free — a discard rate r just
means 1/(1−r)× candidates for the same 35%. Discard *composition* is
diagnostic: mostly-A failures → cloze construction is bad; mostly-B →
entity substitution isn't biting. Fix the generator, don't brute-force.
Discarded items keep their passages as carrier/interference — only the
recall credit is dropped, so filtering wastes probes, not tokens.

## 8. Implementation kickoff (this session)

Two parallel Opus implementation streams, disjoint file domains, interface
= separate per-slice `.pt` files conforming to the existing `train.pt`
schema (documented additions allowed):

- **Data:** IMR cram generator + needle generator + three-test filter
  (`sft/prepare_*`), curriculum encoded data-side.
- **Trainer:** recall-weight ramp 1→16 tied to the beta-anneal window;
  per-dataset chunk-len (512 + gradient checkpointing on cram slices; BX
  constants untouched for chains/ballast) (`sft/train.py`).

Transcript-consolidation null and the pilot-artifact read follow once these
land. GPU validation (filter pilot, VRAM/tok-s) stays in the main session —
subagents write and CPU-test only.
