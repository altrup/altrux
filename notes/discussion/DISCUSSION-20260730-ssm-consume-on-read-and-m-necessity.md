# Discussion notes — 2026-07-30: consume-on-read in the SSM state, and what it does to the M-necessity question

> **SUPERSEDED (2026-08-05):** the wake-time consume-on-read design (§1) and
> the gate-collapse check (§5a) recorded here were a drift from the intended
> design. The erase is sleep-only, hard-coded γ, no learned gate — see
> `DISCUSSION-20260805-dream-distillation-cl-ab.md` §3. Do not resurrect the
> wake-time version. §3 (M-necessity) and the algebra note remain valid.

Team discussion (altrup + Claude). Standing direction. Builds on
`DISCUSSION-20260725-cl-sleep-analysis-and-filter-testc.md` (§1 sleep-loop
analysis, §2 KL-gated erase, §4 M-necessity, §5 CL evaluation),
`../research/RESEARCH-20260724-local-diagnostics.md` §1 (the interference measurement),
and the algebra/literature scan in
`../research/RESEARCH-20260730-erase-on-read-in-the-ssm-state.md` (referenced below as
"the algebra note").

Starting question: *decay-on-read currently operates on M. Can it operate on
`ssm_state` instead — and if it can, do we need M?*

Short answer: the mechanism transfers exactly and is cheaper there, the
loop-level argument transfers with one inversion worth knowing about, and
the M-necessity answer does **not** change — because it was never blocked on
the erase mechanism. It is blocked on a measured capacity number.

## 1. The mechanism transfers, and is strictly better-behaved on the SSM

`ssm_state` is a linear associative memory: `B` is the key, `C` the query,
`x` the value (algebra note §1). Decay-on-read becomes

```
S ← S (I − γ_t ĉ_t ĉ_tᵀ)
```

which is the **closed form** of the gradient step toward `(q, 0)` that we
currently take approximately on M. Three consequences:

- **Exact, not iterative.** On M we approach the fixed point; here we land
  on it. `γ ∈ (0,1)` is the lossiness knob — the bounded analogue of the
  learning rate, with the unstable regime unreachable by construction.
- **Nearly free.** `S·C` is already computed in `_mixer_step`. The erase is
  one outer product plus a per-head sigmoid gate (~48 params/layer). No
  gradient computation, no separate memory module, no extra matvec.
- **§2's KL-gating comes partly for free.** The 07-25 change was "only decay
  M at reads that showed a real KL gap" — forget what you actually
  committed. On the SSM, erasure is proportional to retrieval *by
  construction*: each stored key rotates away from the query by exactly its
  overlap, so an item that contributed nothing to the read is preserved
  bit-for-bit. That is not identical to KL-gating (contribution ≠
  consolidation), but it is the cheap approximation, and it is automatic
  rather than bolted on.

The erase must fire **after** the read — forced, not chosen. If the address
is the readout query, erasing first deletes what you were about to retrieve.

## 2. The loop-level argument inverts — and that inversion matters

07-25 §1 established that the sleep loop does not train the model away from
M, because target and student share tokens **and** SSM state, so the KL gap
isolates M's contribution and "cannot be closed by reading the SSM instead
— the SSM term cancels."

Erase the SSM instead of M and the argument runs symmetrically: target and
student share tokens and M, the gap isolates the SSM's contribution, and
consolidation-from-SSM works by the same logic.

**But the escape hatch inverts with it.** With M present and the SSM eroded,
the cheapest way for the student to close the gap is to lean harder on M —
the mirror of the failure §1 ruled out. An SSM-erase loop with M in place
would train exactly the suppression dynamic we went to some trouble to avoid,
pointed the other way.

This is not a reason to reject SSM-erase. It is a reason the two are
**mutually exclusive within one sleep pass**: erase M *or* erase the SSM,
never both-with-the-other-intact-and-trainable. The clean configuration for
an SSM-erase loop is M absent entirely — then the only thing that can close
the KL gap is weights, and the argument is tighter than the current loop's,
not weaker.

Which is convenient, because "M absent entirely" is the arm we would be
testing anyway.

## 3. What does *not* change: the M-necessity answer

07-25 §4 already settled this, and this session's algebra does not move it:

> M's only non-substitutable role is **wake-time working memory** beyond the
> SSM's interference capacity (measured: dead by ~192 tokens of dense
> interference, diagnostics §1).

The erase mechanism was never what M was for. Dropping M does not fail
because we lacked a way to erase the SSM — it fails, if it fails, because
`d_state = 128` per head is a hard capacity bound and our own measurement
puts the practical horizon at ~192 tokens of dense interference. Beyond
that, content is gone before any sleep can consolidate it.

So the honest framing of "maybe we don't need memory" is:

**An SSM-only CL loop trades memory horizon for mechanism simplicity, and
the exchange rate is the sleep cadence.** If sleeps are frequent relative to
the interference horizon, the SSM holds everything that a sleep would
consolidate anyway and M is dead weight for CL. If sleeps are sparse, M is
carrying content the SSM has already lost, and dropping it loses that
content permanently.

That is a measurable quantity, not a matter of taste, and nobody has
measured it. It is the single most decision-relevant experiment available
right now and it is cheap — see §5.

Note this leaves 07-25 §4's conclusion intact in both directions: even under
an "M-not-needed-for-CL" outcome, M's wake-time working-memory role survives
and the planned dataset trains exactly that role.

## 4. Ordering — this does not jump the queue

The transcript-consolidation null (07-25 §4) is still the field-default
baseline and still gates everything downstream. If lossless-transcript
distillation cannot install recallable facts, then neither M-replay nor
SSM-replay can, and CL-by-consolidation needs a rethink before any of this
matters. Run it first. Nothing in this note changes that ordering.

Likewise the stage-2 780M integration screen and the data work in flight
(07-25 §8) proceed unchanged. Consume-on-read touches `_mixer_step` and adds
one gate; it does not interact with the mix/state arm comparison, and the
A/B purity contract is unaffected.

## 5. Standing direction — three experiments, in this order

**(a) Gate-collapse check. Cheapest, most decisive, run first.**
Add per-head `γ_t = sigmoid(W·h_t)` with `bias = -4` to the 780M backbone's
injected layers, erase post-read, train briefly under the **CL objective**
(not LM perplexity), and log `γ` statistics per layer per head.

The failure mode is that `γ` learns to be ~0 and switches the mechanism off.
Under an LM objective this is close to a foregone conclusion — EDA's probe
shows an optimizer given a free choice moves the erase *away* from
disturbing the readable state (algebra note §5). Under the CL objective,
erasing consumed material is principled rather than perverse, and that
difference is the entire bet. If `γ → 0` here too, the idea is dead and we
bought that knowledge for one short run.

Log `γ` distribution, saturation fraction, and per-layer means. "Gate
collapse" must be distinguishable from "gate selective" in the run log, not
inferred afterwards — the same standard as §5's process metrics.

**(b) The horizon measurement.** Extend diagnostics §1 from "when does the
SSM stop answering" to "how much of a wake session is still recoverable from
the SSM alone at sleep time, as a function of intervening interference."
This is the exchange rate in §3. It determines whether an SSM-only loop is
viable at our intended sleep cadence, and it is a measurement, not a
training run.

**(c) Only then: the SSM-only CL arm.** As a fourth arm alongside 07-25
§5's no-sleep / transcript-consolidation / M-replay. Same evaluation shape —
behavioral generation-based recall, M emptied *and* SSM displaced at probe
time, locality control, multi-sleep durability. The SSM-only arm has M
absent by construction (§2), which makes its probe condition simpler than
the others': displace the SSM and anything recalled is necessarily in
weights.

## 6. Not doing

- **Not dropping M** on the strength of this. §3 — the blocker is a capacity
  measurement, and it has not been taken.
- **Not stacking EDA's cleanup erase** (learned address, pre-write) on top
  of consume-on-read. They are compatible — two rank-1 projections at
  different addresses at different points in the step — but that is two new
  gates per layer before we know whether one of them does anything. If
  consume-on-read collapses to `γ ≈ 0`, stacking a second erase would not
  have told us why.
- **Not changing the memory arms.** The 780M screen (`memory_mix` vs
  `memory_state`) is orthogonal and proceeds as specified.
- **Not writing this up externally yet.** The operator is three months old
  and published (algebra note §3–4); the contribution is the loop, and the
  loop is untested. There is nothing to claim until (a) and (b) return.

## 7. Honesty markers

- The algebra in §1 is verified against our own `_mixer_step` and derived
  from scratch; it is not in doubt.
- The literature claims it rests on are mostly [S] — see the algebra note
  §8. The DNC free-gate comparison in particular is load-bearing for "this
  is not novel" and has not been read from source.
- "Nobody has measured the exchange rate in §3" is a claim about our own
  work, not the field. It is possible the long-context literature has an
  equivalent number; not searched for.
- EDA's probe numbers come from v1 HTML, not the PDF. Do not quote publicly
  without re-verifying.
