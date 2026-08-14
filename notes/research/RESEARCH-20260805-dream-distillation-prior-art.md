# Research — 2026-08-05: prior art for dream-distillation CL (two web sweeps)

Two literature sweeps run during the 08-05 debrief (web-search subagents;
titles verified, mechanisms summarized from abstracts/HTML — items marked
[S] are search-level reads, not full-paper reads). Context:
`../discussion/DISCUSSION-20260805-dream-distillation-cl-ab.md`. Sweep 1 asked whether
**erase-on-read** (retrieval consuming memory) has prior art; sweep 2 asked
the same for **generative replay + state-ablation-driven distillation**.

## 1. Erase-on-read / consume-on-read (sweep 1)

- **DNC free gates** — Graves et al., *Hybrid computing using a neural
  network with dynamic external memory*, Nature 2016. Read heads emit a
  "free gate"; freed locations return to an allocator. **Closest prior
  art**: read-triggered, capacity-freeing — but explicit slot memory with
  an allocator, and freeing is a soft learned gate, not a consequence of
  the read. (A ~2021 follow-up applies the deallocation to LM-ing.)
- **Stack-RNNs** — Grefenstette et al., *Learning to Transduce with
  Unbounded Memory*, 2015 (also the Neural Network Pushdown Automaton,
  arXiv:1711.05738). Pop is a genuinely destructive read — but stack
  semantics on structured memory, not distributed vector state.
- **LRUA / sparse MANNs** — Santoro et al. 2016; Rae et al.
  arXiv:1610.09027. Usage-tracked slot recycling — **write-time** policy,
  not read-triggered.
- **DeltaNet lineage** — delta rule, Gated DeltaNet, and two 2026 papers
  decoupling erase/write addresses (arXiv:2605.22791, arXiv:2606.26560).
  Architecturally adjacent (rank-1 erase in linear-attention state), but
  erasure always fires **on write**, never on a query.
- **KV-cache eviction** — H2O/TOVA/SnapKV family. Importance-*accumulation*
  decides eviction; some (MemDecay, arXiv:2607.10582) do the opposite —
  attention *extends* a token's life. Nobody frames a read as debiting the
  slot.
- **NTM** (Graves 2014): erase-then-add is write-time only — confirms the
  baseline the DNC extended.
- Tangential: *The Orthogonalized Read Is a Removable Training Scaffold
  for Recurrent Memory* (arXiv:2607.19390) — the only found paper
  instrumenting read-time transforms of a fast-weight matrix; negative
  result about a training scaffold, not erase-on-read.

**Verdict:** retrieval-triggered erasure in a dense recurrent/SSM hidden
state appears unexplored; bounded by DNC (slot memory, learned gate) and
stack-pop (structured memory).

## 2. Generative replay, context distillation, ablation-as-signal (sweep 2)

**Generative replay ("dreaming") — thoroughly established:**
- Robins 1995, *Catastrophic Forgetting, Rehearsal and Pseudorehearsal* —
  the ur-cite: the network manufactures its own rehearsal pairs.
- Shin et al. 2017, *Continual Learning with Deep Generative Replay*,
  NeurIPS — the modern canonical cite.
- LLM era: *Self-Synthesized Rehearsal* (SSR, arXiv:2403.01244, ACL 2024);
  **SEAL** (arXiv:2506.10943) — model generates its own fine-tuning data;
  on-policy replay for continual SFT (arXiv:2605.29495).

**Context/state → weights distillation — established:**
- Snell, Klein, Zhong 2022, *Learning by Distilling Context*
  (arXiv:2209.15189) — KL from context-conditioned teacher to context-free
  student. Direct ancestor of our objective.
- **Cartridges** (arXiv:2506.06266) — self-generated conversations about a
  corpus + KL to a full-context teacher; the closest single relative
  (self-study + context distillation), but distills into an auxiliary
  KV-cache artifact, not base weights, and no ablation defines the target.
- Gisting (arXiv:2304.08467) — prompt → few cached activations; no replay,
  no ablation.

**Ablation/perturbation of internal state as the training signal — not
found:**
- KV-cache eviction *self-distillation* (RestoreKV arXiv:2608.01247 and
  kin): reduced-cache student vs full-cache teacher, KL — structurally the
  nearest analogue, but token-slot eviction for serving efficiency, no
  CL/sleep framing, not a recurrent state, not read-direction-targeted.
- Counterfactual-erasure RL for state commitment (arXiv:2606.05201):
  erasure-vs-keep comparison as signal, but reward-based, scratchpad
  tokens, transformer.
- Direction-ablation / "abliteration" (interpretability): subtracting a
  direction is the intervention, never a distillation trigger.
- Tadros et al. 2022, *Sleep-like Unsupervised Replay Reduces Catastrophic
  Forgetting*, Nature Comms — sleep-framed Hebbian replay in a spiking
  net; thematically resonant, mechanistically distant.

**Sleep-framed LLM consolidation — two 2026 preprints checked in full:**
- *Do Language Models Need Sleep? Offline Recurrence…* (arXiv:2605.26099,
  already cited in `consolidation_null.py`): sleep = extra recurrent
  passes updating fast weights; no ablation, no self-generated replay.
- *Language Models Need Sleep: Learning to Self-Modify and Consolidate
  Memories* (arXiv:2606.03979): **closest paper overall** — has both a
  dream phase (RL-generated curriculum) and a distillation phase — but the
  target is a frozen teacher's outputs, never state-ablation-derived, and
  it's transformer/MoE-MLP, not an SSM state.

## 3. Where our loop sits

- **Well-trodden (cite as scaffolding):** dreaming/generative replay
  (Robins → Shin → SSR/SEAL) and context-into-weights distillation
  (Snell → Cartridges). Arm A of the 08-05 A/B is deliberately *the field
  method* built from these parts.
- **Apparently novel (lead with this in any writeup):** the erase-defined
  KL target — ablating a recurrent SSM state along the token's own read
  query (deflated, γ=1) and training weights to close exactly that gap,
  inside a sleep loop. Nearest neighbors each hold one half
  (KV-eviction self-distillation; 2606.03979), none the fusion. Sweep 1
  independently found the read-triggered-erase half unexplored in
  recurrent state.
- Claim nothing externally until the A/B returns: the operator is trivial;
  the contribution, if any, is the measured frontier (07-30 §6's honesty
  rule carries over).
