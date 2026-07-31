# Research note — 2026-07-30: erase-on-read in the SSM state itself

Not a run log, not standing direction. Algebra + literature scan prompted by
a session question: *decay-on-read currently operates on M (the Titans
neural memory, a gradient step toward `(q, 0)`). Can the same operation be
performed on Mamba2's own `ssm_state` — and if it can, is M needed at all?*

The algebra answers cleanly and in our favour. The literature says the
operator is three months old and belongs to someone else; the loop we would
wrap around it does not.

**Confidence markers:** [V] = verified against the paper's own text (fetched
this session); [S] = search snippet only, NOT read — pointer, verify before
relying. Companion standing-direction note:
`DISCUSSION-20260730-ssm-consume-on-read-and-m-necessity.md`.

## 1. The SSM state is a linear associative memory (why this is possible)

Per head, `ssm_state` has shape `[headdim=64, d_state=128]`. Reading
`models/mamba2_2_7b_memory/model.py:_mixer_step`:

```python
dBx       = einsum("bh,bn,bhp->bhpn", dt, B, x_h)   # outer product x ⊗ B
ssm_state = ssm_state * dA + dBx
y         = einsum("bhpn,bn->bhp", ssm_state, C)    # contract n with C
```

Every token writes a rank-1 outer product: the value `x` on the `p` axis,
`B` on the `n` axis. Unrolling and reading out:

```
S_t    = Σ_{s≤t} λ_{s→t} · dt_s · x_s B_sᵀ          λ_{s→t} = Π dA_r
y_t[p] = Σ_{s≤t} λ_{s→t} · dt_s · (B_s · C_t) · x_s[p]
```

`(B_s · C_t)` is a dot-product attention score. **`B` is the key, `C` is the
query, `x` is the value, and the `n` axis is the address space.** This is
the whole content of Mamba2's state-space-duality claim: the recurrence *is*
linear attention with decay, and `ssm_state` *is* a KV cache summed into a
fixed-size matrix rather than stored per-token.

Linear **in the state**, not globally — `dA`, `B`, `C`, `x` are all computed
nonlinearly from the input. That is exactly the property the erase needs,
and it survives the network being nonlinear everywhere else.

Capacity, stated honestly: with `n = 128`, at most 128 linearly independent
keys can coexist without destroying information. Below that, values are
recoverable in principle (`V = S K(KᵀK)⁻¹`) but *not by the architecture's
own read*, which is a single matvec and is contaminated by key overlap.
Beyond 128, information is genuinely destroyed. Our own measurement puts the
practical ceiling far lower — dead by ~192 tokens of dense interference
(`RESEARCH-20260724-local-diagnostics.md` §1).

## 2. Consume-on-read: the operator

The information a token "used" is the projection of `S` onto `C_t`. Remove
exactly that:

```
S ← S (I − γ_t ĉ_t ĉ_tᵀ)          ĉ_t = C_t / ‖C_t‖,  γ_t = sigmoid(W·h_t)
```

`ĉĉᵀ` is the rank-1 projector onto the query; `(I − γĉĉᵀ)` contracts the
address space along it. At `γ = 1` a re-read with the same query returns
exactly zero.

**This is the closed form of the gradient step we already do on M.** Take
the associative objective with target zero on a *linear* memory:

```
L   = ½‖S c‖²
∇_S = (S c) cᵀ
S   ← S − θ (S c) cᵀ  =  S (I − θ c cᵀ)
```

One gradient step toward `(q, 0)` **is** the rank-1 projection, scaled by
`θ‖c‖²`. On M the step is approximate because M is a nonlinear MLP and you
approach the fixed point iteratively. On `ssm_state` you land on it in one
shot, in closed form, with no gradient computation at all. `γ` is the
lossiness knob — the exact analogue of the learning rate, bounded to `(0,1)`
by construction so the unstable `θ‖c‖² > 2` regime is unreachable.

What the erase does to stored content, term by term:

```
x_s B_sᵀ (I − γ ĉĉᵀ) = x_s (B_s − γ(B_s·ĉ) ĉ)ᵀ
```

Values are untouched; **keys rotate away from the query**, each losing
exactly its query component. An item fully retrieved has its key annihilated
and vanishes from the matrix; an item orthogonal to the query is preserved
bit-for-bit. Forgetting in proportion to retrieval falls out of the algebra
rather than being engineered — this is the property `DISCUSSION-20260725`
§2 had to *add* to M by gating on the KL signal.

### Implementation notes

- **Normalize `C`.** Mamba2's `C` is unnormalized (no softmax anywhere), so
  without dividing by `‖C‖²` the erase strength rides on query magnitude.
- **Reuse the existing read.** `S·C` is already computed at
  `_mixer_step`'s `y = einsum(...)`. The erase is one outer product on top:
  `S -= γ · (S C) Cᵀ / ‖C‖²`. No extra matvec.
- **Use the pre-`D` `y`.** `y += D·x_h` is a skip path that never entered
  the state; subtracting it would remove something that was never there.
- **Order is forced.** If the erase address is the readout query, the erase
  *must* follow the read — erasing first deletes what you were about to
  retrieve (`y = (1−γ)·SC`, and at `γ=1` the layer reads zero).
- **Open: pre- vs post-write.** The current token's `dBx` is in `S` when
  `y` is read, so it did contribute — consuming it is consistent. But it
  means the freshest write is partially erased on arrival, and recent
  context is what Mamba leans on hardest. Cheap to flag; no principled
  argument either way.
- **Gate init.** `sigmoid(-4) ≈ 0.018` — starts as a near-no-op and has to
  earn its erasure, matching the existing `beta_proj.bias = -4` convention.
  Per-head `γ` costs ~48 params per injected layer.

## 3. Prior art

### Direct: the operator is published, twice, in the last three months

- **Erase-then-Delta Attention (EDA), arXiv 2606.26560** [V]. Update:
  `S_t = (I − β_t k_t k_tᵀ)(I − γ_t e_t e_tᵀ) D_t S_{t−1} + β_t k_t v_tᵀ`.
  The middle factor is our operator exactly, derived from the same objective
  (`L^erase = ½‖Ŝᵀe‖²`, one gradient step at rate `γ`). Differences: `e_t`
  is a **separately learned** address from a per-head rank-16 low-rank
  projection, L2-normalized; the erase fires **before the write**, and
  readout is last; motivation is LM quality at 2.5B dense / 25B-A2.8B MoE.
- **Gated DeltaNet-2, arXiv 2605.22791** [S]. Same family; splits the delta
  rule's coupled erase/write into channel-wise gates, but the erase
  direction is still built from the write key.
- **Gated DeltaNet, arXiv 2412.06464** — the accumulator our Stage 2 already
  near-implements.

Gains are small: EDA 28.44 vs GDN-2 28.14 average at 2.5B dense [V].

### The address-tied-to-read precedent is older and outside linear attention

- **DNC free gates** (Graves et al., Nature 2016) [S]. Retention vector
  `ψ_t = Π_i (1 − f_t^i · w_{t−1}^{r,i})` — the address is the *previous
  read weighting*, magnitude is a learned sigmoid, fires after the read.
  Our mechanism in slot-memory form, ten years old. Our contribution over it
  is the continuous rank-1 form in a matrix-valued SSM state, not the
  concept. **Make this comparison ourselves before a reviewer does.**
- **SOB-CS removal** (Oberauer & Lewandowsky) [S]. Two-layer Hebbian
  outer-product memory — structurally our `S`. Removal is: cue, retrieve,
  then Hebbian anti-learning of that binding. The closest existing formal
  statement of the idea anywhere, and it is from psychology.

### Cognitive framing — do NOT cite retrieval-induced forgetting

RIF weakens *non-retrieved competitors* while the retrieved item is
strengthened (the testing effect) [S]. It predicts the opposite of
consume-on-read; cited carelessly it hands a reviewer the rebuttal.
Oberauer's **removal from working memory** — active, cue-addressed
unbinding — is the correct hook.

## 4. Novelty verdict

Blunt:

- **The operator** — published (EDA, GDN-2). Zero novelty.
- **Erase address = readout query** — not published in the linear-attention/
  SSM line as far as a ~10-formulation search found, but it is the direct
  continuous analogue of DNC free gates. Defensible as a novel *address
  choice*; not defensible as a novel idea.
- **Exact closed form of the `(q, 0)` step on a linear state** — no paper
  states it. A two-line derivation; framing, not a result.
- **The gen → erase → consolidate loop** — no prior work found. TTT-E2E
  (2606.21803) [S], In-Place TTT (2604.06169) [S], Agentic TTT
  (2607.03441) [S] all update fast weights per token or chunk; none clear
  the state of what was consumed as a step in the loop. **This is the
  contribution. The architecture is the least novel part.**

## 5. Evidence bearing on whether it works

**Against — the one real data point.** EDA's memory probe reports a
"collateral perturbation score": 0.064 for the learned address, 0.115
shuffled, 0.143 random, 0.223 for `e_t = k_t` (lower = better) [V], and
states that a smaller score means "the chosen address changes the currently
readable state less than an alternative address" [V]. An optimizer given a
free choice moved the erase away from disturbing the readable state. Under
an LM objective, expect `γ → 0` on a query-tied erase.

Two things blunt this: EDA never reports `cos(e_t, q_t)` at all — the only
orthogonality statistic is against the *write key* (`|cos(e_t,k_t)| ≈ 0.105`
[V]) — and their objective is LM perplexity, not ours.

**For, weakly.** For nearly-persistent heads (`ᾱ ≥ 0.9`) EDA's independent
erase contributes 69.1% of contraction strength, ~3.0× the same-address
correction [V] — a learned erase gate does not collapse to zero when it has
work to do. Transfer to a query-tied gate is weak but it is the only
gate-liveness data in either paper.

**Untested, not disfavoured.** No published ablation tests query-tied
erasure. EDA's counterfactuals cover learned / shuffled / random /
key-collapsed — the query is not among them [V]. GDN-2 ablates channel-wise
vs scalar, never the address [S]. So the design is unexplored rather than
refuted.

**Unmeasured everywhere.** Neither paper reports `γ` saturation or collapse
statistics. Gate collapse is not measured-and-benign; it is simply not
measured.

## 6. The obvious failure mode

Language re-reads the same content constantly — pronouns, sustained topics,
repeated entities. Unconditional consume-on-read plausibly breaks fluent
generation outright. That is what the learned `γ_t` is for, and it makes
the first experiment obvious: **does a gated version learn to be selective,
or does it just learn `γ ≈ 0` and switch itself off?** Under an LM
objective EDA has effectively already answered that. It must be run under
the CL objective, where erasing consumed material is principled (data
hygiene for the next gradient step) rather than perverse.

EDA concedes the same shape of cost in its own limitations: the independent
erase "reduces raw write-key recall" and is "a conditional cleanup
mechanism rather than a uniform improvement to memory fidelity" [V].

## 7. What to read

1. **EDA, arXiv 2606.26560** — nearest neighbour, main baseline; §4.5's
   counterfactual probes are the ablation table to extend with a
   query-tied row. Kernels promised at `github.com/QwenLM/FlashQLA` [V].
2. **Graves et al., DNC, Nature 2016** — the free gate; address it up front.
3. **Oberauer & Lewandowsky, SOB-CS removal** — correct cognitive framing,
   replaces RIF.
4. **Gated DeltaNet-2, arXiv 2605.22791** — second baseline; channel-wise
   axis is orthogonal to ours and composable.
5. **TTT-E2E, arXiv 2606.21803** — closest framing of LM-as-continual-
   learning; the loop ours sits inside.
6. **Beyond Perplexity (TTT deployment-memory eval), arXiv 2607.00368** —
   already cited in the data-structure note; the evaluation protocol to
   adopt rather than invent, given we are explicitly not optimizing
   perplexity.

## 8. Caveats on this note

- EDA numbers come from the v1 HTML (`arxiv.org/html/2606.26560v1`); the PDF
  fetch returned truncated content. Verify against the PDF before quoting
  them anywhere public.
- Absence-of-prior-work claims rest on ~10 search formulations, not an
  exhaustive sweep. In a subfield publishing this fast, "no hits" is weak
  evidence. A same-idea preprint from the last few weeks under different
  terminology could easily have been missed.
- Items marked [S] were not read. The DNC retention-vector formula and the
  SOB-CS mechanism in particular are load-bearing for §3 and §4 and should
  be verified from source before either appears in a design doc.
