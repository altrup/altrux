# mamba2_780m_memory

## Source

[`state-spaces/mamba2-780m`](https://huggingface.co/state-spaces/mamba2-780m) — the same 780M-parameter Mamba2 backbone as [`mamba2_780m_continuous_learning`](../mamba2_780m_continuous_learning/README.md), wrapped with a trainable long-term memory subsystem: an addressable associative store updated by the gated delta rule, fed by a Titans-style nonlinear front-end that decides what's worth writing. The memory subsystem (front-end, gate projections) is new and trained from scratch with full gradients. The backbone itself is **not** frozen — it's adapted via LoRA, on the theory that a fully frozen backbone is unlikely to integrate a memory signal injected straight into its SSM state well, since none of its own weights ever get a chance to adjust to that new input. Letting LoRA nudge the backbone's own projections should help it learn to actually use what the memory injects, rather than just tolerate it.

Backbone shape: `d_model=1536`, `n_layer=48`, `d_inner=3072` (expand 2), `nheads=48`, `headdim=64`, `d_state=128`, `ngroups=1` (B/C shared across heads). Per head, Mamba2's own SSM state is a 64×128 matrix, written with a rank-1 outer product and read via the C projection. At injected layers (see below), the memory subsystem's gated-delta rule is merged directly into this same state tensor — not a side accumulator added on top — so memory shares both the *form* (rank-1/low-rank associative writes, projection-based reads) and the actual storage with Mamba2's own state.

## Tokenizer

`EleutherAI/gpt-neox-20b`, same as the other Mamba2 models in this repo — the checkpoint ships without its own tokenizer, so the GPT-NeoX-20B tokenizer (the standard pairing from the original Mamba training recipe) is used here too.

## LoRA target modules

`["in_proj", "out_proj"]` — the same choice as the other Mamba2 models in this repo (Mamba2's analogue of a transformer's q/k/v/o projections). Plain full-precision LoRA, no 4-bit quantization — this 780M backbone fits this project's dev GPU (8GB) comfortably without it.

The memory subsystem itself (front-end, gate projections) stays outside LoRA either way — it's trained with full gradients from a random init, since it has no pretrained weights to adapt.

## Special tokens

`USER_OPEN`/`ASST_OPEN` (`"[USER]"`/`"[ASSISTANT]"`), registered as tokenizer special tokens the same way as the other models in this repo, so each role marker is a single atomic token id. Bare, no trailing space — callers append a literal `" "` separator explicitly.

## `Model` wrapper quirks

The wrapper runs the frozen Mamba2 stack with one shared memory block spliced in, rather than per-layer memory blocks. Per-token data flow:

```
token t
  │
  ▼
embedding
  │
  ▼
layers 0..15 ───────────────────────────────────────────────► (unmodified Mamba2 mixer step)
  │
  ▼
layer 16: ssm_state = ssm_state·dA + dBx                  (Mamba2's own decay+write)
          ssm_state = clear·(ssm_state - β·(ssm_state·key)key) + β·p·key   (gated-delta merge, signals from token t-1's o_t)
          y = C·ssm_state ; persisted ssm_state IS the merged one
  │
  ▼
layer 17 ─────────────────────────────────────────────────────► (unmodified, odd layers have no injection)
  │
  ▼
  ⋮ (even layers 18, 20, ..., 30 each merge their own signals the same way as layer 16)
  │
  ▼
layer 32: same merge as above, using signals from token t-1's o_t
  │
  ├──► residual (pre-layer-32 residual stream, NOT affected by layer 32's own merge above)
  │       │
  │       ▼
  │     Stage 1: Titans front-end
  │       write: M_t = (1-α)M_{t-1} + (η·S_{t-1} - θ·∇L),  L = ‖M(k)-v‖²
  │       read:  o_t = M_t(q_t)              ──┐
  │       surprise = ‖M(k)-v‖² (detached)  ────┤ broadcast to all 16 injected layers (incl. layer 32 itself)
  ▼                                            │
layer 33 (no injection) ◄───────────────────────┘
  │
  ▼
layer 34: gated-delta merge using THIS token's o_t (already computed at layer 32) ──► ⋮ ──► layer 46
  │
  ▼
layer 47 (no injection) ──► norm_f ──► lm_head ──► logits_t
```

Note the subtlety at layer 32: the merge at layer 32 (using signals derived from `o_t` computed on the *previous* token) happens first, then the residual entering layer 32 — unaffected by that merge — is what Stage 1 reads. So this token's `o_t` isn't visible to layer 32's own merge until token *t+1*; there's no same-token cycle. Layers 34–46 get this token's `o_t` immediately (same token, later in the layer stack); layers 16–30 (and 32 itself) always lag by one token.

### Sequential, per-token execution (not chunked)

Because the gated-delta accumulator at layer *i* depends on the memory read at `READ_LAYER`, and that read for token *t* must be visible to *later* layers of the *same* token while only being visible to *earlier* layers on the *next* token, the backbone can't run through Mamba2's fused/chunked parallel-scan kernels — those process a whole sequence in one kernel call and don't expose a per-token, pre-readout hook. `Model.forward` therefore loops over time explicitly, replicating Mamba2's own incremental-decode arithmetic (`_mixer_step`) for every layer, every token — the same asymptotic cost as autoregressive decoding, just paid during training too. This is materially slower than the library's native chunked training path; revisit if it becomes a bottleneck (e.g. a custom chunked kernel that exposes the pre-readout state).

### Read location

The memory reads the residual stream once, at roughly 2/3 depth (layer ~32 of 48, `READ_LAYER`). Earlier layers produce weak keys (too little semantic content yet); later layers are already collapsed toward next-token prediction. This depth is exposed as a hyperparameter and is the first thing to sweep if recall is weak.

### Stage 1 — Titans-style neural front-end (shared, single instance)

Three dedicated projections turn the layer-32 residual into query/key/value (`q`, `k`, `v`); a small deep MLP `M` (~2 layers, wide hidden) *is* the memory — its weights are the memory content, not an activation. Writing is a test-time gradient step on the associative loss `L = ‖M(k) - v‖²`:

```
S_t = η_t·S_{t-1} - θ_t·∇L
M_t = (1 - α_t)·M_{t-1} + S_t
```

with `η`/`θ`/`α` produced by small data-dependent sigmoid projections. Reading is a pure forward pass, `o_t = M_t(q_t)` (a 1536-dim vector), with no weight update. The gradient magnitude from the write step ("surprise") is exported downstream to modulate Stage 2's write strength.

### Stage 2 — gated-delta merge directly into Mamba2's own SSM state (one signal-generator per injected layer)

Each injected layer derives its own write signals from `o_t` through a dedicated `1536 → 128 → expand` bottleneck: a per-head value `p_t` (64-dim × 48 heads = 3072), a shared 128-dim key/address `B_t`, a write strength `beta_t = sigmoid(W_beta·o_t + surprise_t)`, and a clear gate `clear_t = sigmoid(W_clear·o_t)` initialized near 1. Unlike an earlier version of this design, there is **no separate accumulator** — the gated-delta rule is applied directly to Mamba2's own per-layer `ssm_state`, immediately after Mamba2's own decay+write and before the `C` readout, in `Model._mixer_step`:

```
ssm_state_t = ssm_state_t · dA + dBx                                      (Mamba2's own update, unmodified)
ssm_state_t = clear_t·(ssm_state_t - beta_t·(ssm_state_t·B_t)·B_t^T) + beta_t·p_t·B_t^T   (gated-delta merge)
y_t = C_t · ssm_state_t                                                   (Mamba2's own readout, unmodified)
```

and the *merged* `ssm_state_t` — not a separate copy — is what gets persisted to the next token. The delta term overwrites only the address being written and leaves other content intact (content-addressed forgetting); `clear_t` is a data-dependent global wipe for topic/segment boundaries, initialized near 1 so it never forces decay on its own. `beta_t` is initialized near 0 (`beta_proj.bias = -4`) so the merge is a no-op at init, matching `clear_t`'s near-1 init — together they mean the wrapped backbone behaves exactly like the unmodified pretrained model until these gates learn otherwise. This stage is essentially a near-reference implementation of the accumulator in Yang, Kautz & Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025, arXiv:2412.06464) — applied to the backbone's own state rather than a side accumulator.

Because memory now lives in the same tensor Mamba2 itself decays via `A` each step, memory content is subject to the backbone's own (frozen, pretrained-for-its-own-purposes) per-head decay rate in addition to the gated-delta dynamics — there's no longer an independent persistence mechanism insulated from the backbone's recurrence. That's a deliberate tradeoff versus the earlier separate-accumulator design (see git history): simpler (one state per layer, not two), but memory's effective retention is now coupled to whatever decay rate `A` already encodes for that head, which was learned for the backbone's own purposes, not for long-term memory retention.

### Injection

Mamba2's existing `C` projection reads the merged state directly, so no separate reader module is needed — "injection" here means *which* layers run the gated-delta merge above, not an additive side-channel.

Only a subset of layers carry a merge point — every even layer from 16 to 46 inclusive (`INJECTED_LAYERS = range(16, 48, 2)` in `model.py`), 16 of 48 layers total:

```
16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46
```

`READ_LAYER` (32) falls inside this range, so layer 32 both merges memory into its own state *and* is read by Stage 1 — see the diagram note above for why that doesn't create a same-token cycle. The injected-layer set is itself a hyperparameter, not architecturally required to be this exact stride/range — it (and `READ_LAYER`) follows this architecture's general depth/coverage proportions (roughly 2/3 depth for `READ_LAYER`, roughly the last third of layers for `INJECTED_LAYERS`).

### Parameter budget

~17M trainable against the 780M frozen backbone (~2.2%), excluding LoRA adapter params (which scale with `--lora-rank` and aren't fixed by the architecture) — Titans front-end (QKV + knob projections) ~7.1M, per-layer injection bottlenecks ~9.8M (16 layers × ~0.61M each, including that layer's gate projections).

### Open architectural question

The Titans front-end and the gated-delta accumulator are both gradient-based associative memories (the delta rule is one step of gradient descent on the same associative objective), so they may be partially redundant. Before committing to the full two-stage stack, the plan is to ablate a gated-delta-only variant (plain nonlinear projection straight to `p`/`B`, no Stage 1) against the full stack, and keep Stage 1 only if it earns its cost in long-context recall.

This model has no `<revise>`-tag behavior — that's specific to [`mamba2_780m_continuous_learning`](../mamba2_780m_continuous_learning/README.md). The two models target different problems (revision-on-feedback vs. long-context memory) on the same backbone.

## Training data

The whole point of this model is long-range recall, so it needs long-session training data, not the short multi-turn chats the other models in this repo use. See [`sft/README.md`](../../sft/README.md#long-context-data-mamba2_780m_memory) — `MODEL_NAME=mamba2_780m_memory make data-memory` builds a mix of real long conversations ([`THUDM/LongAlign-10k`](https://huggingface.co/datasets/THUDM/LongAlign-10k)) and synthetic needle-in-haystack recall QA ([`RMT-team/babilong`](https://huggingface.co/datasets/RMT-team/babilong)), since long text alone doesn't force the memory gates to actually do anything — only tasks that depend on far-back information do.
