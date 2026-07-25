# mamba2_2_7b_memory

## Source

[`state-spaces/mamba2-2.7b`](https://huggingface.co/state-spaces/mamba2-2.7b) — the 2.7B-parameter Mamba2 backbone, wrapped with a trainable long-term memory subsystem: an addressable associative store updated by the gated delta rule, fed by a Titans-style nonlinear front-end that decides what's worth writing. The memory subsystem (front-end, gate projections) is new and trained from scratch with full gradients. The backbone itself is **not** frozen — it's adapted via LoRA, on the theory that a fully frozen backbone is unlikely to integrate a memory signal injected straight into its SSM state well, since none of its own weights ever get a chance to adjust to that new input.

Backbone shape: `d_model=2560`, `n_layer=64`, `d_inner=5120` (expand 2), `nheads=80`, `headdim=64`, `d_state=128`, `ngroups=1` (B/C shared across heads). Per head, Mamba2's own SSM state is a 64×128 matrix, written with a rank-1 outer product and read via the C projection. At injected layers (see below), the memory subsystem's gated-delta rule is merged directly into this same state tensor — not a side accumulator added on top — so memory shares both the *form* (rank-1/low-rank associative writes, projection-based reads) and the actual storage with Mamba2's own state.

## Design goal — a three-tier memory hierarchy

The intended division of labor between the model's three stores:

| Tier | Store | Holds | Regime |
|---|---|---|---|
| Working | Mamba2's own SSM state | everything about the current context | recent *or* few — a fixed-size linear buffer that decays every token and can't choose not to forget |
| Episodic | the neural memory `M` | *selected* residue that must survive context turnover — gist, not verbatim tokens | high fact-count and/or post-reset survival; surprise-gated selective writes, content-addressed reads, persistence set by `α` |
| Semantic/procedural | the trained parameters | skills and world knowledge | consolidated over training, static at inference |

The boundary between the first two is capacity and interference, **not duration**: probes on a trained checkpoint (2026-07-17, `sft/probe_recall.py`) showed the SSM alone carries a *single* fact at ~96% out to 12k tokens, but collapses to chance at 128+ competing facts by 2.5k tokens. Training data must therefore demand recall the SSM structurally cannot provide — many competing facts, and recall across SSM-state resets — or the optimizer will keep routing everything through the backbone and suppress `M` (observed: alpha climbing, o_t_norm shrinking, ablation deltas ~0).

## Tokenizer

`EleutherAI/gpt-neox-20b`, same as the other Mamba2 models in this repo — the checkpoint ships without its own tokenizer, so the GPT-NeoX-20B tokenizer (the standard pairing from the original Mamba training recipe) is used here too.

## LoRA target modules

`["in_proj", "out_proj"]` — the same choice as the other Mamba2 models in this repo (Mamba2's analogue of a transformer's q/k/v/o projections). Plain full-precision LoRA, no 4-bit quantization — the 2.7B backbone in bf16 is ~5.4 GB, comfortable on a cloud A100/H100 (80 GB) without QLoRA.

The memory subsystem itself (front-end, gate projections) stays outside LoRA either way — it's trained with full gradients from a random init, since it has no pretrained weights to adapt.

## Special tokens

`USER_OPEN`/`ASST_OPEN` (`"[USER]"`/`"[ASSISTANT]"`), registered as tokenizer special tokens the same way as the other models in this repo, so each role marker is a single atomic token id. Bare, no trailing space — callers append a literal `" "` separator explicitly. Their embedding rows are initialized to the mean of their BPE-spelling rows and trained through the wrapper's `MarkerDelta` while the rest of the tied embedding/`lm_head` table stays frozen — see `models/common.py` and `models/mamba2_780m/README.md` for the mechanism.

## `Model` wrapper quirks

The wrapper runs the frozen Mamba2 stack with one shared memory block spliced in, rather than per-layer memory blocks. Per-token data flow:

```
token t
  │
  ▼
embedding
  │
  ▼
layers 0..21 ───────────────────────────────────────────────► (unmodified Mamba2 mixer step)
  │
  ▼
layer 22: ssm_state = ssm_state·dA + dBx                  (Mamba2's own decay+write)
          ssm_state = clear·(ssm_state - β·(ssm_state·key)key) + β·p·key   (gated-delta merge, signals from token t-1's o_t)
          y = C·ssm_state ; persisted ssm_state IS the merged one
  │
  ▼
layer 23 ─────────────────────────────────────────────────────► (unmodified, odd layers have no injection)
  │
  ▼
  ⋮ (even layers 24, 26, ..., 40 each merge their own signals the same way as layer 22)
  │
  ▼
layer 42: same merge as above, using signals from token t-1's o_t
  │
  ├──► residual (pre-layer-42 residual stream, NOT affected by layer 42's own merge above)
  │       │
  │       ▼
  │     Stage 1: Titans front-end
  │       write: M_t = (1-α)M_{t-1} + (η·S_{t-1} - θ·∇L),  L = ‖M(k)-v‖²
  │       read:  o_t = M_t(q_t)              ──┐
  │       surprise = ‖M(k)-v‖² (detached)  ────┤ broadcast to all 21 injected layers (incl. layer 42 itself)
  ▼                                            │
layer 43 (no injection) ◄───────────────────────┘
  │
  ▼
layer 44: gated-delta merge using THIS token's o_t (already computed at layer 42) ──► ⋮ ──► layer 62
  │
  ▼
layer 63 (no injection) ──► norm_f ──► lm_head ──► logits_t
```

Note the subtlety at layer 42: the merge at layer 42 (using signals derived from `o_t` computed on the *previous* token) happens first, then the residual entering layer 42 — unaffected by that merge — is what Stage 1 reads. So this token's `o_t` isn't visible to layer 42's own merge until token *t+1*; there's no same-token cycle. Layers 44–62 get this token's `o_t` immediately (same token, later in the layer stack); layers 22–40 (and 42 itself) always lag by one token.

### Sequential, per-token execution — only where it's actually required

Because the gated-delta accumulator at layer *i* depends on the memory read at `READ_LAYER`, and that read for token *t* must be visible to *later* layers of the *same* token while only being available to *earlier* layers on the *next* token, per-token injection can't run through Mamba2's fused/chunked parallel-scan kernels — those process a whole sequence in one kernel call and don't expose a per-token, pre-readout hook. But injection only fires on a memory-window's closing token (see "Memory-window batching" below), so only that one token per window still needs sequential treatment.

`Model.forward` dispatches between two implementations, chosen by feature detection (`fused_kernel_usable`: CUDA, not ROCm/HIP, `causal_conv1d` importable), not a manual flag:

- **`_forward_manual`** — the original all-manual loop: every layer, every token, replicating Mamba2's own incremental-decode arithmetic (`_mixer_step`) sequentially. The same asymptotic cost as autoregressive decoding, paid during training too. This is the only path available on this project's local ROCm dev box (broken `causal-conv1d`/Triton — see root `CLAUDE.md`), and always correct everywhere else too, just slower there than the alternative below.
- **`_forward_fused`** — restructures the backbone loop from token-major to layer-major (mirroring `mamba_ssm`'s own full-sequence `MixerModel.forward`). The 43 layers never in `INJECTED_LAYERS` run through `mamba_ssm`'s native fused/chunked kernel (`_mixer_span`, wrapping `causal_conv1d_fn` + `mamba_chunk_scan_combined`) for the *entire* chunk in one call — no per-token dependency ever exists for them. The 21 injected layers (including `READ_LAYER`) run per memory-window: one fused-kernel call over the window's `window - 1` non-closing tokens, then one manual `_mixer_step` call for the window-closing token. `READ_LAYER`'s own read/write additionally needs every token's residual in the window, not just the closing token's — `_NeuralMemory.read_windowed`/`.surprise_windowed` compute that for the whole window in one batched call (no sequential loop), preserving the exact same surprise-weighted pooling and same-token/next-token ordering subtlety described above.

This is genuinely the large training-cost reduction that motivated the memory-window work in the first place — it's what lets a chunk's non-injection tokens skip the manual per-token loop entirely, not just batch the memory subsystem's own write cadence. `causal-conv1d` can't be installed or exercised on this project's local ROCm dev box at all (see root `CLAUDE.md`), so this path can only be smoke-tested on a real CUDA host (e.g. the rented H100) — see `docs/superpowers/specs/2026-07-02-chunked-memory-injection-design.md`'s "Testing/rollout" section. The first such test (2026-07-02) surfaced and fixed a real bug: `_mixer_span`'s `new_conv_state` (built via `torch.cat`) loses the channel-last memory layout (`stride(1) == 1`) that `causal_conv1d_fn`'s CUDA kernel hard-requires for `initial_states` — `mamba_ssm`'s `allocate_inference_cache` produces that layout, so only a slot's *first* chunk worked before this fix; every chunk after it crashed with `Expected initial_states.stride(1) == 1`. Fixed by forcing `conv_history` back into channel-last layout on read in `_mixer_span` rather than relying on the incoming tensor's layout. Re-verify end-to-end on the H100 before trusting this path for a full paid training run.

### Memory-window batching

Both the Titans front-end's *write* (`_NeuralMemory.write`) and the gated-delta *injection* into `ssm_state` are decoupled from firing every token via `Model.set_memory_window(w)` / `sft/train.py`'s `--memory-window` flag:

- **Write**: instead of taking one test-time gradient step every token, `_NeuralMemory.write` buffers `w` tokens' worth of key/value/knob signals and takes one consolidated step per window, following the Titans paper's own chunk-size-`b` formulation (`b ≥ 1`, arXiv:2501.00663 §3.2) — `w=1` (the default) reproduces the original exact per-token update, not a separate code path. (The window-*internal* trajectory isn't reconstructed exactly the way Titans' own parallel scan does — see the design spec's "Non-goals" for why that's deferred, not a small addition.)
- **Injection**: the gated-delta merge into `ssm_state` only fires on the token that closes each window, using a **surprise-weighted pooling** of every token's `o_t`/`surprise` in that window (softmax over `surprise`, so the most-surprising token in the window dominates the merge) rather than just the closing token's own raw read. For the other `w - 1` tokens, every injected layer gets no merge at all — `ssm_state` evolves purely under Mamba2's own dynamics. `w=1` makes the softmax degenerate to a single element (weight 1.0), reproducing today's exact per-token injection.
- **Read** (`_NeuralMemory.read`/`.surprise`) stays per-token regardless of `w` either way, since those are just forward passes against whatever weights are currently active, not a sequential recurrence — the per-token values are what feed the pooling above.

`w=1` at inference (always, regardless of what `w` a checkpoint trained with) is what keeps training (any `w`) and inference mathematically consistent — inference is already token-by-token regardless of training's window size, so there's no reason to expose it to behavior training never produced.

This was the prerequisite for letting the backbone use a fused kernel for the spans between window boundaries, which is now also implemented — see "Sequential, per-token execution" above for `_forward_fused`/`_mixer_span`. See `docs/superpowers/specs/2026-07-02-chunked-memory-injection-design.md` for the full design, what's still deferred, and why. `w` must evenly divide `--chunk-len` (a window can't span the chunk boundary where BPTT gets truncated) and has no principled default above 1 yet — sweep small values cheaply (e.g. via `make smoke-test`) before committing real training hours to one.

### Gradient checkpointing

`Model.set_grad_checkpoint(enabled, block=GRAD_CHECKPOINT_BLOCK)` (reached from `sft/train.py` via each model's `train_hooks.set_grad_checkpoint`, per data slice) splits a chunk into `block`-token blocks and runs each under `torch.utils.checkpoint`, threading `MemoryState` across them: only block-boundary state is retained, and a block's own graph is recomputed when backward reaches it. This is what makes a long `--chunk-len` (512 on the cram slices) affordable — see `GRAD_CHECKPOINT_BLOCK` in `model.py` for the retained-vs-recompute arithmetic behind the default of 64. It wraps `forward`'s path dispatch, so `_forward_manual` and `_forward_fused` are covered identically and neither knows about it. Disabled (the default, and any `no_grad` call) is an exact no-op; nothing about it reaches a checkpoint file.

Two things this required, both non-obvious:

- **The write step's gradient is written out in closed form** (`_NeuralMemory._write_grads`) instead of coming from a nested `torch.autograd.grad(..., create_graph=True)`. A nested `autograd.grad` runs as its own autograd graph task, and a `torch.utils.checkpoint` frame can only serve a foreign graph task by recomputing itself in full — so every window's write replayed its whole block *during the forward pass*, and checkpointing raised peak memory and runtime instead of lowering them (both alternatives were measured before this was written; pinning the tensors instead, via identity saved-tensor hooks, removed the replays but retained everything and saved nothing). With the closed form it is ordinary first-order autograd over six einsums, and checkpointing frees and recomputes it like anything else. `models/tests/test_grad_checkpoint.py` pins it against `autograd.grad`, including through a second differentiation — the path that carries `k_proj`/`v_proj`'s gradient.
- **M, not the backbone, is what checkpointing has to shrink.** Each window close produces a fresh `w1/b1/w2/b2`, a fresh momentum, and a gradient of the same shape (~225 MB per batch slot per window at the 780m geometry), against ~2 MB per token of backbone activation on the fused path. Measured on a scaled-down model at 32 windows: peak 2.31 → 0.79 GiB.

### Sleep — backbone-state wipe

`Model.sleep_slot(state, slot_idx)` wipes one slot's backbone state (per-layer SSM/conv, plus the `last_o_t`/`last_surprise` read snapshot) to the zeros a fresh sequence starts from, leaving the neural memory (`w1`/`w2`, momentum, drift baselines) untouched. It is the training/eval realization of the deployment "sleep": context turnover that only the episodic store survives. `sft/train.py` triggers it at a dataset's `sleep_positions` offsets (see the episodic-chains design spec); contrast `Model.reset_slot`, which additionally re-randomizes the neural memory and is used only between chains/examples.

### Read location

The memory reads the residual stream once, at roughly 2/3 depth (layer 42 of 64, `READ_LAYER`). Earlier layers produce weak keys (too little semantic content yet); later layers are already collapsed toward next-token prediction. This depth is exposed as a hyperparameter and is the first thing to sweep if recall is weak.

### Stage 1 — Titans-style neural front-end (shared, single instance)

Three dedicated projections turn the layer-42 residual into query/key/value (`q`, `k`, `v`); a small deep MLP `M` (~2 layers, wide hidden) *is* the memory — its weights are the memory content, not an activation. Writing is a test-time gradient step on the associative loss `L = ‖M(k) - v‖²`:

```
S_t = η_t·S_{t-1} - θ_t·∇L
M_t = (1 - α_t)·M_{t-1} + S_t
```

with `η`/`θ`/`α` produced by small data-dependent sigmoid projections (`η` capped at 0.9, `θ` at 0.1, `α` at `ALPHA_CAP` = 1e-4). `α`'s far tighter cap is a deliberate prior, not a tuning detail: BPTT truncation at `--chunk-len` means `α`'s projection only ever receives short-horizon gradient, so the model cannot learn the long-horizon cost of erosion — at a 0.1-scale cap a real run walked `α` up ~20× in an hour and eroded ~80% of `M`'s content within one long example. At 1e-4 the worst-case half-life is ~7k writes (~55k tokens at memory-window 8): sustained decay still compounds into era-level forgetting, but a fast wipe is unreachable, leaving `α` as a stabilizer with a slow leak (bounding `w1`/`w2` drift, reclaiming never-revisited content) while *targeted* forgetting runs through the delta-overwrite path (writing a new value at a key). Reading is a pure forward pass, `o_t = M_t(q_t)` (a 2560-dim vector), with no weight update. The gradient magnitude from the write step ("surprise") is exported downstream to modulate Stage 2's write strength.

The knob projection reads an RMS-normalized copy of the residual (same treatment as `q`/`k`/`v`, for the same reason: the raw residual's rms sits in the tens at `READ_LAYER`, which drives sigmoid pre-activations tens deep into their zero-gradient rails), and is initialized with zero weights plus fixed per-knob biases — `η`/`θ` at the sigmoid midpoint (0.45 momentum decay, 0.05 write step, full gradient), `α` at bias −4 so decay starts near the bottom of its already-small range (~2e-6/window) and forgetting is *learned* rather than the default. A measured fresh-init/checkpoint run of the previous parameterization (`sft/measure_knobs.py`) showed why this matters: `θ` pinned at ~0 (the memory was never written), `η` pinned at its cap, `α` at a random rail per token averaging ~0.04/window — a fast-weight half-life of ~125 tokens against examples tens of thousands of tokens long, with no usable gradient through the saturated sigmoids to recover. The store decayed to ~150× below init scale by step 45 of a real run while `knob_proj`'s trained weights stayed frozen at init.

Before entering the `S_t` update, `∇L` is soft-clipped by norm: `g_soft = GRAD_SCALE · tanh(‖∇L‖ / GRAD_SCALE) · (∇L / ‖∇L‖)`, i.e. direction preserved exactly, magnitude left untouched for typical gradients (`tanh(x) ≈ x` near 0) but smoothly bounded below `GRAD_SCALE` (`model.py`, currently `300`) no matter how large the raw gradient spikes. Capping `η` below 1 alone bounds the momentum recurrence only *given* a bounded per-token gradient -- this clip is what actually bounds the gradient itself, since a spike (most likely early in training, before `η`/`θ` are learned) can otherwise push the memory non-finite in one step regardless of `η`. `300` is not a dimensionality guess (a naive one, ~7211 from treating `w1`/`w2`'s ~52M elements as independent, is wrong -- `L`'s mean reduction over `MEM_DIM` attenuates the gradient too, not just the loss): a direct numeric check of `∇L` at `w1`/`w2`'s init scale gives a combined norm of only ~2 (nearly identical to the 780m model's ~2 despite the larger dims), and the runaway is quadratic in how far `w2` has drifted from that scale (~2754 at 100x drift, ~24765 at 300x) -- see `model.py`'s `GRAD_SCALE` comment for the full derivation. `300` leaves ~150x headroom over the healthy baseline while still intervening well before that drift compounds too far. The pre-clip norm is logged live (`grad_norm` in `extra_log`/`chunk_extra_log`) so this ceiling can be retuned from real training telemetry. Note this clip bounds the *gradient used to update* `w1`/`w2`, not `w1`/`w2`'s own magnitude directly -- nothing stops them drifting large over many windows even with every individual update small. `w1_abs_max`/`w2_abs_max` (`extra_log`) track the largest single weight magnitude in `M` seen since the last optimizer step, watching for exactly that drift directly rather than only inferring it from the clipped gradient. The opposite failure -- memory erased or never written -- has its own `extra_log` stats: `alpha` (write()'s direct weight decay on `M`), `retain_min` (episodic wipes the `retain` mean would hide), `w1_rms_min`/`w2_rms_min` (smallest per-slot RMS of `M`'s weights; a healthy fresh slot sits near its init RMS, `bound/√3` ≈ 0.011 for `w1` / 0.0057 for `w2`, and ~0 means a zeroed slot), and `w1_drift`/`w2_drift` (RMS distance from the per-example random init, i.e. cumulative written content).

### Stage 2 — gated-delta merge directly into Mamba2's own SSM state (one signal-generator per injected layer)

Each injected layer derives its own write signals from `o_t` through a dedicated `2560 → 128 → expand` bottleneck: a per-head value `p_t` (64-dim × 80 heads = 5120), a shared 128-dim key/address `B_t`, a write strength `beta_t = sigmoid(W_beta·o_t + surprise_t)`, and a clear gate `clear_t = sigmoid(W_clear·o_t)` initialized near 1. The gated-delta rule is applied directly to Mamba2's own per-layer `ssm_state`, immediately after Mamba2's own decay+write and before the `C` readout, in `Model._mixer_step`:

```
ssm_state_t = ssm_state_t · dA + dBx                                      (Mamba2's own update, unmodified)
ssm_state_t = clear_t·(ssm_state_t - beta_t·(ssm_state_t·B_t)·B_t^T) + beta_t·p_t·B_t^T   (gated-delta merge)
y_t = C_t · ssm_state_t                                                   (Mamba2's own readout, unmodified)
```

and the *merged* `ssm_state_t` — not a separate copy — is what gets persisted to the next token. The delta term overwrites only the address being written and leaves other content intact (content-addressed forgetting); `clear_t` is a data-dependent global wipe for topic/segment boundaries, initialized near 1 (`clear_proj.bias = 4`) so it never forces decay on its own at init. `beta_t` is kept near 0 at init via a non-learnable `beta_anneal_offset`, linearly annealed from `-3` to `0` over the first `BETA_BIAS_ANNEAL_STEPS` (32) optimizer steps (`Model.set_beta_anneal`, called once per optimizer step via the `on_step` training hook, keyed on step count rather than tokens so it's independent of `--chunk-len`/`--accum-tokens`/`--memory-window`) and held at `0` after.

### Injection

Only a subset of layers carry a merge point — every even layer from 22 to 62 inclusive (`INJECTED_LAYERS = range(22, 64, 2)` in `model.py`), 21 of 64 layers total:

```
22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62
```

`READ_LAYER` (42) falls inside this range, so layer 42 both merges memory into its own state *and* is read by Stage 1 — see the diagram note above for why that doesn't create a same-token cycle. The injected-layer set follows a ~1/3 depth start (layer 22 of 64 = 34%), ~2/3 depth for `READ_LAYER` (42/64 = 66%).

### Parameter budget

~40M trainable against the 2.7B frozen backbone (~1.5%, similar ratio to the 780m variant's ~2.2%), excluding LoRA adapter params (which scale with `--lora-rank` and aren't fixed by the architecture) — Titans front-end (QKV + knob projections) ~19.7M, per-layer injection bottlenecks ~21M (21 layers × ~1M each, including that layer's gate projections).

### Open architectural question

The Titans front-end and the gated-delta accumulator are both gradient-based associative memories (the delta rule is one step of gradient descent on the same associative objective), so they may be partially redundant. Before committing to the full two-stage stack, the plan is to ablate a gated-delta-only variant against the full stack, and keep Stage 1 only if it earns its cost in long-context recall.

## Training data

Long-context datasets — see [`sft/README.md`](../../sft/README.md#long-context-data-mamba2_2_7b_memory): a mix of real long conversations ([`THUDM/LongAlign-10k`](https://huggingface.co/datasets/THUDM/LongAlign-10k)) and synthetic needle-in-haystack recall QA ([`RMT-team/babilong`](https://huggingface.co/datasets/RMT-team/babilong)).

## Chunk length

`DEFAULT_CHUNK_LEN = 7` — the fast-weight MLP snapshots (`w1`/`w2`) held in the backward graph scale with `MEM_HIDDEN` (10240 here vs 6144 for the 780m variant), so each token in a chunk costs ~210 MB; infinity-length chunks are not practical for 1k+ token training sequences. Tuned for `--batch-size 4` on a cloud H100 (80 GB) — see `sft/README.md`'s "Training mamba2_2_7b_memory" section for real run settings, which have used both this default and smaller explicit `--chunk-len` values. Benchmark with `make smoke-test --chunk-len N` across 10+ consecutive chunks to find the comfortable ceiling on your specific GPU — a single isolated chunk's peak VRAM is not representative of real multi-chunk training.

## Memory window

`DEFAULT_MEMORY_WINDOW = 1` (must evenly divide whatever `--chunk-len` is in use) — see the "Memory-window batching" section above and `sft/README.md`'s `--memory-window` entry.
