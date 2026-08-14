# mamba2_2_7b

The **plain** Mamba2-2.7B substrate registered by
`notes/discussion/DISCUSSION-20260808-headline-collapse-deep-block-and-regime-bridge.md`
§2.10.14, which switched the run's substrate from 780M to 2.7B: the §4 pilot
showed generator quality was the binding constraint on the un-spliced dream
regime, and scale attacks exactly that. Every run command reads
`MODEL_NAME=mamba2_2_7b` with checkpoints under this folder's `checkpoints/`.

`models/mamba2_2_7b_memory` is the **memory-augmented** variant on the same
backbone and is *not* this — it carries a fast-weight memory subsystem, its own
read/injection layers, and a training loop that trains on every token. This
folder has none of that: it is `mamba2_780m` with a bigger backbone, nothing
else.

## Source

[`state-spaces/mamba2-2.7b`](https://huggingface.co/state-spaces/mamba2-2.7b) — a
2.7B-parameter Mamba2 state-space language model: 64 layers, `d_model` 2560,
80 heads of `headdim` 64, `d_state` 128. Loaded in bf16 (~5.4 GB), so LoRA runs
full-precision on the adapter targets and there is no `QUANTIZE_LORA_BASE`.

## Tokenizer

`EleutherAI/gpt-neox-20b`, same as every other model here. Mamba2 checkpoints
from `state-spaces` ship without their own tokenizer; the GPT-NeoX-20B tokenizer
is the standard pairing from the original Mamba training recipe and vocabulary.

## LoRA target modules

`in_proj`, `out_proj` — the input/output projections of the SSM mixer block,
the linear layers that dominate each Mamba2 block's parameter count and where
adapting gives the most leverage, analogous to targeting `q_proj`/`v_proj` in a
transformer.

## Special tokens

`USER_OPEN`/`ASST_OPEN` (`"[USER]"`/`"[ASSISTANT]"`) and `EOC`
(`"<|endofconversation|>"`) are registered as tokenizer special tokens
(`SPECIAL_TOKENS` in `model.py`), so each is a single atomic token id rather
than several ordinary BPE pieces. The markers are bare — no trailing space —
since the marker is a vocab-level concept distinct from prompt formatting;
callers append a literal `" "` separator explicitly (see `sft/prepare_data.py`,
`backend/app/model/registry.py`). `load_base` resizes
`backbone.embedding`/`lm_head` to fit the grown vocabulary, re-tying them
(`tie_embeddings: true` in this model's config), and initializes each new row to
the mean of its BPE-spelling rows (`models/common.py:extend_embeddings`). The
table stays frozen during SFT, but the `Model` wrapper carries a `MarkerDelta`
(`models/common.py`) — a small zero-initialized trainable parameter added to
those rows' embedding output and, via the tie, to their logit columns — so
exactly those rows train while the other ~50k stay frozen and out of the
checkpoints.

`EOC` is a conversation boundary of the same family as `<|endoftext|>` (which
ends an assistant turn in this corpus). `sft/prepare_data.py` appends it to
every rendered conversation, so the corpus teaches it as conversation-end;
`sft/dream_sleep.py` reads the same token as the point a self-terminating dream
stops, and as the dream-start steer prefix.

## `Model` wrapper quirks

Identical to `models/mamba2_780m` — the wrapper reads every dimension off the
loaded backbone, so nothing here is size-specific. See
`models/mamba2_780m/README.md` for the full investigation behind each of these;
summarized:

- Reassembles the backbone (`embedding`, `layers`, `norm_f`) and `lm_head` from
  `MambaLMHeadModel` directly rather than calling it as a black box, and threads
  a `MixerState` (per-layer SSM/conv state) across `forward` calls instead of
  using an `inference_params` cache.
- Two interchangeable mixer paths, same states in and out: `_mixer_chunk` runs a
  whole chunk through the fused Triton SSD chunk-scan (conv as a plain grouped
  `F.conv1d` over the incoming `conv_state`, so no `causal_conv1d` build is
  needed), and `_mixer_step` replicates `Mamba2.step()`'s arithmetic in plain
  PyTorch one token at a time. `forward` takes the fused path only for `T > 1`
  on a non-ROCm CUDA host with the kernel importable and no `erase_hook` set;
  the dev box (an unsupported ROCm gfx arch, where both of `mamba_ssm`'s fused
  kernel families are broken) always takes the token loop.
- State threading gives chunked training for free: `sft/train.py` (via
  `train_hooks.py`) carries and detaches `MixerState` across `--chunk-len`
  chunks of the same example, bounding training RAM by chunk length rather than
  example length.
- `Model.c_capture`: set to a list to have `_mixer_step` append each layer's
  post-conv read query `C` (detached), token-major — how `sft/erase_probe.py`
  and the dream-sleep loop address the state erase. Per-token path only.
- `Model.erase_hook`: set to `hook(layer_idx, ssm_state, C)` to replace a
  layer's carried state just before its decay+write — the
  ablate-then-write-then-read order the dream-sleep counterfactual arms need. A
  set hook keeps a multi-token forward on the token loop. Per-token only;
  `None` is a byte-identical no-op.
- `Model.set_grad_checkpoint(enabled, block=GRAD_CHECKPOINT_BLOCK)` trades
  recompute for VRAM at long `--chunk-len` via `models/common.py`'s
  `blockwise_checkpoint`. Disabled (the default, and any `no_grad` call) is an
  exact no-op. Boundary state here is ~43.3M values, ~87 MB per batch slot in
  bf16 — 2.2x the 780M figure, from 64 layers of 80x64x128 `ssm_state`.
- `chunk_loss` (`train_hooks.py`) trains on assistant-turn tokens only, so a
  chunk that is entirely user-turn content runs under `torch.no_grad()` — the
  caller never backwards it, and without the guard its forward graph is left for
  the cyclic GC and OOMs the next chunk.
- `train_hooks.DEFAULT_CHUNK_LEN` is carried over from 780M **unmeasured** at
  this scale; the box smoke sets the real value (see the comment there).
