# mamba2_780m

## Source

[`state-spaces/mamba2-780m`](https://huggingface.co/state-spaces/mamba2-780m) — a 780M-parameter Mamba2 state-space language model.

## Tokenizer

`EleutherAI/gpt-neox-20b`. Mamba2 checkpoints from `state-spaces` ship without their own tokenizer; the GPT-NeoX-20B tokenizer is the standard pairing used in the original Mamba training recipe and vocabulary.

## LoRA target modules

`in_proj`, `out_proj` — the input/output projections of the SSM mixer block. These are the linear layers that dominate parameter count in each Mamba2 block and where adapting them gives the most leverage for fine-tuning, analogous to targeting `q_proj`/`v_proj` in a transformer.

## Special tokens

`USER_OPEN`/`ASST_OPEN` (`"[USER]"`/`"[ASSISTANT]"`) are registered as tokenizer special tokens (`SPECIAL_TOKENS` in `model.py`), so each role marker is a single atomic token id instead of several ordinary BPE pieces. They're bare — no trailing space — since the marker is a vocab-level concept distinct from prompt formatting; callers append a literal `" "` separator explicitly when building text (see `sft/prepare_data.py`, `backend/app/model/registry.py`). `load_base` resizes `backbone.embedding`/`lm_head` to fit the grown vocabulary, re-tying them (`tie_embeddings: true` in this model's config). Because `TARGET_LORA_MODULES` doesn't include the embedding or `lm_head`, the new tokens' embedding rows stay frozen at their (random) initial values during LoRA SFT — the model can only learn to use the markers via the LoRA-adapted `in_proj`, not by adjusting the embeddings themselves.

## `Model` wrapper quirks

- Reassembles the backbone (`embedding`, `layers`, `norm_f`) and `lm_head` from `MambaLMHeadModel` directly rather than calling it as a black box. `forward` loops over tokens manually and threads a `MixerState` (per-layer SSM/conv state) across calls, instead of calling `Mamba2.forward`/`Block.forward` with an `inference_params` cache — `mamba_ssm`'s own fused kernels (the `causal_conv1d` compiled extension *and* its Triton SSD chunk-scan kernel) are broken on this project's dev hardware (an unsupported ROCm GPU architecture): the former segfaults, the latter hangs, both independent of model size or package version. `_mixer_step` manually replicates `Mamba2.step()`'s arithmetic in plain PyTorch instead (verified to match the library's own reference fallback to float32-epsilon precision). This is slower per-token than the fused path would be, but it's the only thing proven to actually run on this hardware — see `models/mamba2_780m_memory/README.md`, which already used this approach for an unrelated reason (splicing in its memory subsystem) before this was known to be necessary here too.
- The state-threading also gives chunked training for free: `sft/train.py` (via this model's `train_hooks.py`) processes long examples in `--chunk-len`-token pieces, carrying (and detaching) `MixerState` across chunks of the same example, so training RAM is bounded by chunk length rather than example length.
- `_apply_norm_f`/`_prenorm` branch on `fused_add_norm`: when fused, they call the Triton `layer_norm_fn` kernel directly instead of `norm`/`norm_f`, matching how the upstream model applies normalization. (This Triton kernel is *not* one of the broken ones above — it's exercised extensively by `mamba2_780m_memory` already.)
- The model is fine-tuned to emit a `<revise>` tag after its response; this wrapper does not currently act on that tag (see TODO in `model.py`).
