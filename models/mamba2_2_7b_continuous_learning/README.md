# mamba2_2_7b_continuous_learning

## Source

[`state-spaces/mamba2-2.7b`](https://huggingface.co/state-spaces/mamba2-2.7b) — a 2.7B-parameter Mamba2 state-space language model.

## Tokenizer

`EleutherAI/gpt-neox-20b`. Mamba2 checkpoints from `state-spaces` ship without their own tokenizer; the GPT-NeoX-20B tokenizer is the standard pairing used in the original Mamba training recipe and vocabulary.

## LoRA target modules

`in_proj`, `out_proj` — the input/output projections of the SSM mixer block. These are the linear layers that dominate parameter count in each Mamba2 block and where adapting them gives the most leverage for fine-tuning, analogous to targeting `q_proj`/`v_proj` in a transformer.

## Special tokens

`USER_OPEN`/`ASST_OPEN` (`"[USER]"`/`"[ASSISTANT]"`) are registered as tokenizer special tokens (`SPECIAL_TOKENS` in `model.py`), so each role marker is a single atomic token id instead of several ordinary BPE pieces. They're bare — no trailing space — since the marker is a vocab-level concept distinct from prompt formatting; callers append a literal `" "` separator explicitly when building text (see `sft/prepare_data.py`, `backend/app/model/registry.py`). `load_base` resizes `backbone.embedding`/`lm_head` to fit the grown vocabulary via `models.common.extend_embeddings`, which branches on `model.config.tie_embeddings` (re-tying if `true`, growing independently if `false`). This checkpoint's cache (`state-spaces/mamba2-2.7b`) isn't present locally, so its `tie_embeddings` value wasn't directly verified here; the resize logic itself was verified against the real `mamba2-780m` checkpoint and `extend_embeddings`'s unit tests in Task 2, and applies generically to any Mamba2 checkpoint regardless of which way `tie_embeddings` is set. Because `TARGET_LORA_MODULES` doesn't include the embedding or `lm_head`, the new tokens' embedding rows stay frozen at their (random) initial values during LoRA SFT — the model can only learn to use the markers via the LoRA-adapted `in_proj`, not by adjusting the embeddings themselves.

## `Model` wrapper quirks

- Reassembles the backbone (`embedding`, `layers`, `norm_f`) and `lm_head` from `MambaLMHeadModel` directly rather than calling it as a black box, so `forward` can thread `inference_params` through for incremental (single-token) decoding.
- `_apply_norm_f` branches on `fused_add_norm`: when fused, it calls the Triton `layer_norm_fn` kernel directly instead of `norm_f`, matching how the upstream model applies the final norm.
- The model is fine-tuned to emit a `<revise>` tag after its response; this wrapper does not currently act on that tag (see TODO in `model.py`).
