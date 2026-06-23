# mamba2_2_7b_continuous_learning

## Source

[`state-spaces/mamba2-2.7b`](https://huggingface.co/state-spaces/mamba2-2.7b) — a 2.7B-parameter Mamba2 state-space language model.

## Tokenizer

`EleutherAI/gpt-neox-20b`. Mamba2 checkpoints from `state-spaces` ship without their own tokenizer; the GPT-NeoX-20B tokenizer is the standard pairing used in the original Mamba training recipe and vocabulary.

## LoRA target modules

`in_proj`, `out_proj` — the input/output projections of the SSM mixer block. These are the linear layers that dominate parameter count in each Mamba2 block and where adapting them gives the most leverage for fine-tuning, analogous to targeting `q_proj`/`v_proj` in a transformer.

This model sets `QUANTIZE_LORA_BASE = True`: `load_base` quantizes `in_proj`/`out_proj` to 4-bit (NF4, via `bitsandbytes`, see `models/common.py:quantize_lora_targets`) before LoRA adapters are attached, so this is true QLoRA rather than plain full-precision LoRA — see [`mamba2_2_7b_memory`](../mamba2_2_7b_memory/README.md), which uses the same backbone and the same flag. Without it, this 2.7B-parameter backbone needs ~11GB of VRAM at full fp32 precision, more than this project's dev GPU (8GB) has, so training/preflight would OOM before ever reaching a forward pass. `bitsandbytes`'s 4-bit quantization segfaults on this dev machine's unsupported `gfx1102` GPU arch unless `HSA_OVERRIDE_GFX_VERSION=11.0.0` is set — see the root `CLAUDE.md`.

## Special tokens

`USER_OPEN`/`ASST_OPEN` (`"[USER]"`/`"[ASSISTANT]"`) are registered as tokenizer special tokens (`SPECIAL_TOKENS` in `model.py`), so each role marker is a single atomic token id instead of several ordinary BPE pieces. They're bare — no trailing space — since the marker is a vocab-level concept distinct from prompt formatting; callers append a literal `" "` separator explicitly when building text (see `sft/prepare_data.py`, `backend/app/model/registry.py`). `load_base` resizes `backbone.embedding`/`lm_head` to fit the grown vocabulary via `models.common.extend_embeddings`, which branches on `model.config.tie_embeddings` (re-tying if `true`, growing independently if `false`). This checkpoint's cache (`state-spaces/mamba2-2.7b`) isn't present locally, so its `tie_embeddings` value wasn't directly verified here; the resize logic itself was verified against the real `mamba2-780m` checkpoint and `extend_embeddings`'s unit tests in Task 2, and applies generically to any Mamba2 checkpoint regardless of which way `tie_embeddings` is set. Because `TARGET_LORA_MODULES` doesn't include the embedding or `lm_head`, the new tokens' embedding rows stay frozen at their (random) initial values during LoRA SFT — the model can only learn to use the markers via the LoRA-adapted `in_proj`, not by adjusting the embeddings themselves.

## Planned: critic-gated gradient accumulation

`model.py` is removed for now, pending a redesign around a dedicated, integrated critic path: a small head (likely outputting a scalar) that judges the model's own output as it goes. Gradients accumulate continuously rather than being applied per-example; once the critic signals that its output warrants it, the accumulated gradient is applied in a training step. The previous `Model` wrapper (manual `_mixer_step` token loop, `MixerState` threading, chunked training) and `train_hooks.py`'s training loop are gone along with it — see git history for the prior implementation if useful as a starting point. This README section will be replaced with real `Model wrapper quirks`/`Critic` documentation once the redesign lands.
