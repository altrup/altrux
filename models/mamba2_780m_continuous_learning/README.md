# mamba2_780m_continuous_learning

## Source

[`state-spaces/mamba2-780m`](https://huggingface.co/state-spaces/mamba2-780m) — a 780M-parameter Mamba2 state-space language model.

## Tokenizer

`EleutherAI/gpt-neox-20b`. Mamba2 checkpoints from `state-spaces` ship without their own tokenizer; the GPT-NeoX-20B tokenizer is the standard pairing used in the original Mamba training recipe and vocabulary.

## LoRA target modules

`in_proj`, `out_proj` — the input/output projections of the SSM mixer block. These are the linear layers that dominate parameter count in each Mamba2 block and where adapting them gives the most leverage for fine-tuning, analogous to targeting `q_proj`/`v_proj` in a transformer. Plain full-precision LoRA, no 4-bit quantization — the 780M backbone fits this project's dev GPU (8GB) comfortably without it.

## Special tokens

`USER_OPEN`/`ASST_OPEN` (`"[USER]"`/`"[ASSISTANT]"`) are registered as tokenizer special tokens (`SPECIAL_TOKENS` in `model.py`), so each role marker is a single atomic token id instead of several ordinary BPE pieces. They're bare — no trailing space — since the marker is a vocab-level concept distinct from prompt formatting; callers append a literal `" "` separator explicitly when building text (see `sft/prepare_data.py`, `backend/app/model/registry.py`). `load_base` resizes `backbone.embedding`/`lm_head` to fit the grown vocabulary, re-tying them (`tie_embeddings: true` in this model's config). Because `TARGET_LORA_MODULES` doesn't include the embedding or `lm_head`, the new tokens' embedding rows stay frozen at their (random) initial values during LoRA SFT — the model can only learn to use the markers via the LoRA-adapted `in_proj`, not by adjusting the embeddings themselves.

## Planned: critic-gated gradient accumulation

`model.py` is removed for now, pending a redesign around a dedicated, integrated critic path: a small head (likely outputting a scalar) that judges the model's own output as it goes. Gradients accumulate continuously rather than being applied per-example; once the critic signals that its output warrants it, the accumulated gradient is applied in a training step. The previous `Model` wrapper (manual `_mixer_step` token loop, `MixerState` threading, chunked training) and `train_hooks.py`'s training loop are gone along with it — see git history for the prior implementation if useful as a starting point. This README section will be replaced with real `Model wrapper quirks`/`Critic` documentation once the redesign lands.

This model has no `<revise>`-tag behavior — that's specific to [`mamba2_780m`](../mamba2_780m/README.md). The two models target different problems (revision-on-feedback vs. continuous learning) on the same backbone.
