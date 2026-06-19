# mamba2_2_7b_continuous_learning

## Source

[`state-spaces/mamba2-2.7b`](https://huggingface.co/state-spaces/mamba2-2.7b) — a 2.7B-parameter Mamba2 state-space language model.

## Tokenizer

`EleutherAI/gpt-neox-20b`. Mamba2 checkpoints from `state-spaces` ship without their own tokenizer; the GPT-NeoX-20B tokenizer is the standard pairing used in the original Mamba training recipe and vocabulary.

## LoRA target modules

`in_proj`, `out_proj` — the input/output projections of the SSM mixer block. These are the linear layers that dominate parameter count in each Mamba2 block and where adapting them gives the most leverage for fine-tuning, analogous to targeting `q_proj`/`v_proj` in a transformer.

## `Model` wrapper quirks

- Reassembles the backbone (`embedding`, `layers`, `norm_f`) and `lm_head` from `MambaLMHeadModel` directly rather than calling it as a black box, so `forward` can thread `inference_params` through for incremental (single-token) decoding.
- `_apply_norm_f` branches on `fused_add_norm`: when fused, it calls the Triton `layer_norm_fn` kernel directly instead of `norm_f`, matching how the upstream model applies the final norm.
- The model is fine-tuned to emit a `<revise>` tag after its response; this wrapper does not currently act on that tag (see TODO in `model.py`).
