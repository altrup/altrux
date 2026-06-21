# Claude Guidelines — models

Each model folder's `README.md` should briefly cover:

- **Source** — the HuggingFace model ID and what it is
- **Tokenizer** — which tokenizer is paired with it and why
- **LoRA target modules** — why those specific modules were chosen as adapter targets
- **`Model` wrapper quirks** — anything non-obvious about the inference wrapper (e.g. deviations from the upstream model's forward pass)
- **Special tokens** — which role markers are registered as tokenizer special tokens and why

See `models/mamba2_780m/README.md` for an example.

## Checkpoints

Training checkpoints for a model live in `models/{name}/checkpoints/`, gitignored via `models/.gitignore`, since they're a model artifact consumed by both `sft` (writes) and `backend` (reads), not an `sft`-only concern.

## `train_hooks.py`

Each model folder also contains a `train_hooks.py`, sibling to `model.py`, exporting the training-specific interface `sft/train.py`'s generic loop dispatches to:

| Name | Description |
|------|------|
| `setup_training(device, lora_rank, lora_alpha, lora_dropout)` | Loads/wraps the model for training (LoRA attachment, any wrapping the loss computation needs). Returns `(model, trainable_params)`. |
| `process_example(model, ids, mask, device, eos_weight, backward_scale, chunk_len)` | Runs forward and one or more `.backward()` calls for one example (already on `device`). Returns `(loss_sum, weight_sum, n_backward_calls)` — `n_backward_calls` is what the generic loop counts against `--accum-steps`, so it must match however many times this function actually called `.backward()`. |
| `eval_loss(model, eval_ids, eval_masks, device, max_len, chunk_len)` | Held-out loss, no backward. |
| `preflight(model, trainable_params, all_ids, all_masks, device, max_len, eos_weight, chunk_len)` | Runs one example through `process_example` and asserts gradients actually reached `trainable_params` — this is what catches a model whose forward silently fails to connect some part of itself to the loss (see `models/mamba2_2_7b_memory/train_hooks.py`'s history for why this check matters). |

`mask`/`chunk_len` exist for interface parity across models even when a given model ignores them (e.g. a stateless model ignores `chunk_len`; `mamba2_2_7b_memory` ignores `mask` since it trains on every token, not just assistant turns). Checkpoint save/load is *not* a per-model hook — it's generic in `sft/train.py` (saves every parameter with `requires_grad=True`), since that criterion is already correct for both a LoRA-only model and one with an additional full-gradient subsystem.

See `models/mamba2_780m/train_hooks.py` for the simple (stateless) case and `models/mamba2_2_7b_memory/train_hooks.py` for the chunked, state-threaded one.
