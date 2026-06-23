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
| `DEFAULT_CHUNK_LEN` | This model's default tokens-per-chunk, used when `--chunk-len` isn't passed. |
| `chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight)` | Runs forward for *one chunk* (already on `device`) and returns `(loss_sum, weight_sum, state)` — no backward call, no chunk iteration. `sft/train.py` owns chunking, gradient accumulation, checkpointing (including mid-example resume), evaluation, and preflight generically; this is the only irreducibly model-specific piece. |
| `extra_log(model) -> str \| None` | Optional. Printed once per optimizer step if defined. |
| `chunk_extra_log(model) -> str \| None` | Optional. If defined, `sft/train.py`'s generic per-chunk live progress display is shown (in-place, overwritten each chunk) with this as an extra status line — useful for a model whose chunks are slow enough that per-chunk feedback matters (e.g. `mamba2_2_7b_memory`). Models that don't define it get no live progress display, same as before this was generic. |

`mask_slice` exists for interface parity across models even when a given model ignores it (`mamba2_2_7b_memory` trains on every token, not just assistant turns, so it ignores `mask_slice`). Checkpoint save/load is *not* a per-model hook — it's generic in `sft/train.py` (saves every parameter with `requires_grad=True`), since that criterion is already correct for both a LoRA-only model and one with an additional full-gradient subsystem.

See `models/mamba2_780m/train_hooks.py` for the simple case and `models/mamba2_2_7b_memory/train_hooks.py` for the one that also defines `chunk_extra_log`.
