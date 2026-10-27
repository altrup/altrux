# Model instructions

Each model folder's `README.md` should briefly cover:

- **Source** — the HuggingFace model ID and what it is
- **Tokenizer** — which tokenizer is paired with it and why
- **LoRA target modules** — why those specific modules were chosen as adapter targets
- **`Model` wrapper quirks** — anything non-obvious about the inference wrapper (e.g. deviations from the upstream model's forward pass)
- **Special tokens** — which role markers are registered as tokenizer special tokens and why

See `models/mamba2_2_7b/README.md` for an example.

## Experiment header

Each module's docstring starts with `Experiment: <slug>` — see `sft/AGENTS.md` for the slug set and rule.

## Kernel paths

Inference hot paths take the fastest available kernel path on CUDA: the fused SSD chunk-scan for multi-token input, the fused decode step (`causal_conv1d_update` + `selective_state_update`) for single tokens. The manual per-token loop runs only where the kernels are unusable (HIP, or a kernel that fails to import) or where gradients or hooks (`erase_hook`, `c_capture`) need the manual arithmetic. Every fast path has an equivalence test against the manual path it replaces, which skips where its kernels are unavailable (`sft/tests/experiments/dreams/test_mixer_fused.py`, `models/tests/test_mamba2_780m_fused_step.py`).

By default the manual path is not twice differentiable. Every norm it calls (`layer_norm_fn`, the block and final `RMSNorm`, the mixer's gated `RMSNorm`) is a mamba_ssm Triton `autograd.Function` whose backward returns tensors with no graph. A `create_graph=True` gradient through the model therefore loses the second-order path at each norm and raises no error. Set `model.torch_norms = True` (`mamba2_780m`'s `Model`, so also `mamba2_2_7b`) before a double backward (arm (b) of `DISCUSSION-20261024`): every norm then runs mamba_ssm's own plain-torch reference arithmetic (`models/norms.py`). On a tiny model the double backward's d/dπ misses a finite difference by 90% with the Triton norms and 0.3% with `torch_norms` (`models/tests/test_mamba2_780m_torch_norms.py`). The switch makes only the norms twice differentiable: the fused chunk-scan and decode kernels are not, so drive the double backward through `_forward_tokens` (as the gate does). The memory models (`mamba2_2_7b_memory` and its 780M variants) do not have the switch.

## Checkpoints

Training checkpoints for a model live in `models/{name}/checkpoints/`, gitignored via `models/.gitignore`, since they're a model artifact consumed by both `sft` (writes) and `backend` (reads), not an `sft`-only concern.

## `train_hooks.py`

Each model folder also contains a `train_hooks.py`, sibling to `model.py`, exporting the training-specific interface `sft/training/loop.py`'s generic loop dispatches to:

| Name | Description |
|------|------|
| `setup_training(device, lora_rank, lora_alpha, lora_dropout)` | Loads/wraps the model for training (LoRA attachment, any wrapping the loss computation needs). Returns `(model, trainable_params)`. |
| `DEFAULT_CHUNK_LEN` | This model's default tokens-per-chunk, used when `--chunk-len` isn't passed. |
| `chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight)` | Runs forward for *one chunk* (already on `device`) and returns `(loss_sum, weight_sum, state)` — no backward call, no chunk iteration. `sft/training/loop.py` owns chunking, gradient accumulation, checkpointing (including mid-example resume), evaluation, and preflight generically; this is the only irreducibly model-specific piece. |
| `set_grad_checkpoint(model, enabled, block=None)` | Optional. Called at the start of every config-group segment with that data slice's `grad_checkpoint` setting, so a long-`chunk_len` slice can pay the recompute tax while short-chunk slices in the same run don't. Enabling must be an exact no-op on results and `enabled=False` must restore the un-checkpointed forward byte-for-byte. |
| `extra_log(model) -> str \| None` | Optional. Printed once per optimizer step if defined. |
| `chunk_extra_log(model) -> str \| list[str] \| None` | Optional. If defined, `sft/training/loop.py`'s generic per-chunk live progress display is shown (in-place, overwritten each chunk). Return a single `str` for single-example models, or a `list[str]` with one entry per batch slot for slot-batched models (e.g. `mamba2_2_7b_memory`). `None` to suppress. |

`mask_slice` is a `(B, T)` float weight tensor: 0 for padding and for tokens the dataset's train mask excludes (user turns), otherwise 1 times any `--recall-weight`/`--head-weight` boost. Checkpoint save/load is *not* a per-model hook — it's generic in `sft/training/checkpoints.py` (saves every parameter with `requires_grad=True`), since that criterion is already correct for both a LoRA-only model and one with an additional full-gradient subsystem.

See `models/mamba2_2_7b/train_hooks.py` for the simple case and `models/mamba2_2_7b_memory/train_hooks.py` for the one that also defines `chunk_extra_log`.
