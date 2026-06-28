# sft — LoRA fine-tuning for mamba2-780m

LoRA supervised fine-tuning of [`state-spaces/mamba2-780m`](https://huggingface.co/state-spaces/mamba2-780m) on conversational data. Produces a LoRA adapter that can be loaded into the `continual-learning` model as a better base.

Uses `mamba_ssm` directly (not HF peft/trl) to avoid Mamba-2 loading issues.

## Setup

```bash
make sync
```

Installs torch (ROCm/CUDA auto-detected) plus `causal-conv1d` and `mamba-ssm` from source.

## Data

The default dataset is [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) — the same dataset used by the original mamba-chat fine-tunes. Download and tokenize 20k examples with:

```bash
make data
# outputs data/train.pt
```

To use a local JSONL file instead:

```bash
make prepare ARGS="--input data/raw.jsonl"
```

Each line is a single JSON object with a `messages` list of `{"role", "content"}` turns. Loss is computed on assistant turns only; user tokens are masked out.

Add `"train": false` to an assistant turn to keep it in the context but exclude it from the loss — e.g. an earlier answer that a later turn revises, so the model learns the revision without learning the original mistake. The field is optional and defaults to `true`.

```json
{
  "messages": [
    { "role": "user", "content": "Tell me about the 2024 election." },
    { "role": "assistant", "train": false, "content": "Trump won, losing the popular vote to Biden." },
    { "role": "user", "content": "Actually he beat Kamala Harris." },
    { "role": "assistant", "content": "Trump won the 2024 election, beating Kamala Harris." }
  ]
}
```

(Shown formatted for readability; in the file each object must be on a single line.)

See `python prepare_data.py --help` for all options (`--max-examples`, `--hf-split`, `--max-len`, etc.).

### Long-context data (`mamba2_780m_memory`)

That model's whole point is long-range recall, so its training data needs long sessions, not the short ones above. `make data-memory` (set `MODEL_NAME=mamba2_780m_memory` first) builds and merges two sources:

```bash
MODEL_NAME=mamba2_780m_memory make data-memory
# outputs data/train_memory.pt
```

- [`THUDM/LongAlign-10k`](https://huggingface.co/datasets/THUDM/LongAlign-10k) — real long multi-turn conversations, already in the `messages` shape `prepare_data.py` expects.
- [`RMT-team/babilong`](https://huggingface.co/datasets/RMT-team/babilong) — synthetic needle-in-haystack recall QA (`prepare_babilong.py` converts its `{input, question, target}` schema into a synthetic one-turn `messages` conversation first). Mixed in deliberately: long natural text alone doesn't force a model to actually *use* far-back information, only babilong-style tasks do, since getting the answer right depends on it.

`merge_data.py` concatenates the two tokenized outputs into one `.pt` file, since `train.py` only accepts a single `--data` path.

`--max-len 100000` here is intentionally far above any real example (LongAlign-10k's longest is ~65k tokens) — it only controls what gets written to disk, which is nearly free, so there's no reason to truncate at prep time. `format_conversation`'s truncation (drops trailing turns once a conversation exceeds `--max-len`) used to silently discard conversations where a single long turn alone exceeded a too-small `--max-len` before the assistant turn was ever reached; at a too-tight `--max-len 16384` this dropped roughly half of LongAlign-10k as "no assistant turns." Bounding training-time RAM belongs in chunked/truncated-BPTT training (`Model.forward`'s `state` param exists for exactly this — see `models/mamba2_780m_memory/README.md`), not in dropping data at prep time.

## Training

```bash
make train                      # start fresh
make resume                     # resume from latest checkpoint
make resume ARGS="--epochs 2"   # resume and train an extra epoch
make preflight                  # load the real model+data, run the preflight gradient check, exit
```

Pass any `train.py` flag through any of these targets with `ARGS="..."` (e.g. `make train ARGS="--eos-weight 10"`).

`make preflight` is worth running before a real training run, especially after touching a model's `train_hooks.py` or `model.py`: it loads the actual model and dataset (so it still pays for that, unlike the synthetic-model unit test in `models/tests/`) and runs one example through the same chunked forward+backward path training uses, asserting gradients actually reached every trainable parameter — then exits before the full loop. This is what would have caught this model's "the gated-delta merge never actually ran" and "k_proj/v_proj never received gradient" bugs immediately, instead of after a full run.

**Slot-based batching** (`--batch-size`, default 4): `train.py` runs `B` examples in parallel using a slot-based loop. Each slot independently advances through its own example; when a slot finishes, it resets and picks up the next example. All slots are processed in one batched forward+backward per chunk, so the GPU sees a `(B, chunk_len)` tensor every step rather than `(1, chunk_len)`. This is the primary mechanism for saturating GPU utilisation on stateful models like `mamba2_2_7b_memory` whose per-token step prevents the parallel-scan kernel from helping. Gradient accumulation (`--accum-steps`, default 12) sums gradients across all active slots before stepping; the optimizer step fires every `accum_steps` chunks (i.e., after `accum_steps × B × chunk_len` token-positions processed, minus any padding).

The live per-chunk progress display (in-place terminal output) shows one line per active slot:

```
[14:32:01]  avg_loss 2.34
  slot 0  token   3421/65000  beta 0.6369  clear 0.9821  active 21/21  surprise 0.1922  o_t_norm 1.0752
  slot 1  token    891/12400  beta 0.5821  clear 0.9734  active 19/21  surprise 0.2103  o_t_norm 0.9841
  slot 2  token  12004/98221  beta 0.7012  clear 0.9901  active 21/21  surprise 0.1654  o_t_norm 1.1203
  slot 3  token    203/8831   beta 0.6543  clear 0.9812  active 20/21  surprise 0.1877  o_t_norm 1.0341
```

Checkpoints are saved every `--ckpt-every-tokens` tokens of training (default 2000) to `../models/{MODEL_NAME}/checkpoints/epoch-E/step-N/` (every trainable parameter + optimizer state), where `E` is the 1-indexed epoch and `MODEL_NAME` is read from `.env`. Cadence is counted in cumulative tokens trained on rather than optimizer steps, since examples vary enormously in length (a few thousand to ~100k tokens) — a step-based cadence means wildly different amounts of training between checkpoints depending on what examples happen to fall in the window. The trigger is checked after every gradient-accumulation boundary, so a checkpoint can land *mid-example* for long examples; resuming replays (forward-only) each slot's already-seen prefix to regenerate carried state, then continues from exactly where it left off. Only the last 20 checkpoints **per epoch** are kept; older ones in the same epoch are deleted automatically, so completed epochs retain their final 20. Tune with `--ckpt-every-tokens` and `--keep-ckpts`.

**EOS under-generation**: if the model doesn't emit `<|endoftext|>` to end turns, pass `--eos-weight 5` (or higher) to upweight EOS positions in the loss. EOS tokens are ~0.8% of assistant tokens so they get little gradient by default.

See `python train.py --help` for all options (learning rate, rank, accumulation steps, etc.).

`train.py` is generic across every model in `models/` — it dispatches to that model's `train_hooks.py` (`models/{name}/train_hooks.py`) for the only part that genuinely differs: how to load/wrap the model for training, and how to compute loss for one chunk. Everything else — chunk iteration, shuffling, accumulation counting, checkpoint cadence/rotation (including mid-example resume), evaluation, preflight, non-finite checks — is shared, since both models here are chunked, state-threaded ones (just with very different `--chunk-len`s).

### Training `mamba2_780m_memory`

This model's `train_hooks.py` differs from `mamba2_780m`'s in two ways, both visible in its module docstring: `Model.forward(input_ids, state)` is stateful, so `train.py` processes examples in `--chunk-len`-token chunks with `state` carried (and detached) across chunks of the *same* example — never across different examples — bounding training RAM by chunk length rather than example length (see the "Long-context data" section above for why examples themselves aren't truncated at prep time instead). And loss is computed over every token, not just assistant turns, since for this model the content worth exercising long-range recall on is mostly in the long user turns.

```bash
MODEL_NAME=mamba2_780m_memory make train ARGS="--data data/train_memory.pt"
MODEL_NAME=mamba2_780m_memory make resume ARGS="--data data/train_memory.pt"
```

For `mamba2_2_7b_memory`, use the same pattern. The default `--batch-size 4` and `--chunk-len 12` are tuned for that model on an H100 (80 GB); the 2.7B model's per-token fast-weight snapshot is ~210 MB, so a single chunk of length 12 with B=4 costs roughly 10 GB of backward graph on top of the 5–6 GB model weight floor. Benchmark with `make preflight` before raising either.

```bash
MODEL_NAME=mamba2_2_7b_memory make train ARGS="--data data/train_memory.pt"
MODEL_NAME=mamba2_2_7b_memory make resume ARGS="--data data/train_memory.pt"
```

## Using the adapter

Each checkpoint directory contains `lora_config.json` with the rank and alpha used during training, so callers don't need to hard-code them, and `trainable.pt` with every trainable parameter (LoRA adapters, plus a model's own full-gradient subsystem if it has one — e.g. `mamba2_780m_memory`'s `front_end`/`injections`):

```python
import json
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from lora import apply_lora

ckpt = "../models/mamba2_780m/checkpoints/epoch-1/step-N"
cfg = json.loads(open(f"{ckpt}/lora_config.json").read())
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-780m", dtype=torch.bfloat16)
model = apply_lora(model, ["in_proj", "out_proj"], cfg["rank"], cfg["alpha"])
state = torch.load(f"{ckpt}/trainable.pt", weights_only=True)
model.load_state_dict(state, strict=False)
```

(The backend's loader, `backend/app/model/lora.py:load_checkpoint`, does the same thing — wrap the model first, then load `trainable.pt` into the wrapped model, not the raw backbone, so a model with extra trainable state outside the backbone still loads correctly.)
