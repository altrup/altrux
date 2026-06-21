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

### Long-context data (`mamba2_2_7b_memory`)

That model's whole point is long-range recall, so its training data needs long sessions, not the short ones above. `make data-memory` (set `MODEL_NAME=mamba2_2_7b_memory` first) builds and merges two sources:

```bash
MODEL_NAME=mamba2_2_7b_memory make data-memory
# outputs data/train_memory.pt
```

- [`THUDM/LongAlign-10k`](https://huggingface.co/datasets/THUDM/LongAlign-10k) — real long multi-turn conversations, already in the `messages` shape `prepare_data.py` expects.
- [`RMT-team/babilong`](https://huggingface.co/datasets/RMT-team/babilong) — synthetic needle-in-haystack recall QA (`prepare_babilong.py` converts its `{input, question, target}` schema into a synthetic one-turn `messages` conversation first). Mixed in deliberately: long natural text alone doesn't force a model to actually *use* far-back information, only babilong-style tasks do, since getting the answer right depends on it.

`merge_data.py` concatenates the two tokenized outputs into one `.pt` file, since `train.py` only accepts a single `--data` path.

`--max-len 100000` here is intentionally far above any real example (LongAlign-10k's longest is ~65k tokens) — it only controls what gets written to disk, which is nearly free, so there's no reason to truncate at prep time. `format_conversation`'s truncation (drops trailing turns once a conversation exceeds `--max-len`) used to silently discard conversations where a single long turn alone exceeded a too-small `--max-len` before the assistant turn was ever reached; at a too-tight `--max-len 16384` this dropped roughly half of LongAlign-10k as "no assistant turns." Bounding training-time RAM belongs in chunked/truncated-BPTT training (`Model.forward`'s `state` param exists for exactly this — see `models/mamba2_2_7b_memory/README.md`), not in dropping data at prep time.

## Training

```bash
make train                      # start fresh
make resume                     # resume from latest checkpoint
make resume ARGS="--epochs 2"   # resume and train an extra epoch
make preflight                  # load the real model+data, run the preflight gradient check, exit
```

Pass any `train.py` flag through any of these targets with `ARGS="..."` (e.g. `make train ARGS="--eos-weight 10"`).

`make preflight` is worth running before a real training run, especially after touching a model's `train_hooks.py` or `model.py`: it loads the actual model and dataset (so it still pays for that, unlike the synthetic-model unit test in `models/tests/`) and runs one example through `process_example`, asserting gradients actually reached every trainable parameter — then exits before the full loop. This is what would have caught this model's "the gated-delta merge never actually ran" and "k_proj/v_proj never received gradient" bugs immediately, instead of after a full run.

Checkpoints are saved every 50 optimizer steps to `../models/{MODEL_NAME}/checkpoints/epoch-E/step-N/` (every trainable parameter + optimizer state), where `E` is the 1-indexed epoch and `MODEL_NAME` is read from `.env`. Only the last 20 checkpoints **per epoch** are kept; older ones in the same epoch are deleted automatically, so completed epochs retain their final 20. Tune with `--ckpt-every` and `--keep-ckpts`.

**EOS under-generation**: if the model doesn't emit `<|endoftext|>` to end turns, pass `--eos-weight 5` (or higher) to upweight EOS positions in the loss. EOS tokens are ~0.8% of assistant tokens so they get little gradient by default.

See `python train.py --help` for all options (learning rate, rank, accumulation steps, etc.).

`train.py` is generic across every model in `models/` — it dispatches to that model's `train_hooks.py` (`models/{name}/train_hooks.py`) for the parts that genuinely differ: how to load/wrap the model for training, and how to run forward+backward for one example. Everything else (shuffling, accumulation counting, checkpoint cadence/rotation, resume, non-finite checks) is shared.

### Training `mamba2_2_7b_memory`

This model's `train_hooks.py` differs from the others' in two ways, both visible in its module docstring: `Model.forward(input_ids, state)` is stateful, so examples are processed in `--chunk-len`-token chunks with `state` carried (and detached) across chunks of the *same* example — never across different examples — bounding training RAM by chunk length rather than example length (see the "Long-context data" section above for why examples themselves aren't truncated at prep time instead). And loss is computed over every token, not just assistant turns, since for this model the content worth exercising long-range recall on is mostly in the long user turns.

```bash
MODEL_NAME=mamba2_2_7b_memory make train ARGS="--data data/train_memory.pt --chunk-len 512"
MODEL_NAME=mamba2_2_7b_memory make resume ARGS="--data data/train_memory.pt --chunk-len 512"
```

On this machine's GPU (unsupported `gfx1102` arch), set `HSA_OVERRIDE_GFX_VERSION=11.0.0` in your shell before training this model — see the root `CLAUDE.md`.

## Using the adapter

Each checkpoint directory contains `lora_config.json` with the rank and alpha used during training, so callers don't need to hard-code them, and `trainable.pt` with every trainable parameter (LoRA adapters, plus a model's own full-gradient subsystem if it has one — e.g. `mamba2_2_7b_memory`'s `front_end`/`injections`):

```python
import json
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from lora import apply_lora

ckpt = "../models/mamba2_780m/checkpoints/epoch-1/step-N"
cfg = json.loads(open(f"{ckpt}/lora_config.json").read())
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-780m", dtype=torch.float32)
model = apply_lora(model, ["in_proj", "out_proj"], cfg["rank"], cfg["alpha"])
state = torch.load(f"{ckpt}/trainable.pt", weights_only=True)
model.load_state_dict(state, strict=False)
```

(The backend's loader, `backend/app/model/lora.py:load_checkpoint`, does the same thing — wrap the model first, then load `trainable.pt` into the wrapped model, not the raw backbone, so a model with extra trainable state outside the backbone still loads correctly.)
