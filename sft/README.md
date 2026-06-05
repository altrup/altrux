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

To use a local JSONL file instead (one `{"messages": [...]}` object per line):

```bash
make prepare ARGS="--input data/raw.jsonl"
```

Loss is computed on assistant turns only; user tokens are masked out.

See `python prepare_data.py --help` for all options (`--max-examples`, `--hf-split`, `--max-len`, etc.).

## Training

```bash
make train    # start fresh
make resume   # resume from latest checkpoint
```

Checkpoints are saved every 100 steps to `checkpoints/step-N/` (LoRA adapter + optimizer state). Only the last 3 checkpoints are kept; older ones are deleted automatically.

See `python train.py --help` for all options (learning rate, rank, accumulation steps, etc.).

## Using the adapter

Each checkpoint directory contains `lora_config.json` with the rank and alpha used during training, so callers don't need to hard-code them:

```python
import json
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from lora import apply_lora, load_lora

ckpt = "checkpoints/step-N"
cfg = json.loads(open(f"{ckpt}/lora_config.json").read())
model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-780m", dtype=torch.float32)
model = apply_lora(model, ["in_proj", "out_proj"], cfg["rank"], cfg["alpha"])
load_lora(model, ckpt)
```
