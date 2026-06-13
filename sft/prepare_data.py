import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

# Add the repo root to sys.path so the models/ package is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

_model_mod = importlib.import_module(f"models.{os.getenv('MODEL_NAME', 'mamba2_780m')}")
TOKENIZER_ID = _model_mod.TOKENIZER_ID
USER_OPEN = _model_mod.USER_OPEN
ASST_OPEN = _model_mod.ASST_OPEN


def format_conversation(
    messages: list[dict], tokenizer, max_len: int
) -> tuple[list[int], list[bool]]:
    ids: list[int] = []
    mask: list[bool] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            text = USER_OPEN + content + "\n"
            turn_ids = tokenizer.encode(text, add_special_tokens=False)
            turn_mask = [False] * len(turn_ids)
        elif role == "assistant":
            text = ASST_OPEN + content
            toks = tokenizer.encode(text, add_special_tokens=False)
            turn_ids = toks + [tokenizer.eos_token_id]
            # "train": false keeps the turn as context but excludes it from the
            # loss (e.g. an earlier answer that a later turn revises)
            trainable = msg.get("train", True)
            turn_mask = [trainable] * len(turn_ids)
        else:
            continue

        if len(ids) + len(turn_ids) > max_len:
            break  # drop this and all remaining turns; keeps only complete turns

        ids.extend(turn_ids)
        mask.extend(turn_mask)

    return ids, mask


def iter_records(args) -> list[dict]:
    if args.hf_dataset:
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, split=args.hf_split)
        if args.max_examples:
            ds = ds.select(range(min(args.max_examples, len(ds))))
        return list(ds)
    else:
        records = []
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize conversation data for training")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Local JSONL file")
    group.add_argument("--hf-dataset", help="HuggingFace dataset ID (e.g. HuggingFaceH4/ultrachat_200k)")
    parser.add_argument("--hf-split", default="train_sft", help="Dataset split (default: train_sft)")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap number of examples loaded")
    parser.add_argument("--output", default="data/train.pt", help="Output .pt file")
    parser.add_argument("--max-len", type=int, default=1024, help="Max tokens per example")
    args = parser.parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})

    all_ids: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    skipped = 0

    for record in iter_records(args):
        ids, mask = format_conversation(record.get("messages", []), tokenizer, args.max_len)
        if not any(mask):
            skipped += 1
            continue
        all_ids.append(torch.tensor(ids, dtype=torch.long))
        all_masks.append(torch.tensor(mask, dtype=torch.bool))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": all_ids, "masks": all_masks}, out)
    print(f"Saved {len(all_ids)} examples to {out} ({skipped} skipped — no assistant turns)")


if __name__ == "__main__":
    main()
