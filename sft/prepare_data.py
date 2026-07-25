import argparse
import importlib
import json
import multiprocessing
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

# Add the repo root to sys.path so the models/ package is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.common import build_tokenizer

_worker_tokenizer = None
_worker_max_len = None
_worker_markers: tuple[str, str] | None = None


def _worker_init(model_name: str, max_len: int) -> None:
    global _worker_tokenizer, _worker_max_len, _worker_markers
    # Imported here, not at module top: importing a model package pulls in
    # mamba_ssm, which needs a working GPU even to import -- keeping it lazy
    # keeps format_conversation importable (and testable) without one.
    mod = importlib.import_module(f"models.{model_name}")
    _worker_tokenizer = build_tokenizer(mod)
    _worker_max_len = max_len
    _worker_markers = (mod.USER_OPEN, mod.ASST_OPEN)


def _worker_format(record: dict) -> tuple[list[int], list[bool]]:
    return format_conversation(record.get("messages", []), _worker_tokenizer, _worker_max_len, *_worker_markers)


def format_conversation(
    messages: list[dict], tokenizer, max_len: int, user_open: str, asst_open: str
) -> tuple[list[int], list[bool]]:
    ids: list[int] = []
    mask: list[bool] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            text = user_open + " " + content + "\n"
            turn_ids = tokenizer.encode(text, add_special_tokens=False)
            turn_mask = [False] * len(turn_ids)
        elif role == "assistant":
            text = asst_open + " " + content
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
    parser.add_argument("--max-len", type=int, default=32768, help="Max tokens per example (whole-turn truncation; chunked training handles long examples, so this only guards pathological outliers)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel tokenization workers")
    args = parser.parse_args()

    all_ids: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    skipped = 0

    records = iter_records(args)
    model_name = os.getenv("MODEL_NAME", "mamba2_780m")
    model_mod = importlib.import_module(f"models.{model_name}")

    if args.workers > 1:
        with multiprocessing.Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(model_name, args.max_len),
        ) as pool:
            for i, (ids, mask) in enumerate(pool.imap(_worker_format, records), 1):
                print(f"\r{i}/{len(records)}", end="", flush=True)
                if not any(mask):
                    skipped += 1
                    continue
                all_ids.append(torch.tensor(ids, dtype=torch.long))
                all_masks.append(torch.tensor(mask, dtype=torch.bool))
    else:
        tokenizer = build_tokenizer(model_mod)
        for i, record in enumerate(records, 1):
            print(f"\r{i}/{len(records)}", end="", flush=True)
            ids, mask = format_conversation(
                record.get("messages", []), tokenizer, args.max_len, model_mod.USER_OPEN, model_mod.ASST_OPEN
            )
            if not any(mask):
                skipped += 1
                continue
            all_ids.append(torch.tensor(ids, dtype=torch.long))
            all_masks.append(torch.tensor(mask, dtype=torch.bool))
    print()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": all_ids, "masks": all_masks}, out)
    print(f"Saved {len(all_ids)} examples to {out} ({skipped} skipped — no assistant turns)")


if __name__ == "__main__":
    main()
