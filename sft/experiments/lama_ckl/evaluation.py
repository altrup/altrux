"""Cross-tokenizer implementation of the published object-token metric."""

from __future__ import annotations

import time
from collections.abc import Sequence

from progress import fmt_duration, ts


def last_object_token_positions(
    text: str,
    obj: str,
    offsets: Sequence[tuple[int, int]],
) -> list[int]:
    """Return token indices overlapping the object's last character occurrence."""
    start = text.rfind(obj)
    if start < 0:
        raise ValueError(f"object {obj!r} does not occur in task {text!r}")
    end = start + len(obj)
    positions = [index for index, (lo, hi) in enumerate(offsets) if lo < end and hi > start]
    if not positions or positions[0] == 0:
        raise ValueError(f"object {obj!r} has no causally scoreable token span")
    return positions


def object_token_accuracy(logits, input_ids, positions: Sequence[Sequence[int]]) -> list[float]:
    """Score argmax next-token accuracy over each aligned object span."""
    predicted = logits[:, :-1].argmax(dim=-1)
    scores: list[float] = []
    for row, indices in enumerate(positions):
        correct = [
            int(predicted[row, index - 1]) == int(input_ids[row, index])
            for index in indices
        ]
        scores.append(sum(correct) / len(correct))
    return scores


def encode_metric_batch(tokenizer, rows: Sequence[dict[str, object]], task_key: str,
                        max_length: int, device):
    """Tokenize and pad tasks while preserving object character alignment."""
    import torch

    encoded_rows: list[list[int]] = []
    positions: list[list[int]] = []
    for row in rows:
        text, obj = str(row[task_key]), str(row["object"])
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        ids = [int(token) for token in encoded["input_ids"]]
        offsets = [(int(lo), int(hi)) for lo, hi in encoded["offset_mapping"]]
        encoded_rows.append(ids)
        positions.append(last_object_token_positions(text, obj, offsets))
    width = max(map(len, encoded_rows))
    pad_id = int(tokenizer.pad_token_id)
    batch = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    for row, ids in enumerate(encoded_rows):
        batch[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return batch, positions


def score_records(model, tokenizer, rows: Sequence[dict[str, object]], task_key: str,
                  batch_size: int, max_length: int, device) -> list[float]:
    """Score records in GPU batches from fresh recurrent state."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scores: list[float] = []
    started = time.time()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            ids, positions = encode_metric_batch(tokenizer, batch_rows, task_key, max_length, device)
            logits, _ = model(ids, state=None)
            scores.extend(object_token_accuracy(logits, ids, positions))
            count = len(scores)
            elapsed = time.time() - started
            zero = sum(score == 0.0 for score in scores)
            one = sum(score == 1.0 for score in scores)
            print(
                f"\r[{ts()}] {task_key} {count}/{len(rows)} zero={zero} one={one} "
                f"{count / elapsed:.2f} item/s ETA "
                f"{fmt_duration(elapsed / count * (len(rows) - count))}",
                end="" if count < len(rows) else "\n",
                flush=True,
            )
    return scores


__all__ = [
    "encode_metric_batch",
    "last_object_token_positions",
    "object_token_accuracy",
    "score_records",
]
