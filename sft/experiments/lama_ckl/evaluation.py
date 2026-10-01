"""Experiment: lama-ckl

Scorer for LAMA-CKL object-token metric
"""

import time
from collections.abc import Mapping, Sequence
from typing import Protocol

import torch

from progress import fmt_duration, ts


class Tokenizer(Protocol):
    """The slice of a HF tokenizer the scorer uses."""

    pad_token_id: int

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
        return_offsets_mapping: bool,
    ) -> Mapping[str, Sequence[object]]: ...


class Scorable(Protocol):
    """A model that maps ids to logits from fresh state."""

    def eval(self) -> object: ...

    def __call__(
        self, ids: torch.Tensor, state: object | None = None
    ) -> tuple[torch.Tensor, object]: ...


def last_object_token_positions(
    text: str, obj: str, offsets: Sequence[tuple[int, int]]
) -> list[int]:
    """
    Get token indices of the last occurence of obj in provided text

    text: the sentence
    obj: the object string to score
    offsets: [start, end) span of each token

    Returns the indices in ascending order
    Raises ValueError if obj is not in text, or if span starts at token 0
    """
    start = text.rfind(obj)
    if start < 0:
        raise ValueError(f"{obj} not found in text")

    end = start + len(obj)
    token_positions = [i for i, (low, high) in enumerate(offsets) if low < end and high > start]
    if len(token_positions) == 0 or token_positions[0] == 0:
        raise ValueError(f"{obj} in invalid position")
    return token_positions


def object_token_accuracy(
    logits: torch.Tensor, input_ids: torch.Tensor, positions: Sequence[Sequence[int]]
) -> list[float]:
    """
    Argmax next-token accuracy at the specified positions in a row's object span

    logits: (rows, width, vocab) model output
    input_ids: (rows, width) the tokens that produced logits
    positions: per row, the token indices to score

    A token at index is correct when the generated token is equal to the input_id token
    Returns percent of obj tokens generated correctly per row
    """

    predicted = logits[:, :-1].argmax(dim=-1)
    scores = []
    for row, tokens in enumerate(positions):
        correct = [int(predicted[row, i - 1]) == int(input_ids[row, i]) for i in tokens]
        scores.append(sum(correct) / len(correct))
    return scores


def encode_metric_batch(
    tokenizer: Tokenizer,
    rows: Sequence[Mapping[str, object]],
    task_key: str,
    max_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, list[list[int]]]:
    """
    Tokenize rows[task_key] into one padded batch and locate each object span

    tokenizer: provides pad_token_id and offset mapping
    rows: records with task_key and "object" fields
    task_key: which task text to score
    max_length: truncation limit in tokens
    device: the device where the batch tensor is built

    Returns (ids, positions): ids is (batch, width) padded on the right with pad_token_id
                              positions is one list per row from last_object_token_positions
    """
    ids: list[list[int]] = []
    positions: list[list[int]] = []
    for row in rows:
        text = str(row[task_key])
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        token_ids = list(encoding["input_ids"])
        ids.append(token_ids)
        positions.append(
            last_object_token_positions(text, str(row["object"]), encoding["offset_mapping"])
        )

    width = max(len(t) for t in ids)
    padded = [t + [tokenizer.pad_token_id] * (width - len(t)) for t in ids]
    return torch.tensor(padded, device=device), positions


def score_records(
    model: Scorable,
    tokenizer: Tokenizer,
    rows: Sequence[Mapping[str, object]],
    task_key: str,
    batch_size: int,
    max_length: int,
    device: str | torch.device,
) -> list[float]:
    """
    Score every row, in batches, from fresh recurrent state

    model: the model to score
    tokenizer, rows, task_key, max_length, device: as encode_metric_batch
    batch_size: rows per forward call; raises ValueError if < 1

    Runs in eval mode under no_grad. Returns one score per row in input
    order. Prints a \\r progress line per batch with counts of 0.0 and 1.0
    scores, rate, and ETA.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size {batch_size} must be at least 1")

    start_time = time.monotonic()
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            ids, positions = encode_metric_batch(
                tokenizer, rows[start:end], task_key, max_length, device
            )
            logits, _ = model(ids, state=None)
            scores.extend(object_token_accuracy(logits, ids, positions))

            elapsed = time.monotonic() - start_time
            rate = end / elapsed if elapsed > 0 else 0.0
            eta = (len(rows) - end) / rate if rate > 0 else 0.0
            print(
                f"\r[{ts()}] scored {end}/{len(rows)}  zeros={scores.count(0.0)} ones={scores.count(1.0)}  {rate:.1f} item/s  ETA {fmt_duration(eta)}",
                end="" if end < len(rows) else "\n",
                flush=True,
            )
    return scores
