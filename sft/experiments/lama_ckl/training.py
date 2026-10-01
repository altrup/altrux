"""Experiment: lama-ckl

Minimal causal-LM document training used by LAMA-CKL baseline arms.
"""

import time
import torch
import torch.nn.functional as F

from collections.abc import Sequence
from progress import fmt_duration, ts


def epoch_batches(size: int, batch_size: int, seed: int) -> list[list[int]]:
    """
    Shuffle the indices 0..size-1 and cut them into batches.

    size: number of documents
    batch_size: documents per optimizer step
    seed: used to make shuffling deterministic

    Returns one list of document indices per optimizer step
    Every index appears exactly once. Raises ValueError if size or batch_size < 1
    """

    if size < 1:
        raise ValueError(f"Invalid size: {size}")
    elif batch_size < 1:
        raise ValueError(f"Invalid batch_size: {batch_size}")

    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return [order[i : i + batch_size] for i in range(0, size, batch_size)]


def train_document_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    documents: Sequence[Sequence[int]],
    batches: Sequence[Sequence[int]],
    *,
    pad_id: int,
    device: str | torch.device,
    label: str,
) -> dict[str, float | int]:
    """
    One epoch of next-token training over documents, one optimizer step per batch

    documents: list of documents to train on
    batches: output of epoch_batches
    pad_id: token to pad documents with to fit into batch size
    device: device to train on, where tensors are built
    label: prefix for the per-batch progress line

    Returns {
            "optimizer_steps": number of optimizer steps,
            "token_gradients": total number of tokens trained on,
            "seconds": total time spent training
    }
    """

    start_time = time.time()
    token_gradients = 0

    model.train()
    for c, batch in enumerate(batches):
        rows = [documents[i] for i in batch]
        width = max(map(len, rows))

        ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(rows), width - 1), dtype=torch.bool, device=device)
        for r, tokens in enumerate(rows):
            ids[r, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            mask[r, : len(tokens) - 1] = True

        logits, _ = model(ids, state=None)
        losses = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1),
            reduction="none",
        ).reshape_as(mask)

        loss = losses[mask].mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        token_gradients += int(mask.sum())

        elapsed = time.time() - start_time
        count = c + 1
        print(
            f"[{ts()}] {label} batch {count}/{len(batches)} loss {loss.item():.4f} "
            f"{count / elapsed:.2f} batch/s ETA "
            f"{fmt_duration(elapsed / count * (len(batches) - count))}",
            flush=True,
        )

    return {
        "optimizer_steps": len(batches),
        "token_gradients": token_gradients,
        "seconds": time.time() - start_time,
    }
