"""Minimal causal-LM document training used by LAMA-CKL baseline arms."""

from __future__ import annotations

import time
from collections.abc import Sequence

from progress import fmt_duration, ts


def epoch_batches(size: int, batch_size: int, seed: int) -> list[list[int]]:
    import torch

    if size < 1 or batch_size < 1:
        raise ValueError("size and batch_size must be positive")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return [order[start:start + batch_size] for start in range(0, size, batch_size)]


def train_document_epoch(model, optimizer, documents: Sequence[Sequence[int]],
                         batches: Sequence[Sequence[int]], *, pad_id: int, device,
                         label: str) -> dict[str, float | int]:
    """Train one causal-LM pass, starting each document batch from fresh state."""
    import torch
    import torch.nn.functional as F

    model.train()
    started = time.time()
    token_gradients = 0
    for step, indices in enumerate(batches):
        rows = [documents[index] for index in indices]
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(rows), width - 1), dtype=torch.bool, device=device)
        for row, tokens in enumerate(rows):
            ids[row, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            mask[row, :len(tokens) - 1] = True
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
        count = step + 1
        elapsed = time.time() - started
        print(f"[{ts()}] {label} batch {count}/{len(batches)} loss {loss.item():.4f} "
              f"{count / elapsed:.2f} batch/s ETA "
              f"{fmt_duration(elapsed / count * (len(batches) - count))}", flush=True)
    return {
        "optimizer_steps": len(batches),
        "token_gradients": token_gradients,
        "seconds": time.time() - started,
    }


__all__ = ["epoch_batches", "train_document_epoch"]
