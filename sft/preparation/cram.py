"""Shared token-span operations for cram preparation."""

import torch


def find_subsequence(ids: torch.Tensor, pat: torch.Tensor) -> list[int]:
    n, m = len(ids), len(pat)
    if m == 0 or n < m:
        return []
    hit = torch.ones(n - m + 1, dtype=torch.bool)
    for j in range(m):
        hit &= ids[j:n - m + 1 + j] == pat[j]
    return hit.nonzero().flatten().tolist()


__all__ = ["find_subsequence"]
