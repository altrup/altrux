"""Model-agnostic chunk execution, generation, scoring, and KL helpers."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run_chunks(model, ids: torch.Tensor, state, chunk_len: int, label: str, keep_logits: bool = True):
    """Forward ``ids`` in chunks, threading and detaching state."""
    import torch

    parts = []
    n = (ids.shape[1] + chunk_len - 1) // chunk_len
    with torch.no_grad():
        for chunk in range(n):
            logits, state = model(ids[:, chunk * chunk_len : (chunk + 1) * chunk_len], state=state)
            state = state.detach()
            if keep_logits:
                parts.append(logits.detach().to("cpu"))
            print(f"\r[{_timestamp()}]  {label}: chunk {chunk + 1}/{n}", end="", flush=True)
    print()
    return (torch.cat(parts, dim=1) if keep_logits else None), state


def generate(model, prompt: torch.Tensor, state, n_tokens: int, temperature: float) -> torch.Tensor:
    """Generate an autoregressive continuation from ``prompt`` and ``state``."""
    import torch

    out = []
    with torch.no_grad():
        logits, state = model(prompt, state=state)
        for _ in range(n_tokens):
            last = logits[:, -1].float()
            if temperature <= 0:
                token = last.argmax(dim=-1, keepdim=True)
            else:
                token = torch.multinomial(torch.softmax(last / temperature, dim=-1), num_samples=1)
            out.append(token)
            logits, state = model(token, state=state)
    return torch.cat(out, dim=1)


def target_logprob(model, prompt: torch.Tensor, target: torch.Tensor, state) -> float:
    """Return the teacher-forced mean log-probability per target token."""
    import torch

    sequence = torch.cat([prompt, target], dim=1)
    with torch.no_grad():
        logits, _ = model(sequence, state=state)
    logprobs = torch.log_softmax(logits[0].float(), dim=-1)
    start = prompt.shape[1] - 1
    indices = torch.arange(start, sequence.shape[1] - 1, device=sequence.device)
    return logprobs[indices, target[0]].mean().item()


def replay_step(step: int, n_chunks: int, fresh_state_replay: bool) -> tuple[int, bool]:
    """Return the replay chunk index and whether its state must be reset."""
    chunk = step % n_chunks
    return chunk, (fresh_state_replay or chunk == 0)


def kl_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor, temp: float) -> torch.Tensor:
    import torch.nn.functional as F

    teacher = F.log_softmax(teacher_logits.reshape(-1, teacher_logits.shape[-1]).float() / temp, dim=-1)
    student = F.log_softmax(student_logits.reshape(-1, student_logits.shape[-1]).float() / temp, dim=-1)
    return F.kl_div(student, teacher, log_target=True, reduction="batchmean") * temp**2


__all__ = ["generate", "kl_loss", "replay_step", "run_chunks", "target_logprob"]
