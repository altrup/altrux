"""Conversational wake and read-only dream diagnostics for LAMA-CKL."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Sequence

from experiments.dreams.cache import copy_fraction
from experiments.dreams.generation import sample_next
from experiments.dreams.probes import longest_verbatim_run
from progress import fmt_duration, ts


DREAM_INSTRUCTION = (
    "Dream about the preceding experience. Rehearse what matters without copying it verbatim."
)


def _encode(tokenizer, text: str) -> list[int]:
    return [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]


def dream_instruction_ids(tokenizer, user_open: str, asst_open: str, device):
    import torch

    text = f"{user_open} {DREAM_INSTRUCTION}{asst_open} "
    return torch.tensor([_encode(tokenizer, text)], dtype=torch.long, device=device)


def run_conversational_wake(model, tokenizer, documents: Sequence[str], state,
                            user_open: str, asst_open: str, eoc: str, *,
                            reply_tokens: int, evidence_tokens: int) -> tuple[dict[str, object], object]:
    """Feed one document per user turn and greedily generate each assistant turn."""
    import torch

    if reply_tokens < 1 or evidence_tokens < 1:
        raise ValueError("reply_tokens and evidence_tokens must be positive")
    eoc_id = int(tokenizer.convert_tokens_to_ids(eoc))
    eos_id = int(tokenizer.eos_token_id)
    transcript: list[int] = []
    turns: list[dict[str, object]] = []
    started = time.time()
    model.eval()
    with torch.no_grad():
        for index, document in enumerate(documents):
            evidence = tokenizer(
                " " + document,
                add_special_tokens=False,
                truncation=True,
                max_length=evidence_tokens,
            )["input_ids"]
            prompt_ids = (
                _encode(tokenizer, user_open)
                + [int(token) for token in evidence]
                + _encode(tokenizer, f"{asst_open} ")
            )
            prompt = torch.tensor([prompt_ids], dtype=torch.long,
                                  device=next(model.parameters()).device if hasattr(model, "parameters") else "cpu")
            logits, state = model(prompt, state=state)
            reply: list[int] = []
            for _ in range(reply_tokens):
                token = int(sample_next(logits[:, -1], 0.0).item())
                if token == eoc_id:
                    raise RuntimeError(f"assistant emitted EOC inside wake turn {index + 1}")
                reply.append(token)
                value = torch.tensor([[token]], dtype=torch.long, device=prompt.device)
                logits, state = model(value, state=state)
                if token == eos_id:
                    break
            else:
                raise RuntimeError(
                    f"assistant turn {index + 1} exhausted {reply_tokens} tokens before EOS"
                )
            transcript.extend(prompt_ids + reply)
            turns.append({
                "document": document,
                "prompt_token_ids": prompt_ids,
                "assistant_token_ids": reply,
                "assistant": tokenizer.decode(reply[:-1], skip_special_tokens=True),
            })
            count = index + 1
            elapsed = time.time() - started
            if count <= 3:
                print(f"[{ts()}] wake sample {count}: document={document[:300]!r} "
                      f"assistant={turns[-1]['assistant']!r}", flush=True)
            if count % 10 == 0 or count == len(documents):
                print(f"[{ts()}] wake {count}/{len(documents)} {count / elapsed:.2f} turn/s ETA "
                      f"{fmt_duration(elapsed / count * (len(documents) - count))}", flush=True)
        closing = torch.tensor([[eoc_id]], dtype=torch.long,
                               device=next(model.parameters()).device if hasattr(model, "parameters") else "cpu")
        _, state = model(closing, state=state)
    transcript.append(eoc_id)
    state = state.detach()
    return {
        "turns": turns,
        "transcript_token_ids": transcript,
        "transcript_sha256": hashlib.sha256(
            ",".join(str(token) for token in transcript).encode()
        ).hexdigest(),
        "invariants": {
            "turns": len(turns),
            "missing_assistant_eos": sum(turn["assistant_token_ids"][-1] != eos_id for turn in turns),
            "internal_eoc": sum(eoc_id in turn["assistant_token_ids"] for turn in turns),
            "closing_eoc": int(transcript[-1] == eoc_id and transcript.count(eoc_id) == 1),
        },
    }, state


def lama_dream_diagnostics(dreams, rows: Sequence[dict[str, object]],
                           transcript_ids: Sequence[int]) -> dict[str, object]:
    """Measure dream bindings and copying without changing generation or training."""
    correct = misbound = 0
    subjects = [str(row["subject"]) for row in rows]
    objects = [str(row["object"]) for row in rows]
    copy_fractions: list[float] = []
    longest: list[int] = []
    lengths: list[int] = []
    for dream in dreams:
        text = "".join(dream.token_texts)
        sentences = re.split(r"[.\n]", text)
        bound: set[int] = set()
        for sentence in sentences:
            low = sentence.lower()
            present_objects = {index for index, obj in enumerate(objects) if obj in sentence}
            if not present_objects:
                continue
            for index, subject in enumerate(subjects):
                if subject.lower() not in low:
                    continue
                if index in present_objects:
                    bound.add(index)
                elif present_objects - {index}:
                    misbound += 1
        correct += len(bound)
        generated = dream.dream_ids[dream.prefix_len:]
        copy_fractions.append(copy_fraction(generated, transcript_ids))
        longest.append(longest_verbatim_run(generated, transcript_ids, n=1))
        lengths.append(len(generated))
    hashes = [dream.dream_sha for dream in dreams]
    return {
        "correct_bindings": correct,
        "misbindings": misbound,
        "contradictions": misbound,
        "unique_dreams": len(set(hashes)),
        "duplicate_dreams": len(hashes) - len(set(hashes)),
        "eoc_termination_rate": (
            sum(dream.stop_reason == "eoc" for dream in dreams) / len(dreams) if dreams else 0.0
        ),
        "mean_generated_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        "min_generated_tokens": min(lengths, default=0),
        "max_generated_tokens": max(lengths, default=0),
        "mean_copy_fraction": sum(copy_fractions) / len(copy_fractions) if copy_fractions else 0.0,
        "max_copy_fraction": max(copy_fractions, default=0.0),
        "max_verbatim_run": max(longest, default=0),
    }


__all__ = [
    "DREAM_INSTRUCTION",
    "dream_instruction_ids",
    "lama_dream_diagnostics",
    "run_conversational_wake",
]
