"""Experiment: lama-ckl

Conversational wake, dream prompt, and read-only dream diagnostics
"""

import hashlib
import re
import time
from collections.abc import Mapping, Sequence
from typing import Protocol, Self, TypedDict

import torch

from experiments.dreams.cache import copy_fraction
from experiments.dreams.generation import copy_state, sample_next
from experiments.dreams.probes import longest_verbatim_run
from progress import fmt_duration, ts

# Can use {eoc}, {user_open}, and {asst_open}
DREAM_PROMPT = "{eoc}"


class Tokenizer(Protocol):
    """The slice of a HF tokenizer the wake uses."""

    eos_token_id: int

    def __call__(
        self, text: str, *, add_special_tokens: bool, truncation: bool = ..., max_length: int = ...
    ) -> Mapping[str, Sequence[int]]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...


class State(Protocol):
    """Recurrent state that can be cut from the autograd graph."""

    def detach(self) -> Self: ...


class Model(Protocol):
    """A model that continues from a given recurrent state."""

    def eval(self) -> object: ...

    def __call__(
        self, ids: torch.Tensor, state: State | None = None
    ) -> tuple[torch.Tensor, State]: ...


class DreamRecord(Protocol):
    """The fields of a generated dream the diagnostics read."""

    token_texts: Sequence[str]
    dream_ids: Sequence[int]
    prefix_len: int
    stop_reason: str
    dream_sha: str


class DreamDiagnostics(TypedDict):
    correct_bindings: int
    misbindings: int
    contradictions: int
    unique_dreams: int
    duplicate_dreams: int
    eoc_termination_rate: float
    mean_generated_tokens: float
    min_generated_tokens: int
    max_generated_tokens: int
    mean_copy_fraction: float
    max_copy_fraction: float
    max_verbatim_run: int


def _encode(tokenizer: Tokenizer, text: str) -> list[int]:
    """Token ids of text with no special tokens added"""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def dream_prompt_ids(
    tokenizer: Tokenizer, user_open: str, asst_open: str, eoc: str, device: str | torch.device
) -> torch.Tensor:
    """
    Returns the prompt used to initiate dreams as a tokenized tensor

    Text is DREAM_PROMPT
    Returns a (1, width) long tensor on device
    """
    return torch.tensor(
        [
            _encode(
                tokenizer, DREAM_PROMPT.format(user_open=user_open, asst_open=asst_open, eoc=eoc)
            )
        ],
        device=device,
    )


def apply_dream_prompt(
    model: Model,
    tokenizer: Tokenizer,
    state: State,
    user_open: str,
    asst_open: str,
    eoc: str,
    device: str | torch.device,
) -> State:
    """
    Feed DREAM_PROMPT into a copy of the open wake state

    The input state is not changed. Runs under no_grad. Returns the detached
    state after the prompt, which the next wake continues from.
    """

    with torch.no_grad():
        _, state = model(
            dream_prompt_ids(tokenizer, user_open, asst_open, eoc, device),
            state=copy_state(state),
        )
    return state.detach()


def lama_dream_diagnostics(
    dreams: Sequence[DreamRecord],
    rows: Sequence[Mapping[str, object]],
    transcript_ids: Sequence[int],
) -> DreamDiagnostics:
    """
    Count fact bindings and copying in the dreams. Reads only, changes nothing

    dreams: the generated dream set
    rows: the facts of this cycle, each with "subject" and "object"
    transcript_ids: the wake transcript tokens

    Per dream, join token_texts and split into sentences on "." and newline.
    In a sentence that contains at least one object (case-sensitive), for each
    subject in it (case-insensitive):
      own object present   -> correct binding, counted once per dream per row
      own object absent    -> misbinding, counted every time
    The generated part of a dream is dream_ids[prefix_len:].

    Returns DreamDiagnostics dict
    """
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
        generated = dream.dream_ids[dream.prefix_len :]
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


def run_conversational_wake(
    model: Model,
    tokenizer: Tokenizer,
    documents: Sequence[str],
    state: State | None,
    user_open: str,
    asst_open: str,
    eoc: str,
    *,
    reply_tokens: int,
    evidence_tokens: int,
    device: str | torch.device,
) -> tuple[dict[str, object], State]:
    """
    Feed one document per user turn, then generate each assistant reply greedily

    documents: one evidence text per turn
    state: recurrent state to continue from, None for fresh
    eoc: the conversation-boundary marker string
    reply_tokens: max tokens per reply; evidence_tokens: truncation limit per document
    Raises ValueError if either limit is < 1

    Per turn the prompt is user_open + " " + document (truncated) + asst_open + " ".
    The reply is argmax tokens until EOS. Each token, EOS included, goes back
    into the model so the state holds the full turn.
    Raises RuntimeError if a reply emits EOC, or has no EOS within reply_tokens.

    Runs in eval mode under no_grad on the model's device.
    Prints the first 3 turns decoded, and a rate/ETA line
    every 10 turns and at the end.

    Returns (artifact, state). artifact has:
      turns: per turn document, prompt_token_ids, assistant_token_ids (EOS
             included), assistant (decoded)
      transcript_token_ids: all prompts and replies in order
      transcript_sha256: sha256 of the ids joined with ","
      invariants: turns, missing_assistant_eos, internal_eoc, closing_eoc
                  (1 when EOC is last and occurs once)
    """
    if reply_tokens < 1 or evidence_tokens < 1:
        raise ValueError("reply_tokens and evidence_tokens must be at least 1")

    eoc_id = tokenizer.convert_tokens_to_ids(eoc)
    eos_id = tokenizer.eos_token_id

    turns: list[dict[str, object]] = []

    started = time.time()
    model.eval()
    current_state = state
    with torch.no_grad():
        for index, document in enumerate(documents):
            evidence = tokenizer(
                " " + document,
                add_special_tokens=False,
                truncation=True,
                max_length=evidence_tokens,
            )["input_ids"]
            prompt_ids = (
                _encode(tokenizer, user_open) + list(evidence) + _encode(tokenizer, f"{asst_open} ")
            )
            logits, current_state = model(
                torch.tensor([prompt_ids], dtype=torch.long, device=device), current_state
            )

            reply: list[int] = []
            for _ in range(reply_tokens):
                token = int(sample_next(logits[:, -1], 0.0).item())
                if token == eoc_id:
                    raise RuntimeError("Model generated <eoc> token")
                reply.append(token)
                logits, current_state = model(
                    torch.tensor([[token]], dtype=torch.long, device=device), state=current_state
                )
                if token == eos_id:
                    break
            else:
                raise RuntimeError("Reached reply tokens limit without generated <eos> token")

            turns.append(
                {
                    "document": document,
                    "prompt_token_ids": prompt_ids,
                    "assistant_token_ids": reply,
                    "assistant": tokenizer.decode(reply),
                }
            )

            count = index + 1
            elapsed = time.time() - started
            if count <= 3:
                print(
                    f"[{ts()}] wake sample {count}: document={document[:300]!r} "
                    f"assistant={turns[-1]['assistant']!r}",
                    flush=True,
                )
            if count % 10 == 0 or count == len(documents):
                print(
                    f"[{ts()}] wake {count}/{len(documents)} {count / elapsed:.2f} turn/s ETA "
                    f"{fmt_duration(elapsed / count * (len(documents) - count))}",
                    flush=True,
                )

    transcript_token_ids = [
        token for turn in turns for token in turn["prompt_token_ids"] + turn["assistant_token_ids"]
    ]
    return {
        "turns": turns,
        "transcript_token_ids": transcript_token_ids,
        "transcript_sha256": hashlib.sha256(
            ",".join(str(t) for t in transcript_token_ids).encode()
        ).hexdigest(),
        "invariants": {
            "turns": len(turns),
            "missing_assistant_eos": sum(
                turn["assistant_token_ids"][-1] != eos_id for turn in turns
            ),
            "internal_eoc": sum(eoc_id in turn["assistant_token_ids"] for turn in turns),
        },
    }, current_state.detach()
