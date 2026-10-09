"""Dream acceptance, leakage, and artifact diagnostics."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

from experiments.dream_generation import copy_state, state_to
from experiments.dream_types import DreamCache, DreamSetCache
from experiments.dreams.cache import (
    assert_aggregate_binding,
    binding_coverage,
    fact_read_positions,
    gate_agreement,
)
from experiments.dreams.generation import frozen_teacher
from experiments.erasure.gating import VARIANTS
from experiments.erasure.probe import group_by_layer
from experiments.facts import extract_answer, normalize
from experiments.inference import run_chunks
from experiments.verbatim import copy_fraction, longest_verbatim_run
from progress import ts


def probe_leakage(
    items,
    answer_probe,
    emit,
    arm: str,
    wave: int,
    phase: str,
    stops: Sequence[str],
    baseline: dict[str, float] | None = None,
    step: int | None = None,
) -> dict[str, float]:
    """Score fresh-state answers for distractor leakage."""
    logprobs: dict[str, float] = {}
    hits = 0
    for i, item in enumerate(items):
        generation, logprob = answer_probe(item.prompt, item.answer.strip())
        answer, truth = extract_answer(generation, stops), normalize(item.answer)
        matched = answer == truth or answer.startswith(truth + " ")
        logprobs[item.label] = logprob
        hits += matched
        delta = logprob - baseline[item.label] if baseline and item.label in baseline else None
        emit(
            {
                "phase": phase,
                "wave": wave,
                "arm": arm,
                "step": step,
                "item": item.label,
                "kind": item.cls,
                "answer": item.answer.strip(),
                "greedy": generation,
                "match": matched,
                "logprob": logprob,
                "logprob_delta": delta,
            }
        )
        print(
            f"[{ts()}]  {phase} w{wave}{'' if step is None else f' s{step}'} {item.label:<11} "
            f"{'HIT ' if matched else 'miss'} lp {logprob:+.3f}"
            f"{'' if delta is None else f' (d {delta:+.3f})'} "
            f"| running leak {hits / (i + 1):.2f} | {generation[:40]!r}",
            flush=True,
        )
    return logprobs


def report_dream(cache: DreamCache) -> None:
    """Print binding-aware coverage and decoded dream text."""
    facts = cache.fact_list
    text = "".join(cache.token_texts)
    bound, misbound = binding_coverage(text, facts)
    print(f"[{ts()}] dream ended: {cache.stop_reason}  ({len(cache.dream_ids)} tokens)")
    print(f"[{ts()}] bound rehearsals {bound}  (misbound: {misbound})")
    print(
        f"[{ts()}] bound code coverage {sum(v > 0 for v in bound.values())}/{len(facts)}, "
        f"{cache.free_tokens}/{len(cache.dream_ids)} tokens freely generated"
    )
    joint = next(
        (
            i
            for i in range(1, len(cache.cue_flags))
            if cache.cue_flags[i] and not cache.cue_flags[i - 1]
        ),
        None,
    )
    if joint is None:
        print(f"[{ts()}] no cue joint in this dream (--cue-every 0?)")
    else:
        lo, hi = max(0, joint - 24), min(len(cache.token_texts), joint + 40)
        print(
            f"[{ts()}] decoded around cue joint at token {joint}:\n"
            f"  ...{''.join(cache.token_texts[lo:joint])!r} >>CUE>> "
            f"{''.join(cache.token_texts[joint:hi])!r}..."
        )
    print(f"[{ts()}] decoded dream:\n{text!r}")
    if sum(v > 0 for v in bound.values()) < len(facts):
        print(
            f"[{ts()}] WARNING: a fact this dream never binds is one no arm can install. "
            "Regenerate this seed at a tighter --cue-every before running the grid."
        )


def dream_is_degenerate(text: str) -> bool:
    """Reject empty, replacement-character, or non-ASCII-flood dreams."""
    if not text or "�" in text:
        return True
    return sum(ord(char) > 127 for char in text) / len(text) > 0.2


def blank_state_logits(model, tokens, chunk_len: int, frozen: bool):
    """Re-score dream tokens from a blank state."""
    with frozen_teacher(model) if frozen else contextlib.nullcontext():
        logits, _ = run_chunks(model, tokens, None, chunk_len, "blank re-score", keep_logits=True)
    return logits[0].float()


def basis_overlap(a, b) -> float:
    """Measure the mean squared principal cosine of two bases."""
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float((a.float() @ b.float().T).pow(2).sum() / min(a.shape[0], b.shape[0]))


def report_dream_set(cache: DreamSetCache, min_dreams: int, rank_rule: str) -> dict[str, object]:
    """Print and gate the decoded, bound, and eraser diagnostics for a set."""
    facts = cache.fact_list
    print(f"\n[{ts()}] === dream set: {len(cache.dreams)} dreams, set_sha {cache.set_sha[:12]} ===")
    reasons = {
        reason: sum(d.stop_reason == reason for d in cache.dreams)
        for reason in ("eoc", "max-tokens", "turn-backstop")
    }
    print(f"[{ts()}] termination reasons: {reasons}")
    gateless = sum(not d.gate_positions for d in cache.dreams)
    if gateless:
        print(
            f"[{ts()}] dreams with an empty gate (empty eraser, no denial pressure): "
            f"{gateless} of {len(cache.dreams)}"
        )
    for i, dream in enumerate(cache.dreams):
        bound, misbound = binding_coverage("".join(dream.token_texts), facts)
        agreement = gate_agreement(
            dream.gate_positions, fact_read_positions(dream.token_texts, facts)
        )
        sizes = [len(b) for b in dream.bases[VARIANTS[0]]]
        print(
            f"[{ts()}]  dream {i}: {len(dream.dream_ids)} tokens, ended {dream.stop_reason}, "
            f"{len(dream.gate_positions)} gated positions, basis rank "
            f"{min(sizes)}-{max(sizes)} over {len(sizes)} layers ({rank_rule})"
        )
        copied = copy_fraction(
            dream.dream_ids[dream.prefix_len :],
            cache.transcript_ids,
            cue_flags=dream.cue_flags[dream.prefix_len :],
        )
        print(
            f"[{ts()}]    within-dream repeats {bound}  (misbound {misbound})  "
            f"verbatim-copied from the wake transcript: {copied:.1%}"
        )
        print(
            f"[{ts()}]    gate vs binding scan: precision {agreement['precision']:.2f} "
            f"recall {agreement['recall']:.2f}, per-fact contribution {agreement['per_fact']}"
        )
        for entity, n in agreement["per_fact"].items():
            if n == 0:
                print(
                    f"[{ts()}]    NOTE: the gate captured no read of {entity} in this dream -- "
                    "the eraser cannot address what it never captured."
                )
    copies = [
        copy_fraction(
            d.dream_ids[d.prefix_len :], cache.transcript_ids, cue_flags=d.cue_flags[d.prefix_len :]
        )
        for d in cache.dreams
    ]
    longest = [
        longest_verbatim_run(d.dream_ids[d.prefix_len :], cache.transcript_ids)
        for d in cache.dreams
    ]
    print(
        f"[{ts()}] longest verbatim run per dream: max {max(longest)} tokens "
        "(a run of hundreds is the transcript being REPLAYED; ~15 is a reused sentence)"
    )
    print(
        f"[{ts()}] verbatim copying of the wake transcript: mean {sum(copies) / len(copies):.1%}, "
        f"max {max(copies):.1%}  (a dream that replays the wake is not a dream -- watch this "
        "when the warm start is recall-heavy)"
    )
    for variant in VARIANTS:
        overlaps = [
            basis_overlap(a.bases[variant][i], b.bases[variant][i])
            for a, b in zip(cache.dreams, cache.dreams[1:], strict=False)
            for i in range(len(a.bases[variant]))
        ]
        if overlaps:
            print(
                f"[{ts()}] cross-dream V-overlap ({variant}): mean {sum(overlaps) / len(overlaps):.3f}, "
                f"max {max(overlaps):.3f}"
            )
    first = cache.dreams[0]
    print(
        f"[{ts()}] decoded start of dream 0 (prefix + first free tokens):\n"
        f"  >>PREFIX>> {''.join(first.token_texts[: first.prefix_len])!r} "
        f">>FREE>> {''.join(first.token_texts[first.prefix_len : first.prefix_len + 48])!r}"
    )
    counts = assert_aggregate_binding(cache.dreams, facts, min_dreams)
    print(f"[{ts()}] aggregate binding gate PASSED (>= {min_dreams} dreams per fact): {counts}")
    return {"stop_reasons": reasons, "binding": counts}


def battery_read_queries(model, items, wake_state, encode, n_layers: int):
    """Capture per-position, per-layer read queries for the offline battery."""
    import torch

    out: dict[str, list[list[torch.Tensor]]] = {}
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, item in enumerate(items):
            prompt = str(item["prompt"])
            ids = encode(prompt).to(device)
            state = state_to(copy_state(wake_state), device)
            positions: list[list[torch.Tensor]] = []
            for t in range(ids.shape[1]):
                model.c_capture = []
                _, state = model(ids[:, t : t + 1], state=state)
                positions.append(
                    [c.half().cpu() for c in group_by_layer(model.c_capture, n_layers)[0]]
                )
                model.c_capture = None
            out[prompt] = positions
            print(
                f"[{ts()}]  battery read queries {i + 1}/{len(items)}: {len(positions)} positions "
                f"from {prompt[:40]!r}",
                flush=True,
            )
    return out


__all__ = [
    "basis_overlap",
    "battery_read_queries",
    "blank_state_logits",
    "dream_is_degenerate",
    "probe_leakage",
    "report_dream",
    "report_dream_set",
]
