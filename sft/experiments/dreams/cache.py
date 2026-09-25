"""Experiment: shared

Identity, validation, and filesystem operations for dream caches."""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from experiments.dreams.types import (
    CachedDream,
    DreamCache,
    DreamSetCache,
    dream_set_sha,
    token_sha,
)
from experiments.erasure.gating import (
    RANK_RULES,
    VARIANTS,
    address_budget,
    aggregate_basis,
    rank_median,
    rank_ratio_gap,
    variant_basis,
)
from experiments.facts import Fact
from progress import ts

if TYPE_CHECKING:
    import torch


CUE_STOPS = (".", "\n")
MIN_DISTINCT_DREAM_TOKENS = 8


def fact_read_positions(token_texts: Sequence[str], facts: Sequence[Fact]) -> dict[str, list[int]]:
    """Return token positions inside bound fact-code rehearsals."""
    text = "".join(token_texts)
    spans, pos = [], 0
    for piece in token_texts:
        spans.append((pos, pos + len(piece)))
        pos += len(piece)
    out: dict[str, list[int]] = {}
    for fact in facts:
        hits: set[int] = set()
        at = text.find(fact.code)
        while at != -1:
            start = max(text.rfind(stop, 0, at) for stop in CUE_STOPS) + 1
            ends = [text.find(stop, at) for stop in CUE_STOPS]
            end = min([e for e in ends if e != -1], default=len(text))
            if fact.entity.lower() in text[start:end].lower():
                hits |= {
                    i for i, (lo, hi) in enumerate(spans) if lo < at + len(fact.code) and hi > at
                }
            at = text.find(fact.code, at + 1)
        out[fact.entity] = sorted(hits)
    return out


def gate_agreement(gate: Sequence[int], fact_positions: dict[str, list[int]]) -> dict[str, object]:
    """Compare state-gate positions with the binding validation overlay."""
    reads = {t for positions in fact_positions.values() for t in positions}
    gated = set(gate)
    hit = len(gated & reads)
    return {
        "precision": hit / len(gated) if gated else 0.0,
        "recall": hit / len(reads) if reads else 0.0,
        "gated": len(gated),
        "read_positions": len(reads),
        "per_fact": {e: len(gated & set(p)) for e, p in fact_positions.items()},
    }


def binding_coverage(text: str, facts: Sequence[Fact]) -> tuple[dict[str, int], dict[str, int]]:
    """Count fact codes only when they occur beside their own entity."""
    bound = {f.entity: 0 for f in facts}
    misbound = {f.entity: 0 for f in facts}
    for sentence in re.split(r"[.\n]", text):
        low = sentence.lower()
        for fact in facts:
            if fact.code not in sentence:
                continue
            if fact.entity.lower() in low:
                bound[fact.entity] += 1
            elif any(other.entity.lower() in low for other in facts if other.entity != fact.entity):
                misbound[fact.entity] += 1
    return bound, misbound


def copy_fraction(
    dream_tokens: Sequence[str],
    transcript_tokens: Sequence[str],
    n: int = 12,
    cue_flags: Sequence[bool] | None = None,
) -> float:
    """Return the fraction in transcript runs of at least ``n`` tokens."""
    if cue_flags is not None:
        keep = [
            i for i, token in enumerate(dream_tokens) if not (i < len(cue_flags) and cue_flags[i])
        ]
        dream_tokens = [dream_tokens[i] for i in keep]
    if not dream_tokens or not transcript_tokens or n <= 0:
        return 0.0
    grams: set[tuple[str, ...]] = {
        tuple(transcript_tokens[i : i + n]) for i in range(len(transcript_tokens) - n + 1)
    }
    copied = [False] * len(dream_tokens)
    for i in range(len(dream_tokens) - n + 1):
        if tuple(dream_tokens[i : i + n]) in grams:
            for j in range(i, i + n):
                copied[j] = True
    return sum(copied) / len(copied)


def aggregate_binding(dreams: Sequence[CachedDream], facts: Sequence[Fact]) -> dict[str, int]:
    """Count how many dreams bind each fact at least once."""
    counts = {f.entity: 0 for f in facts}
    for dream in dreams:
        bound, _ = binding_coverage("".join(dream.token_texts), facts)
        for entity, n in bound.items():
            counts[entity] += n > 0
    return counts


def assert_aggregate_binding(
    dreams: Sequence[CachedDream], facts: Sequence[Fact], min_dreams: int
) -> dict[str, int]:
    """Refuse a set where a fact is bound in too few dreams."""
    counts = aggregate_binding(dreams, facts)
    short = {e: n for e, n in counts.items() if n < min_dreams}
    if short:
        raise SystemExit(
            f"aggregate binding gate FAILED: {short} bound in fewer than {min_dreams} of "
            f"{len(dreams)} dreams. A fact the set never binds is one no arm can install -- "
            "raise --dreams, or revisit the steer prefix, before running any cell."
        )
    return counts


def dream_bases(
    queries: Sequence[Sequence[torch.Tensor]],
    gate: Sequence[int],
    wake_state,
    rank_rule: str,
    weights: Sequence[float] | None = None,
):
    """Build the shared per-layer bases stored in a dream cache."""
    import torch

    n_layers = len(wake_state.ssm_states)
    spectra: list[list[float]] = []
    ranks: dict[str, list[int]] = {rule: [] for rule in RANK_RULES}
    bases: dict[str, list[torch.Tensor]] = {variant: [] for variant in VARIANTS}
    for layer in range(n_layers):
        if not gate:
            d_state = wake_state.ssm_states[layer].shape[-1]
            spectra.append([])
            for rule in RANK_RULES:
                ranks[rule].append(0)
            for variant in VARIANTS:
                bases[variant].append(torch.zeros(0, d_state))
            continue
        v_full, sigma = aggregate_basis([queries[t][layer] for t in gate], weights)
        budget = address_budget(v_full.shape[1])
        chosen = {"ratio-gap": rank_ratio_gap(sigma, budget), "median": rank_median(sigma, budget)}
        for rule, rank in chosen.items():
            ranks[rule].append(rank)
        spectra.append([float(x) for x in sigma])
        for variant in VARIANTS:
            basis = variant_basis(v_full, chosen[rank_rule], variant, wake_state.ssm_states[layer])
            if basis.shape[0] == 0:
                print(
                    f"[{ts()}]  NOTE: layer {layer}'s {variant} basis is empty at rank "
                    f"{chosen[rank_rule]} over {len(gate)} gated queries -- this dream's "
                    f"{variant} eraser removes nothing at this layer."
                )
            identity = basis @ basis.T
            if basis.shape[0] and not bool(
                (identity - torch.eye(basis.shape[0])).abs().max() < 1e-4
            ):
                raise SystemExit(
                    f"layer {layer}'s {variant} basis is not orthonormal (sec 2.7 asserts V^T V = I)"
                )
            bases[variant].append(basis.cpu())
    return spectra, ranks, bases


def rebase_dream_set(cache: DreamSetCache, family: str, rank_rule: str) -> DreamSetCache:
    """Recompute cached erasers for a different gating family."""
    from experiments.erasure.pilot import scheme_weights

    for dream in cache.dreams:
        gate = dream.gate_positions
        if not gate:
            continue
        weights = (
            None
            if family == "hard"
            else scheme_weights([dream.divergence[t] for t in gate], family)
        )
        spectra, ranks, bases = dream_bases(
            dream.queries, range(len(gate)), cache.wake_state, rank_rule, weights
        )
        dream.spectra, dream.ranks, dream.bases = spectra, ranks, bases
    cache.gate_family = family
    cache.rank_rule = rank_rule
    return cache


def merge_dream_sets(caches: Sequence[DreamSetCache]) -> DreamSetCache:
    """Merge compatible cache fragments and recompute the set identity."""
    if not caches:
        raise SystemExit("merge_dream_sets: nothing to merge")
    first = caches[0]
    seen: dict[str, int] = {}
    dreams: list[CachedDream] = []
    for i, cache in enumerate(caches):
        if cache.transcript_ids != first.transcript_ids:
            raise SystemExit(
                f"merge_dream_sets: cache {i} has a different wake transcript "
                f"({token_sha(cache.transcript_ids)[:12]} vs {first.transcript_sha[:12]}) -- "
                "its dreams came from a different state and cannot pool."
            )
        if cache.generator != first.generator:
            raise SystemExit(
                f"merge_dream_sets: cache {i} has generator {cache.generator[:12]}, "
                f"first has {first.generator[:12]} -- a different teacher wrote those dreams."
            )
        for dream in cache.dreams:
            if len(dream.dream_ids) > MIN_DISTINCT_DREAM_TOKENS:
                if dream.dream_sha in seen:
                    raise SystemExit(
                        f"merge_dream_sets: cache {i} repeats a {len(dream.dream_ids)}-token "
                        f"dream already in cache {seen[dream.dream_sha]} "
                        f"(sha {dream.dream_sha[:12]}) -- two builds shared a "
                        "--dream-seed-offset."
                    )
                seen[dream.dream_sha] = i
            dreams.append(dream)
    merged = copy.copy(first)
    merged.dreams = dreams
    merged.set_sha = dream_set_sha(dreams)
    return merged


def save_dream_cache(cache: DreamCache | DreamSetCache, path: str | Path) -> None:
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def load_dream_cache(path: str | Path) -> DreamCache | DreamSetCache:
    """Load and verify either cache shape, including legacy cache fields."""
    import torch

    cache = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(cache, DreamSetCache):
        for i, dream in enumerate(cache.dreams):
            if token_sha(dream.dream_ids) != dream.dream_sha:
                raise SystemExit(
                    f"dream set {path} is corrupt: dream {i}'s tokens do not match its sha-256"
                )
        if (
            token_sha(cache.transcript_ids) != cache.transcript_sha
            or dream_set_sha(cache.dreams) != cache.set_sha
        ):
            raise SystemExit(
                f"dream set {path} is corrupt: the set no longer matches its recorded sha-256"
            )
        return cache
    cache.generator = getattr(cache, "generator", "base")
    cache.stop_reason = getattr(cache, "stop_reason", "max-tokens")
    cache.dream_prompt = getattr(cache, "dream_prompt", "")
    cache.prefix_len = getattr(cache, "prefix_len", 0)
    for name, ids, recorded in (
        ("transcript", cache.transcript_ids, cache.transcript_sha),
        ("dream", cache.dream_ids, cache.dream_sha),
    ):
        if token_sha(ids) != recorded:
            raise SystemExit(
                f"dream cache {path} is corrupt: {name} tokens do not match their recorded sha-256"
            )
    return cache


def dream_sidecar_text(cache: DreamCache) -> str:
    """Render a dream with cue spans bracketed."""
    parts: list[str] = []
    in_cue = False
    for text, cue in zip(cache.token_texts, cache.cue_flags, strict=True):
        if cue and not in_cue:
            parts.append("[CUE]")
        elif in_cue and not cue:
            parts.append("[/CUE]")
        in_cue = cue
        parts.append(text)
    if in_cue:
        parts.append("[/CUE]")
    return "".join(parts)


def sidecar_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.stem.replace("dream_cache", "dream", 1) + ".txt")


def pilot_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".pilot.pt")


def write_dream_sidecar(cache: DreamCache, path: str | Path) -> None:
    facts = cache.fact_list
    bound, misbound = binding_coverage("".join(cache.token_texts), facts)
    header = [
        f"seed {cache.seed}   dream_sha {cache.dream_sha}   transcript_sha {cache.transcript_sha}",
        f"generated by: {cache.generator}   ended: {cache.stop_reason}",
        f"steer prefix: {cache.dream_prompt!r} ({cache.prefix_len} tokens, excluded from every "
        "arm's scored positions)",
        f"{len(cache.dream_ids)} dream tokens, {cache.free_tokens} freely generated, "
        f"{len(cache.dream_ids) - cache.free_tokens} spliced cue text",
        "facts: "
        + ", ".join(f"{f.entity}={f.code} (foil {cache.distractors[f.entity]})" for f in facts),
        "bound rehearsals: " + ", ".join(f"{e}={n}" for e, n in bound.items()),
        "misbound rehearsals: " + ", ".join(f"{e}={n}" for e, n in misbound.items()),
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(header) + dream_sidecar_text(cache) + "\n")


def write_dream_set_sidecar(cache: DreamSetCache, path: str | Path) -> None:
    facts = cache.fact_list
    lines = [
        f"seed {cache.seed}   set_sha {cache.set_sha}   transcript_sha {cache.transcript_sha}",
        f"generated by: {cache.generator}   {len(cache.dreams)} dreams",
        f"steer prefix: {cache.dream_prompt!r}   gate threshold {cache.gate_threshold} nats "
        f"   rank rule {cache.rank_rule}",
        "facts: "
        + ", ".join(f"{f.entity}={f.code} (foil {cache.distractors[f.entity]})" for f in facts),
        "dreams bound per fact: "
        + ", ".join(f"{e}={n}" for e, n in aggregate_binding(cache.dreams, facts).items()),
        "",
    ]
    for i, dream in enumerate(cache.dreams):
        bound, misbound = binding_coverage("".join(dream.token_texts), facts)
        agreement = gate_agreement(
            dream.gate_positions, fact_read_positions(dream.token_texts, facts)
        )
        lines += [
            f"--- dream {i}  sha {dream.dream_sha[:12]}  {len(dream.dream_ids)} tokens  "
            f"ended {dream.stop_reason} ---",
            f"within-dream repeats: {bound}   misbound: {misbound}   "
            f"verbatim-copied: {copy_fraction(dream.dream_ids[dream.prefix_len :], cache.transcript_ids, cue_flags=dream.cue_flags[dream.prefix_len :]):.1%}",
            f"gate: {len(dream.gate_positions)} positions, precision {agreement['precision']:.3f} "
            f"recall {agreement['recall']:.3f}, per-fact contribution {agreement['per_fact']}",
            "per-layer rank: " + "  ".join(f"{rule}={dream.ranks[rule]}" for rule in RANK_RULES),
            "per-layer spectra: "
            + "; ".join(" ".join(f"{s:.3g}" for s in spectrum) for spectrum in dream.spectra),
            "".join(dream.token_texts),
            "",
        ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def default_cache_path(seed: int, dreams: int) -> Path:
    dream_set = Path(f"data/dream_set_s{seed}.pt")
    return dream_set if dreams or dream_set.exists() else Path(f"data/dream_cache_s{seed}.pt")


__all__ = [
    "aggregate_binding",
    "assert_aggregate_binding",
    "binding_coverage",
    "copy_fraction",
    "default_cache_path",
    "dream_bases",
    "dream_sidecar_text",
    "fact_read_positions",
    "gate_agreement",
    "load_dream_cache",
    "merge_dream_sets",
    "pilot_path",
    "rebase_dream_set",
    "save_dream_cache",
    "sidecar_path",
    "write_dream_set_sidecar",
    "write_dream_sidecar",
]
