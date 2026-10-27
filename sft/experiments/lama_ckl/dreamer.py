"""Experiment: lama-ckl

Dreamer trainer, arm (a): ReST-EM selection among the model's own dreams
(DISCUSSION-20261024 sec 3.2 mechanism spec block, sec 6.6).

One round, the sec 3.2 event sequence:
1. instrument (round 0: the box gate's `gate.json`; later rounds: the gate
   module's instrument steps on the starting checkpoint), the P draw, R and
   the per-source g_old, the P rows' g_new (all before any pool);
2. per row: wake, pool of K*n dreams, per-dream gradients, scalars, B* by the
   gate's `assemble`, the random-batch specificity reading, `rows/<row>.json`;
3. round 0: the live reading over the first ten fact rows;
4. the SFT (wake transcript -> kept dream pairs, 1:1 replay);
5. the round readings into `readings.json` (written last: it marks the round
   complete).

The score and the assembly live in the gate module, so the dreamer and the
gate agree by construction. Arm (b) (sec 4.52) is not built: it would replace
step 4's `sft` with a step through the teacher logits on the same scalars.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import shutil
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F

from experiments.dream_distillation import flat_grad, turn_weights
from experiments.dream_generation import generate_replay_dreams
from experiments.dream_types import CachedDream
from experiments.lama_ckl.diagnostics import alignment_gate as gate
from experiments.lama_ckl.diagnostics.dream_census import (
    _counts,
    _text_counts,
    document_trigrams,
    revisits,
)
from experiments.lama_ckl.dreamer_corpus import pair_negatives
from experiments.lama_ckl.protocol import WAKE_FRAMES, dream_prompt_ids, run_conversational_wake
from experiments.lama_ckl.runner import (
    DREAM_TOKENS,
    EVIDENCE_TOKENS,
    REPLY_TOKENS,
    _save_trainable,
)
from experiments.lama_ckl.split import DEFAULT_WARMSTART
from progress import fmt_duration, heartbeat, ts

POOL_K = 4
DREAMS_N = 10
ROWS = 50
FACT_ROWS = 35 / 50
SPECIFICITY_BATCHES = 20
LIVE_FACT_ROWS = 10
R_PER_SOURCE = 25
SFT_LR = 1e-4
SFT_WEIGHT_DECAY = 0.01
SFT_EPOCHS = 2
SFT_BATCH = 8
DECODED = 5
SAMPLE_TEMPERATURE = gate.SAMPLE_TEMPERATURE

Seq = gate.Seq
Record = dict[str, object]


# ---- arithmetic (no model) ----


def sft_weights(rows: Sequence[Mapping[str, object]]) -> list[Record]:
    """Sec 3.2 row 6 SFT weight per row (`kind`, B*'s `score`, random-batch `gap`):
    zero unless score > 0 and gap > 0, else the gap over the mean gap of the kind's
    positive rows. `zero` lists the causes: no_batch, score, gap."""
    out: list[Record] = []
    for row in rows:
        score, gap = row["score"], row["gap"]
        zero = []
        if score is None:
            zero.append("no_batch")
        elif score <= 0:
            zero.append("score")
        if gap is None or gap <= 0:
            zero.append("gap")
        out.append({"weight": 0.0 if zero else float(gap), "zero": zero})
    for kind in {str(row["kind"]) for row in rows}:
        positive = [o for o, r in zip(out, rows, strict=True) if r["kind"] == kind and o["weight"]]
        if positive:
            mean = statistics.fmean(o["weight"] for o in positive)
            for o in positive:
                o["weight"] /= mean
    return out


def weight_readings(rows: Sequence[Mapping[str, object]], weights: Sequence[Record]) -> Record:
    """Per kind: rows, the share at zero weight and its causes, the gap distribution."""
    out: Record = {}
    for kind in sorted({str(row["kind"]) for row in rows}):
        mine = [(r, w) for r, w in zip(rows, weights, strict=True) if r["kind"] == kind]
        gaps = sorted(float(r["gap"]) for r, _ in mine if r["gap"] is not None)
        zero = [w for _, w in mine if not w["weight"]]
        out[kind] = {
            "rows": len(mine),
            "zero_share": len(zero) / len(mine),
            "zero_by_cause": {
                cause: sum(cause in w["zero"] for w in zero) / len(mine)
                for cause in ("no_batch", "score", "gap")
            },
            "gap_quantiles": (
                [gaps[round(q * (len(gaps) - 1))] for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
                if gaps
                else []
            ),
            "gap_positive": sum(g > 0 for g in gaps),
        }
    return out


def specificity_others(i: int, kinds: Sequence[str], partners: Sequence[int]) -> list[int]:
    """Rows of row i's kind for its gap: never i, the row that supplied its C-, or a
    row whose C- came from i."""
    return [
        j
        for j, kind in enumerate(kinds)
        if kind == kinds[i] and j != i and j != partners[i] and partners[j] != i
    ]


def batch_gap(
    batch: Sequence[int],
    *,
    gram: torch.Tensor,
    dots: torch.Tensor,
    own: int,
    others: Sequence[int],
    alpha: float,
    c: float,
) -> float | None:
    """alpha <G'_B, g_new^own - mean_j g_new^j> from the Gram matrix and the dots
    (pool, P) of each dream's gradient with every row's g_new; None with no other row."""
    if not others:
        return None
    index = torch.tensor(list(batch))
    n = len(batch)
    norm2 = float(gram[index][:, index].sum()) / n**2
    clip = min(1.0, c / norm2**0.5) if norm2 > 0 else 1.0
    g = dots[index].sum(0) / n
    return alpha * clip * (float(g[own]) - float(g[list(others)].mean()))


def hinge_by_source(batch: Sequence[int], *, old: torch.Tensor, alpha: float, clip: float):
    """The score's hinge per reference source, for the batch's clip factor."""
    return [max(0.0, -alpha * clip * float(v) / len(batch)) for v in old[list(batch)].sum(0)]


def pair_seq(transcript: Sequence[int], dream: Mapping[str, object], eoc_id: int) -> Seq:
    """SFT pair: input the wake transcript through the dream's prompt (the EOC),
    targets the dream's generated tokens with an EOC appended when it stopped on one
    (the generator does not append the stop token)."""
    ids = list(dream["ids"])
    body = len(transcript) + int(dream["prefix_len"])
    tokens = list(transcript) + ids + ([eoc_id] if dream["stop_reason"] == "eoc" else [])
    return tokens, [i >= body for i in range(len(tokens))]


def sft_pairs(
    records: Sequence[Mapping[str, object]], weights: Sequence[Record], eoc_id: int
) -> list[tuple[Seq, float]]:
    """The kept batches of positive-weight rows as (pair, weight); zero rows drop out."""
    pairs = []
    for record, weight in zip(records, weights, strict=True):
        if weight["weight"] <= 0:
            continue
        for i in record["kept"]:
            pairs.append(
                (pair_seq(record["transcript"], record["pool"][i], eoc_id), weight["weight"])
            )
    return pairs


def replay_draw(pool: Sequence[Record], count: int, rng: random.Random) -> tuple[list[Record], int]:
    """Up to `count` replay rows without replacement; returns (rows, shortfall)."""
    taken = rng.sample(list(pool), min(count, len(pool)))
    return taken, count - len(taken)


def kept_dream_draw(
    records: Sequence[Mapping[str, object]],
    weights: Sequence[Record],
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """(record index, pool index) of `count` kept dreams of positive-weight rows."""
    kept = [
        (r, i)
        for r, (record, weight) in enumerate(zip(records, weights, strict=True))
        if weight["weight"] > 0
        for i in record["kept"]
    ]
    return rng.sample(kept, min(count, len(kept)))


def within_row_ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based, ties shared) over len + 1, so rows of any size pool."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    at = 0
    while at < len(order):
        end = at
        while end + 1 < len(order) and values[order[end + 1]] == values[order[at]]:
            end += 1
        for k in range(at, end + 1):
            ranks[order[k]] = (at + end) / 2 + 1
        at = end + 1
    return [r / (len(values) + 1) for r in ranks]


def mann_whitney_greater(x: Sequence[float], y: Sequence[float]) -> tuple[float, float] | None:
    """One-sided Mann-Whitney U (x above y): (U of x, p) by the tie-corrected normal
    approximation with continuity correction; None when a group is empty."""
    nx, ny = len(x), len(y)
    if not nx or not ny:
        return None
    ranks = within_row_ranks([*x, *y])
    total = nx + ny
    u = sum(r * (total + 1) for r in ranks[:nx]) - nx * (nx + 1) / 2
    counts: dict[float, int] = {}
    for value in [*x, *y]:
        counts[value] = counts.get(value, 0) + 1
    ties = sum(t**3 - t for t in counts.values())
    variance = nx * ny / 12 * ((total + 1) - ties / (total * (total - 1)))
    if variance <= 0:
        return u, 1.0
    z = (u - nx * ny / 2 - 0.5) / math.sqrt(variance)
    return u, 0.5 * math.erfc(z / math.sqrt(2))


def live_reading(records: Sequence[Mapping[str, object]]) -> Record:
    """Sec 3.2 step 3 over the given fact rows: within-row ranks of s_i and cos_i,
    pooled, revisiting pool dreams against the rest."""
    out: Record = {"rows": len(records)}
    for key in ("s", "cos"):
        hits, misses = [], []
        for record in records:
            ranks = within_row_ranks(record[key])
            for rank, i in zip(ranks, record["candidates"], strict=True):
                (hits if record["pool"][i]["revisit"] else misses).append(rank)
        test = mann_whitney_greater(hits, misses)
        out[key] = {
            "revisiting": len(hits),
            "other": len(misses),
            "U": None if test is None else test[0],
            "p": None if test is None else test[1],
        }
    return out


# ---- files and resume ----


def write_json(path: Path, value: object) -> None:
    """Written to a sibling and renamed, so a crash never leaves a half file that
    the resume would read as complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=1))
    os.replace(tmp, path)


def round_dir(output: Path, r: int) -> Path:
    return output / f"round-{r:02d}"


def round_complete(output: Path, r: int) -> bool:
    return (round_dir(output, r) / "readings.json").exists()


def completed_rows(directory: Path) -> dict[str, Record]:
    return {
        path.stem: json.loads(path.read_text())
        for path in sorted((directory / "rows").glob("*.json"))
    }


def checkpoint_dir(model_name: str, arm: str, r: int) -> Path:
    return Path("../models") / model_name / "checkpoints" / f"dreamer-{arm}-r{r:02d}" / "final"


# ---- data ----


def _qa(question: str, answer: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


def draw_plan(
    corpus: Sequence[Mapping[str, object]],
    retained: Sequence[Mapping[str, object]],
    *,
    rows: int,
    excluded_passages: set[str],
    excluded_conversations: set[int],
    rng: random.Random,
) -> list[Record]:
    """P: fact rows from the retained pool's unknown side (one question per passage),
    conversation rows with a drawn boundary, each with its C- partner (sec 3.3)."""
    facts_needed = round(rows * FACT_ROWS)
    passages = {str(r["passage_id"]): r for r in corpus if r["kind"] == "fact"}
    unknown = [
        r
        for r in retained
        if r["side"] == "unknown"
        and str(r["passage_id"]) in passages
        and str(r["passage_id"]) not in excluded_passages
    ]
    rng.shuffle(unknown)
    facts, seen = [], set()
    for r in unknown:
        if r["passage_id"] not in seen and len(facts) < facts_needed:
            facts.append(r)
            seen.add(r["passage_id"])
    conversations = [
        r
        for r in corpus
        if r["kind"] == "conversation" and int(r["ultrachat_index"]) not in excluded_conversations
    ]
    if len(facts) < facts_needed or len(conversations) < rows - facts_needed:
        raise SystemExit(
            f"P draw short: {len(facts)}/{facts_needed} fact rows, "
            f"{len(conversations)} conversation candidates"
        )
    conversations = rng.sample(conversations, rows - facts_needed)
    boundaries = [int(rng.choice(r["boundaries"])["at"]) for r in conversations]
    fact_partners = pair_negatives([str(r["answer"]) for r in facts], rng)
    targets = [
        r["messages"][at + 1]["content"] for r, at in zip(conversations, boundaries, strict=True)
    ]
    conv_partners = pair_negatives(targets, rng)
    plan: list[Record] = []
    for i, (r, j) in enumerate(zip(facts, fact_partners, strict=True)):
        passage = str(passages[str(r["passage_id"])]["passage"])
        plan.append(
            {
                "row_id": f"{i:02d}-fact-{r['passage_id']}",
                "kind": "fact",
                "passage_id": r["passage_id"],
                "squad_id": r["squad_id"],
                "passage": passage,
                "document": passage,
                "c": _qa(str(r["question"]), str(r["answer"])),
                "c_negative": _qa(str(r["question"]), str(facts[j]["answer"])),
                "partner": j,
            }
        )
    for k, (r, at, j) in enumerate(zip(conversations, boundaries, conv_partners, strict=True)):
        messages = r["messages"]
        plan.append(
            {
                "row_id": f"{facts_needed + k:02d}-conversation-{r['ultrachat_index']}",
                "kind": "conversation",
                "ultrachat_index": r["ultrachat_index"],
                "at": at,
                "prefix": messages[:at],
                "document": "\n".join(m["content"] for m in messages[:at]),
                "c": messages[at : at + 2],
                "c_negative": [messages[at], {"role": "assistant", "content": targets[j]}],
                "partner": facts_needed + j,
            }
        )
    return plan


def prod_plan(fixed: Mapping[str, object]) -> list[Record]:
    return [
        {
            "row_id": f"prod-{i:02d}-{row['passage_id']}",
            "kind": "fact",
            "passage": row["passage"],
            "document": row["passage"],
            "c": row["question"]["messages"],
            "c_negative": row["negative"]["messages"],
        }
        for i, row in enumerate(fixed["prod"])
    ]


def replay_pool(corpus: Sequence[Mapping[str, object]], used: Sequence[Mapping[str, object]]):
    """Dolly plain and trained-on UltraChat rows not in this round's R."""
    taken = {(k, row[k]) for row in used for k in ("dolly_line", "ultrachat_index") if k in row}
    pool = []
    for row in corpus:
        if row["kind"] == "dolly_plain":
            key = ("dolly_line", row["dolly_line"])
        elif row["kind"] == "warm_start" and not row["packer_dropped"]:
            key = ("ultrachat_index", row["ultrachat_index"])
        else:
            continue
        if key not in taken:
            pool.append({key[0]: key[1], "messages": row["messages"]})
    return pool


# ---- model ----


@dataclass
class Round:
    """What a row needs of its round: the model, the instrument, and the round's vectors."""

    model: torch.nn.Module
    tokenizer: object
    mod: ModuleType
    device: torch.device
    args: argparse.Namespace
    r: int
    alpha: float
    c: float
    kappa: float
    constants_r0: tuple[float, float, float]
    g_new: torch.Tensor | None = None
    g_old: dict[str, torch.Tensor] | None = None
    g_old_rows: torch.Tensor | None = None
    kinds: list[str] | None = None
    partners: list[int] | None = None
    documents: dict[str, set] | None = None

    @property
    def ids(self) -> dict[str, int]:
        return gate.markers(self.tokenizer, self.mod)


def wake(ctx: Round, row: Mapping[str, object], seed: int) -> tuple[list[int], object, Record]:
    """Sec 3.2 row 1 from empty: a fact row through the protocol's wake with a drawn
    frame; a conversation row's prefix teacher-forced, no frame, no reply."""
    from preparation.conversations import format_conversation

    mod = ctx.mod
    if row["kind"] == "fact":
        frame = random.Random(f"dreamer-frame-{seed}").choice(WAKE_FRAMES)
        out, state = run_conversational_wake(
            ctx.model,
            ctx.tokenizer,
            [str(row["passage"])],
            None,
            mod.USER_OPEN,
            mod.ASST_OPEN,
            mod.EOC,
            reply_tokens=REPLY_TOKENS,
            evidence_tokens=EVIDENCE_TOKENS,
            device=ctx.device,
            temperature=SAMPLE_TEMPERATURE,
            seed=seed,
            frame=frame,
        )
        return (
            out["transcript_token_ids"],
            state,
            {"frame": frame, "reply": out["turns"][0]["assistant"]},
        )
    ids = format_conversation(
        list(row["prefix"]),
        ctx.tokenizer,
        10**9,
        mod.USER_OPEN,
        mod.ASST_OPEN,
        None,
        train_user_turns=True,
    )[0]
    with torch.no_grad():
        _, state = ctx.model(torch.tensor([ids], device=ctx.device), state=None)
    return ids, state.detach(), {"frame": None, "reply": None}


def dream_pool(ctx: Round, state, seed: int) -> list[CachedDream]:
    """Sec 3.2 row 2: K*n dreams from copies of S_W in `--batch-size` launches."""
    tokenizer = ctx.tokenizer
    ids = ctx.ids
    dreams, _ = generate_replay_dreams(
        ctx.model,
        state,
        dream_prompt_ids(tokenizer, ctx.mod.USER_OPEN, ctx.mod.ASST_OPEN, ctx.mod.EOC, ctx.device),
        count=ctx.args.pool_k * ctx.args.dreams_n,
        batch_size=ctx.args.batch_size,
        seed=seed,
        n_tokens=ctx.args.tokens,
        temperature=SAMPLE_TEMPERATURE,
        decode_token=lambda token: tokenizer.decode([token]),
        stop_id=ids["eoc_id"],
        turn_id=ids["eos_id"],
    )
    return dreams


def dream_record(dream: CachedDream) -> Record:
    return {
        "ids": dream.dream_ids,
        "prefix_len": dream.prefix_len,
        "stop_reason": dream.stop_reason,
        "text": "".join(dream.token_texts),
        "length": len(dream.dream_ids) - dream.prefix_len,
    }


def as_cached(dream: Mapping[str, object]) -> CachedDream:
    ids = list(dream["ids"])
    return CachedDream(
        dream_ids=ids,
        token_texts=[],
        teacher_logits=torch.empty(0),
        cue_flags=[False] * len(ids),
        prefix_len=int(dream["prefix_len"]),
        stop_reason=str(dream["stop_reason"]),
        divergence=[],
        gate_positions=[],
        queries=[],
        spectra=[],
        ranks={},
        bases={},
    )


def kept_dream_gradient(ctx: Round, row: Mapping[str, object]) -> torch.Tensor:
    """Sec 3.2 row 4's fourth source: the dream teacher-forced after its recorded wake
    transcript under the current theta, no gradient through the prefix, cross-entropy
    on its scored positions."""
    model, device = ctx.model, ctx.device
    ids = list(row["dream"]["ids"])
    positions = sorted(turn_weights(as_cached(row["dream"]), **ctx.ids, loss_weights="uniform"))
    with torch.no_grad():
        _, state = model(torch.tensor([row["transcript"]], device=device), state=None)
    logits, _ = model(torch.tensor([ids], device=device), state=state.detach())
    targets = torch.tensor([ids[p + 1] for p in positions], device=device)
    F.cross_entropy(logits[0, positions].float(), targets).backward()
    grad = flat_grad(model).float()
    model.zero_grad(set_to_none=True)
    return grad


def reference_rows(ctx: Round, reference: Mapping[str, Sequence[Mapping[str, object]]]):
    """Per-row g_old (rows, d) in source order; g_old^k is the mean of its source's rows
    (the gradient of the mean of per-row means)."""
    grads, spans, started = [], {}, time.time()
    total = sum(map(len, reference.values()))
    for name, rows in reference.items():
        start = len(grads)
        for row in rows:
            if name == "dreams":
                grads.append(kept_dream_gradient(ctx, row))
            else:
                grads.append(gate.mean_loss_grad(ctx.model, [(row["ids"], row["mask"])], 1))
            done = len(grads)
            if done % 10 == 0 or done == total:
                heartbeat()
                print(
                    f"[{ts()}] g_old rows {done}/{total} ({name}), "
                    f"ETA {fmt_duration((time.time() - started) / done * (total - done))}",
                    flush=True,
                )
        spans[name] = (start, len(grads))
    stacked = torch.stack(grads)
    return stacked, {name: stacked[a:b].mean(0) for name, (a, b) in spans.items()}


def stack_states(states: Sequence[object]) -> object:
    stacked = copy.copy(states[0])
    for attr in ("conv_states", "ssm_states"):
        layers = zip(*(getattr(state, attr) for state in states), strict=True)
        setattr(stacked, attr, [torch.cat(tensors) for tensors in layers])
    return stacked


def greedy_replies(model, prompts: Sequence[Sequence[int]], stop_ids: set[int], max_tokens: int):
    """Greedy replies for all prompts in one decode batch: each prompt from empty
    state at batch 1, the states stacked, then one token per step for every row."""
    device = next(model.parameters()).device
    states, last = [], []
    with torch.no_grad():
        for prompt in prompts:
            logits, state = model(torch.tensor([list(prompt)], device=device), state=None)
            states.append(state)
            last.append(logits[:, -1])
        state, logits = stack_states(states), torch.cat(last)
        replies: list[list[int]] = [[] for _ in prompts]
        done = [False] * len(prompts)
        for _ in range(max_tokens):
            tokens = logits.float().argmax(-1)
            for row, token in enumerate(tokens.tolist()):
                if not done[row]:
                    done[row] = token in stop_ids
                    if not done[row]:
                        replies[row].append(token)
            if all(done):
                break
            out, state = model(tokens[:, None], state=state)
            logits = out[:, -1]
    return replies


def sft_batch_loss(model, items: Sequence[tuple[Seq, float]], total_weight: float) -> torch.Tensor:
    """Per-pair token-mean cross-entropy, summed with each pair's weight over the
    batch's total weight (the SFT-weighted mean when the batch is one call)."""
    losses = gate.row_losses(model, [seq for seq, _ in items])
    weights = torch.tensor([w for _, w in items], device=losses.device)
    return (losses * weights).sum() / total_weight


def sft(model, opt, items: Sequence[tuple[Seq, float]], *, batch: int, seed: str) -> Record:
    """Sec 3.2 row 6: 2 epochs, batch `batch`, one unchunked backward per pair; an OOM
    halves the micro-batch and accumulates to the same batch."""
    rng = random.Random(seed)
    micro, step, losses = batch, 0, []
    steps = SFT_EPOCHS * math.ceil(len(items) / batch)
    started = time.time()
    for epoch in range(SFT_EPOCHS):
        order = list(range(len(items)))
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            chunk = [items[i] for i in order[start : start + batch]]
            total = sum(w for _, w in chunk)
            while True:
                try:
                    value = 0.0
                    for at in range(0, len(chunk), micro):
                        loss = sft_batch_loss(model, chunk[at : at + micro], total)
                        loss.backward()
                        value += float(loss)
                    break
                except torch.cuda.OutOfMemoryError:
                    opt.zero_grad()
                    torch.cuda.empty_cache()
                    if micro == 1:
                        raise
                    micro //= 2
                    print(f"[{ts()}] SFT OOM: micro-batch {micro}, accumulating to {batch}")
            opt.step()
            opt.zero_grad()
            step += 1
            losses.append(value)
            heartbeat()
            if step % 10 == 0 or step == steps:
                elapsed = time.time() - started
                print(
                    f"[{ts()}] SFT epoch {epoch + 1}/{SFT_EPOCHS} step {step}/{steps} loss "
                    f"{statistics.fmean(losses[-10:]):.4f} (last 10), {step / elapsed:.2f} step/s, "
                    f"ETA {fmt_duration(elapsed / step * (steps - step))}",
                    flush=True,
                )
    return {"steps": step, "micro_batch": micro, "losses": losses, "seconds": time.time() - started}


# ---- one row ----


def run_row(ctx: Round, index: int, row: Mapping[str, object], seed: int) -> Record:
    """Sec 3.2 step 2 on P row `index`."""
    n = ctx.args.dreams_n
    started = time.time()
    transcript, state, info = wake(ctx, row, 2 * seed)
    dreams = dream_pool(ctx, state, 2 * seed + 1)
    del state
    pool = [dream_record(d) for d in dreams]
    own_doc = ctx.documents[str(row["row_id"])]
    for record in pool:
        record["revisit"] = revisits(record["text"].removeprefix(ctx.mod.EOC), own_doc)

    candidates, grads = [], []
    for i, dream in enumerate(dreams):
        grad = gate.dream_gradient(ctx.model, dream, ctx.ids)
        pool[i]["scored"] = grad is not None
        if grad is not None:
            candidates.append(i)
            grads.append(grad)
        heartbeat()
    del dreams
    out: Record = {
        "row_id": row["row_id"],
        "kind": row["kind"],
        "seed": seed,
        "frame": info["frame"],
        "wake_reply": info["reply"],
        "transcript": transcript,
        "pool": pool,
        "candidates": candidates,
        "kept": None,
        "score": None,
        "gap": None,
    }
    out["census"] = census(pool, [own_doc])
    if len(candidates) < n:
        print(f"[{ts()}] {row['row_id']}: {len(candidates)} scored dreams < n {n}, no batch")
        return {**out, "seconds": time.time() - started}

    stacked = torch.stack(grads)
    del grads
    scalars = gate.dream_scalars(stacked, ctx.g_new[index], ctx.g_old)
    dots = (stacked @ ctx.g_new.T).double().cpu()
    dots_rows = (stacked @ ctx.g_old_rows.T).double().cpu()
    del stacked
    g_new_norm = float(ctx.g_new[index].norm())
    norms = scalars["norms"]
    out.update(
        {
            "s": [ctx.alpha * float(v) for v in scalars["new"]],
            "cos": [
                float(v) / (float(m) * g_new_norm)
                for v, m in zip(scalars["new"], norms, strict=True)
            ],
            "o": [[ctx.alpha * float(v) for v in row_old] for row_old in scalars["old"]],
            "norms": [float(v) for v in norms],
            "gram": scalars["gram"].tolist(),
            "new": scalars["new"].tolist(),
            "old": scalars["old"].tolist(),
            "sources": list(ctx.g_old),
        }
    )
    instrument = {
        "gram": scalars["gram"],
        "new": scalars["new"],
        "old": scalars["old"],
        "alpha": ctx.alpha,
        "kappa": ctx.kappa,
        "c": ctx.c,
    }
    alpha0, c0, kappa0 = ctx.constants_r0
    r0 = {**instrument, "alpha": alpha0, "c": c0, "kappa": kappa0}
    others = specificity_others(index, ctx.kinds, ctx.partners)
    gap_args = {"gram": scalars["gram"], "dots": dots, "own": index, "others": others}

    local, best = gate.assemble(range(len(candidates)), n, **instrument)
    learn, quadratic, hinge, clip = gate.score_terms(local, **instrument)
    top_s = sorted(range(len(candidates)), key=lambda i: -float(scalars["new"][i]))[:n]
    out.update(
        {
            "kept": [candidates[i] for i in local],
            "score": best,
            "terms": {
                "learn": learn,
                "quadratic": quadratic,
                "hinge": hinge,
                "clip": clip,
                "hinge_by_source": dict(
                    zip(
                        ctx.g_old,
                        hinge_by_source(local, old=scalars["old"], alpha=ctx.alpha, clip=clip),
                        strict=True,
                    )
                ),
            },
            "score_r0": gate.score(local, **r0),
            "gap_kept": batch_gap(local, **gap_args, alpha=ctx.alpha, c=ctx.c),
            "kept_is_top_s": set(local) == set(top_s),
            "old_rows": (ctx.alpha * clip * dots_rows[local].sum(0) / n).tolist(),
        }
    )
    rng = random.Random(f"dreamer-specificity-{seed}")
    randoms = []
    for _ in range(SPECIFICITY_BATCHES):
        batch = sorted(rng.sample(range(len(candidates)), n))
        r_learn, r_quadratic, r_hinge, r_clip = gate.score_terms(batch, **instrument)
        randoms.append(
            {
                "batch": [candidates[i] for i in batch],
                "score": r_learn - r_quadratic - r_hinge,
                "score_r0": gate.score(batch, **r0),
                "learn": r_learn,
                "quadratic": r_quadratic,
                "clip": r_clip,
                "gap": batch_gap(batch, **gap_args, alpha=ctx.alpha, c=ctx.c),
            }
        )
    out["random"] = randoms
    gaps = [b["gap"] for b in randoms if b["gap"] is not None]
    out["gap"] = statistics.fmean(gaps) if gaps else None
    out["kept_census"] = census([pool[i] for i in out["kept"]], [own_doc])
    out["seconds"] = time.time() - started
    return out


def census(dreams: Sequence[Mapping[str, object]], documents: Sequence[set]) -> dict[str, int]:
    """The sec 6.1 census columns over a set of dreams (prompt token stripped)."""
    payload = {"dreams": [{"text": d["text"], "stop_reason": d["stop_reason"]} for d in dreams]}
    texts = [str(d["text"]).removeprefix("<|endofconversation|>") for d in dreams]
    counts = {**_counts(payload), **_text_counts(texts, documents)}
    return {k: v for k, v in counts.items() if isinstance(v, int)}


def print_row(record: Mapping[str, object]) -> None:
    if record["kept"] is None:
        print(
            f"[{ts()}] {record['row_id']} no batch ({len(record['candidates'])} scored)", flush=True
        )
        return
    terms = record["terms"]
    kept_norm = statistics.fmean(
        record["norms"][record["candidates"].index(i)] for i in record["kept"]
    )
    print(
        f"[{ts()}] {record['row_id']} B* score {record['score']:+.3e} (learn {terms['learn']:+.3e} "
        f"quad {terms['quadratic']:.3e} hinge {terms['hinge']:.3e} clip {terms['clip']:.3f}) "
        f"r0 {record['score_r0']:+.3e} | gap {record['gap']} kept-gap {record['gap_kept']} | "
        f"|grad| kept {kept_norm:.3e} pool {statistics.fmean(record['norms']):.3e} | "
        f"revisits {sum(d['revisit'] for d in record['pool'])}/{len(record['pool'])} "
        f"top-s {record['kept_is_top_s']} | {fmt_duration(record['seconds'])}",
        flush=True,
    )
    i = record["kept"][0]
    print(f"[{ts()}]   kept: {record['pool'][i]['text'][:400]!r}", flush=True)


# ---- readings ----


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def round_readings(
    records: Sequence[Mapping[str, object]], weights: Sequence[Record], sources: Sequence[str]
) -> Record:
    """Sec 3.2 step 5's readings that are arithmetic over the row records."""
    out: Record = {"weights": weight_readings(records, weights)}
    kept = [r for r in records if r["kept"] is not None]
    for kind in sorted({str(r["kind"]) for r in records}):
        mine = [r for r in kept if r["kind"] == kind]
        randoms = [b for r in mine for b in r["random"]]
        out[kind] = {
            "kept_score_r0": _mean([float(r["score_r0"]) for r in mine]),
            "random_score_r0": _mean([float(b["score_r0"]) for b in randoms]),
            "kept_score": _mean([float(r["score"]) for r in mine]),
            "random_score": _mean([float(b["score"]) for b in randoms]),
            "quadratic_over_learn": sorted(
                b["quadratic"] / b["learn"] for b in randoms if b["learn"] != 0
            ),
            "gap_sign": {
                "positive": sum(r["gap"] is not None and r["gap"] > 0 for r in mine),
                "nonpositive": sum(r["gap"] is not None and r["gap"] <= 0 for r in mine),
                "none": sum(r["gap"] is None for r in mine),
            },
        }
    out["rows_without_batch"] = [r["row_id"] for r in records if r["kept"] is None]
    out["kept_is_top_s"] = sum(bool(r["kept_is_top_s"]) for r in kept)
    out["hinge_fire"] = {
        name: _mean([float(r["terms"]["hinge_by_source"][name] > 0) for r in kept])
        for name in sources
    }
    out["hinge_fire_any"] = _mean([float(r["terms"]["hinge"] > 0) for r in kept])
    out["clip_fire_kept"] = _mean([float(r["terms"]["clip"] < 1) for r in kept])
    out["clip_fire_random"] = _mean([float(b["clip"] < 1) for r in kept for b in r["random"]])
    kept_norms = [r["norms"][r["candidates"].index(i)] for r in kept for i in r["kept"]]
    out["grad_norm_kept"] = _mean(kept_norms)
    out["grad_norm_pool"] = _mean([v for r in kept for v in r["norms"]])
    lengths = [r["pool"][i]["length"] for r in kept for i in r["kept"]]
    out["kept_length_mean"], out["kept_length_max"] = _mean(lengths), max(lengths, default=None)
    out["pool_length_mean"] = _mean([d["length"] for r in records for d in r["pool"]])
    out["old_rows"] = [
        {"mean": statistics.fmean(column), "raised": sum(v < 0 for v in column)}
        for column in zip(*(r["old_rows"] for r in kept), strict=True)
    ]
    out["census_pool"] = _sum_counts([r["census"] for r in records])
    out["census_kept"] = _sum_counts([r["kept_census"] for r in kept])
    out["live"] = live_reading([r for r in kept if r["kind"] == "fact"][:LIVE_FACT_ROWS])
    return out


def _sum_counts(counts: Sequence[Mapping[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for c in counts:
        for key, value in c.items():
            total[key] = total.get(key, 0) + value
    return total


def decoded_samples(records: Sequence[Mapping[str, object]]) -> Record:
    """Five kept and five rejected scored dreams, one per row, printed and returned."""
    kept, rejected = [], []
    for r in records:
        if r["kept"] is None:
            continue
        if len(kept) < DECODED:
            kept.append({"row_id": r["row_id"], "text": r["pool"][r["kept"][0]]["text"]})
        others = [i for i in r["candidates"] if i not in r["kept"]]
        if others and len(rejected) < DECODED:
            rejected.append({"row_id": r["row_id"], "text": r["pool"][others[0]]["text"]})
    for label, samples in (("kept", kept), ("rejected", rejected)):
        for s in samples:
            print(f"[{ts()}] decoded {label} {s['row_id']}: {s['text'][:800]!r}", flush=True)
    return {"kept": kept, "rejected": rejected}


def held_out_readings(ctx: Round, fixed: Mapping[str, object], stems: Sequence[str]) -> Record:
    """Sec 3.2 row 4b: held-out warm-start loss and the over-abstention rate."""
    from preparation.conversations import format_conversation

    mod, tokenizer = ctx.mod, ctx.tokenizer
    seqs = [gate._render(r["messages"], tokenizer, mod) for r in fixed["held_out_warm_start"]]
    values = []
    for start in range(0, len(seqs), ctx.args.eval_batch_size):
        values += gate.losses(ctx.model, seqs[start : start + ctx.args.eval_batch_size])
        heartbeat()
    asst = ctx.ids["asst_id"]
    prompts = [
        format_conversation(
            [{"role": "user", "content": r["question"]["question"]}],
            tokenizer,
            10**9,
            mod.USER_OPEN,
            mod.ASST_OPEN,
            None,
        )[0]
        + [asst]
        for r in fixed["held_out_facts"]
    ]
    stops = set(ctx.ids.values())
    replies = [
        tokenizer.decode(reply).strip()
        for reply in greedy_replies(ctx.model, prompts, stops, REPLY_TOKENS)
    ]
    abstained = [reply.startswith(tuple(stems)) for reply in replies]
    out = {
        "held_out_loss": statistics.fmean(values),
        "held_out_rows": len(values),
        "over_abstention": sum(abstained) / len(abstained),
        "abstentions": [r for r, a in zip(replies, abstained, strict=True) if a],
        "replies_sample": replies[:5],
    }
    print(
        f"[{ts()}] held-out loss {out['held_out_loss']:.4f} over {len(values)} rows; "
        f"over-abstention {sum(abstained)}/{len(abstained)}; replies {replies[:3]!r}",
        flush=True,
    )
    return out


def prod_reading(
    ctx: Round, prod: Sequence[Mapping[str, object]], g_old_r0: Mapping[str, torch.Tensor]
) -> Record:
    """Sec 3.2 step 5: one random n-batch per prod row from its own pool, scored under
    round 0's alpha, c, kappa and R, with cos(G_B, g_new). Seeds depend on the row
    only, so every checkpoint dreams the same draws."""
    alpha0, c0, kappa0 = ctx.constants_r0
    n, rows = ctx.args.dreams_n, []
    for k, row in enumerate(prod):
        seed = ctx.args.seed * 100_000 + 99_000 + k
        _, state, _ = wake(ctx, row, 2 * seed)
        dreams = dream_pool(ctx, state, 2 * seed + 1)
        del state
        contrast, _ = gate.g_new(
            ctx.model,
            gate.reply_seq(row["c"], ctx.tokenizer, ctx.mod),
            gate.reply_seq(row["c_negative"], ctx.tokenizer, ctx.mod),
        )
        grads, chosen = [], []
        for i in random.Random(f"dreamer-prod-{seed}").sample(range(len(dreams)), len(dreams)):
            grad = gate.dream_gradient(ctx.model, dreams[i], ctx.ids)
            if grad is not None:
                grads.append(grad)
                chosen.append(i)
            if len(grads) == n:
                break
        if len(grads) < n:
            rows.append({"row_id": row["row_id"], "score": None, "cos": None})
            continue
        scalars = gate.dream_scalars(torch.stack(grads), contrast, g_old_r0)
        del grads
        batch = list(range(n))
        value = gate.score(
            batch,
            gram=scalars["gram"],
            new=scalars["new"],
            old=scalars["old"],
            alpha=alpha0,
            kappa=kappa0,
            c=c0,
        )
        g_norm = float(scalars["gram"].sum()) ** 0.5 / n
        cos = float(scalars["new"].sum()) / n / (g_norm * float(contrast.norm()))
        rows.append({"row_id": row["row_id"], "score": value, "cos": cos, "batch": chosen})
        heartbeat()
        print(
            f"[{ts()}] prod {k + 1}/{len(prod)} {row['row_id']}: score {value:+.3e} cos {cos:+.4f}",
            flush=True,
        )
    out: Record = {"rows": rows}
    for key in ("score", "cos"):
        values = [float(r[key]) for r in rows if r[key] is not None]
        out[key] = _mean(values)
        out[f"{key}_se"] = (
            statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None
        )
    print(
        f"[{ts()}] prod reading: score {out['score']} (se {out['score_se']}), "
        f"cos {out['cos']} (se {out['cos_se']}) over {len(rows)} rows",
        flush=True,
    )
    return out


# ---- the round ----


def load(args: argparse.Namespace, adapter: Path):
    model, opt, tokenizer, mod = gate.load_model(
        args.model_name, adapter, args.fp32_forward, torch.device("cuda")
    )
    return model, opt, tokenizer, mod


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def instrument_checkpoint(args, adapter: Path, fixed, last_good_h: float | None) -> tuple:
    """The gate module's instrument steps and retained pool on one checkpoint; returns
    (gate.json body, model, opt, tokenizer, mod)."""
    pool_path = gate.retained_pool_pass(
        adapter, args.corpus, args.model_name, args.eval_batch_size, args.fp32_forward
    )
    model, opt, tokenizer, mod = load(args, adapter)
    gate_rows = fixed["gate"]
    result = gate.instrument(
        model,
        opt,
        tokenizer,
        mod,
        gate_rows,
        gate.load_gate_dreams(gate.GATE_DREAMS, gate_rows),
        tokens=args.tokens,
        model_name=args.model_name,
        init_adapter=adapter,
        fp32=args.fp32_forward,
        seed=args.seed,
        verdict=False,
        last_good_h=last_good_h,
    )
    result.update(
        {
            "date": time.strftime("%Y-%m-%d"),
            "init_adapter": str(adapter),
            "tokens": args.tokens,
            "fp32_forward": args.fp32_forward,
            "retained_pool": str(pool_path),
        }
    )
    print(gate.summary_table(result), flush=True)
    return result, model, opt, tokenizer, mod


def constants(result: Mapping[str, object], label: str) -> tuple[float, float, float]:
    """(alpha, c, kappa) of a gate.json; a missing alpha or kappa, or the h rule's stop,
    stops the session."""
    if result["alpha"] is None or result["kappa"] is None or result["kappa_status"] == "stop":
        raise SystemExit(
            f"{label}: alpha {result['alpha']}, kappa {result['kappa']} "
            f"({result['kappa_status']}); the sec 3.2 h rule stops the session"
        )
    return float(result["alpha"]), float(result["c"]), float(result["kappa"])


def run_round(args, r: int, corpus, fixed, stems) -> None:
    directory = round_dir(args.output, r)
    directory.mkdir(parents=True, exist_ok=True)
    start = args.init_adapter if r == 0 else checkpoint_dir(args.model_name, args.arm, r - 1)
    print(f"[{ts()}] ===== round {r} from {start} =====", flush=True)

    # Step 1: the instrument.
    gate_path = directory / "gate.json"
    model = None
    if r == 0:
        if not gate_path.exists():
            shutil.copyfile(args.gate, gate_path)
    elif not gate_path.exists():
        previous = json.loads((round_dir(args.output, r - 1) / "gate.json").read_text())
        last_h = previous["h"] if previous["kappa_status"] in ("fit", "reused") else None
        result, model, opt, tokenizer, mod = instrument_checkpoint(args, start, fixed, last_h)
        write_json(gate_path, result)
    instrument = json.loads(gate_path.read_text())
    if model is None:
        model, opt, tokenizer, mod = load(args, start)
    alpha, c, kappa = constants(instrument, f"round {r}")
    gate0 = json.loads((round_dir(args.output, 0) / "gate.json").read_text())
    ctx = Round(
        model,
        tokenizer,
        mod,
        torch.device("cuda"),
        args,
        r,
        alpha,
        c,
        kappa,
        constants(gate0, "round 0"),
    )
    print(f"[{ts()}] round {r}: alpha {alpha:.1e}, c {c:.4e}, kappa {kappa:.4e}", flush=True)
    retained = json.loads(Path(str(instrument["retained_pool"])).read_text())["rows"]

    if r == 0 and not (directory / "start.json").exists():
        write_json(directory / "start.json", held_out_readings(ctx, fixed, stems))

    # The P draw, excluding every earlier round's P.
    earlier = [json.loads((round_dir(args.output, k) / "plan.json").read_text()) for k in range(r)]
    plan_path = directory / "plan.json"
    if not plan_path.exists():
        write_json(
            plan_path,
            draw_plan(
                corpus,
                retained,
                rows=args.rows,
                excluded_passages={
                    str(p["passage_id"]) for e in earlier for p in e if p["kind"] == "fact"
                },
                excluded_conversations={
                    int(p["ultrachat_index"])
                    for e in earlier
                    for p in e
                    if p["kind"] == "conversation"
                },
                rng=random.Random(f"dreamer-plan-{args.seed}-{r}"),
            ),
        )
    plan = json.loads(plan_path.read_text())
    prod = prod_plan(fixed)
    p_passages = {str(p["passage_id"]) for e in [*earlier, plan] for p in e if p["kind"] == "fact"}
    trigrams = document_trigrams([str(p["document"]) for p in [*plan, *prod]])
    ctx.documents = {str(p["row_id"]): t for p, t in zip([*plan, *prod], trigrams, strict=True)}
    ctx.kinds = [str(p["kind"]) for p in plan]
    ctx.partners = [int(p["partner"]) for p in plan]
    print(
        f"[{ts()}] P: {ctx.kinds.count('fact')} fact, {ctx.kinds.count('conversation')} "
        f"conversation rows; {len(p_passages)} P passages excluded from R",
        flush=True,
    )

    # R and g_old.
    ref_path = directory / "reference.json"
    if not ref_path.exists():
        write_json(
            ref_path, draw_round_reference(args, r, corpus, retained, p_passages, tokenizer, mod)
        )
    reference = json.loads(ref_path.read_text())
    ctx.g_old_rows, ctx.g_old = reference_rows(ctx, reference)
    if r == 0:
        g_old_r0 = dict(ctx.g_old)
    else:
        reference0 = json.loads((round_dir(args.output, 0) / "reference.json").read_text())
        g_old_r0 = gate.reference_gradients(
            model,
            {
                name: [{"ids": x["ids"], "mask": x["mask"]} for x in rows]
                for name, rows in reference0.items()
            },
            args.eval_batch_size,
        )

    # g_new for every P row, before any pool.
    vectors, started = [], time.time()
    for i, p in enumerate(plan):
        contrast, plain = gate.g_new(
            model,
            gate.reply_seq(p["c"], tokenizer, mod),
            gate.reply_seq(p["c_negative"], tokenizer, mod),
        )
        vectors.append(contrast)
        del plain
        heartbeat()
        if (i + 1) % 10 == 0 or i + 1 == len(plan):
            print(
                f"[{ts()}] g_new {i + 1}/{len(plan)}, ETA "
                f"{fmt_duration((time.time() - started) / (i + 1) * (len(plan) - i - 1))}",
                flush=True,
            )
    ctx.g_new = torch.stack(vectors)
    del vectors

    # Step 2 per row (and step 3 as soon as its rows exist).
    done = completed_rows(directory)
    print(f"[{ts()}] rows: {len(done)} complete, {len(plan) - len(done)} to run", flush=True)
    live_printed = False
    for i, p in enumerate(plan):
        row_id = str(p["row_id"])
        if row_id not in done:
            record = run_row(ctx, i, p, args.seed * 100_000 + r * 1_000 + i)
            write_json(directory / "rows" / f"{row_id}.json", record)
            done[row_id] = record
            print_row(record)
            print(f"[{ts()}] round {r} row {len(done)}/{len(plan)} complete", flush=True)
        facts = [
            done[str(q["row_id"])] for q in plan if q["kind"] == "fact" and str(q["row_id"]) in done
        ]
        if r == 0 and not live_printed and len(facts) >= LIVE_FACT_ROWS:
            live = live_reading([f for f in facts if f["kept"] is not None][:LIVE_FACT_ROWS])
            print(f"[{ts()}] live reading (round 0, first ten fact rows): {live}", flush=True)
            live_printed = True
    records = [done[str(p["row_id"])] for p in plan]
    sources = list(ctx.g_old)
    ctx.g_new = ctx.g_old_rows = None
    free()

    # Step 5's prod reading on the starting checkpoint (theta is unchanged until step 4).
    prod_path = directory / "prod.json"
    if not prod_path.exists():
        write_json(prod_path, prod_reading(ctx, prod, g_old_r0))
    del g_old_r0

    # Step 4: the SFT.
    weights = sft_weights(records)
    for record, weight in zip(records, weights, strict=True):
        print(
            f"[{ts()}] weight {record['row_id']}: {weight['weight']:.3f} zero {weight['zero']}",
            flush=True,
        )
    eoc_id = ctx.ids["eoc_id"]
    pairs = sft_pairs(records, weights, eoc_id)
    replay, shortfall = replay_draw(
        replay_pool(corpus, [x for rows in reference.values() for x in rows]),
        len(pairs),
        random.Random(f"dreamer-replay-{args.seed}-{r}"),
    )
    if shortfall:
        print(
            f"[{ts()}] WARNING: replay pool short by {shortfall} of {len(pairs)} rows; "
            f"drawing the {len(replay)} that exist (sec 4.53)",
            flush=True,
        )
    items = pairs + [(gate._render(x["messages"], tokenizer, mod), 1.0) for x in replay]
    print(
        f"[{ts()}] SFT: {len(pairs)} pairs from {sum(w['weight'] > 0 for w in weights)} rows, "
        f"{len(replay)} replay rows; longest {max((len(s[0]) for s, _ in items), default=0)} tokens",
        flush=True,
    )
    from training.master_weights import MasterAdamW

    adamw = MasterAdamW(opt.params, lr=SFT_LR, weight_decay=SFT_WEIGHT_DECAY)
    trained = (
        sft(model, adamw, items, batch=args.sft_batch, seed=f"dreamer-sft-{args.seed}-{r}")
        if items
        else {"steps": 0}
    )
    config = json.loads((start / "lora_config.json").read_text())
    final = checkpoint_dir(args.model_name, args.arm, r)
    tmp = final.with_name(".final.partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    sha = _save_trainable(
        model, tmp / "trainable.pt", int(config["rank"]), float(config["alpha"]), adamw
    )
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    print(f"[{ts()}] round {r} checkpoint {final} sha256 {sha}", flush=True)

    # Step 5's remaining readings on the round checkpoint.
    readings = {
        **round_readings(records, weights, sources),
        "row_weights": {str(r_["row_id"]): w for r_, w in zip(records, weights, strict=True)},
        "replay": {"drawn": len(replay), "shortfall": shortfall},
        "sft": {k: v for k, v in trained.items() if k != "losses"},
        "sft_losses": trained.get("losses", []),
        "checkpoint": str(final),
        "trainable_sha256": sha,
        "prod": json.loads(prod_path.read_text()),
        "decoded": decoded_samples(records),
        **held_out_readings(ctx, fixed, stems),
    }
    write_json(directory / "readings.json", readings)
    print(
        f"[{ts()}] round {r} readings: {json.dumps({k: v for k, v in readings.items() if k not in ('row_weights', 'sft_losses', 'decoded', 'old_rows', 'prod')}, default=str)[:4000]}",
        flush=True,
    )


def draw_round_reference(args, r, corpus, retained, p_passages, tokenizer, mod) -> Record:
    """Round 0: R as the gate draws it (34/33/33). Later rounds: 25 from each of those
    sources and 25 kept dreams of the previous round's positive-weight rows. Fact rows
    never share a passage with any round's P."""
    known = [x for x in retained if str(x["passage_id"]) not in p_passages]
    drawn = gate.draw_reference(corpus, known, tokenizer, mod, args.seed * 100 + r)
    if r == 0:
        return drawn
    rng = random.Random(f"dreamer-reference-{args.seed}-{r}")
    out = {name: rng.sample(rows, min(R_PER_SOURCE, len(rows))) for name, rows in drawn.items()}
    previous = round_dir(args.output, r - 1)
    records = list(completed_rows(previous).values())
    weights = json.loads((previous / "readings.json").read_text())["row_weights"]
    picks = kept_dream_draw(
        records, [weights[str(x["row_id"])] for x in records], R_PER_SOURCE, rng
    )
    if len(picks) < R_PER_SOURCE:
        print(
            f"[{ts()}] WARNING: kept-dream source has {len(picks)} of {R_PER_SOURCE} rows",
            flush=True,
        )
    out["dreams"] = [
        {
            "row_id": records[k]["row_id"],
            "dream": records[k]["pool"][i] | {"index": i},
            "transcript": records[k]["transcript"],
        }
        for k, i in picks
    ]
    return out


def run_final(args, fixed) -> None:
    """The instrument steps and the prod reading on the last round's checkpoint."""
    directory = args.output / "final"
    if (directory / "gate.json").exists():
        print(f"[{ts()}] final complete: {directory}", flush=True)
        return
    last = checkpoint_dir(args.model_name, args.arm, args.rounds - 1)
    previous = json.loads((round_dir(args.output, args.rounds - 1) / "gate.json").read_text())
    last_h = previous["h"] if previous["kappa_status"] in ("fit", "reused") else None
    result, model, _opt, tokenizer, mod = instrument_checkpoint(args, last, fixed, last_h)
    gate0 = json.loads((round_dir(args.output, 0) / "gate.json").read_text())
    # The prod reading scores under round 0's constants; this checkpoint's own are in gate.json.
    nan = float("nan")
    ctx = Round(
        model,
        tokenizer,
        mod,
        torch.device("cuda"),
        args,
        args.rounds,
        nan,
        nan,
        nan,
        constants(gate0, "round 0"),
    )
    reference0 = json.loads((round_dir(args.output, 0) / "reference.json").read_text())
    g_old_r0 = gate.reference_gradients(
        model,
        {
            name: [{"ids": x["ids"], "mask": x["mask"]} for x in rows]
            for name, rows in reference0.items()
        },
        args.eval_batch_size,
    )
    write_json(directory / "prod.json", prod_reading(ctx, prod_plan(fixed), g_old_r0))
    write_json(directory / "gate.json", result)
    print(f"[{ts()}] wrote {directory / 'gate.json'}", flush=True)


# ---- CLI ----


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--arm", choices=("a", "b"), default="a")
    parser.add_argument("--corpus", type=Path, default=Path("data/dreamer_corpus.jsonl"))
    parser.add_argument("--fixed-sets", type=Path, default=Path("data/dreamer_fixed_sets.json"))
    parser.add_argument("--abstentions", type=Path, default=Path("data/dreamer_abstentions.json"))
    parser.add_argument("--init-adapter", type=Path, default=DEFAULT_WARMSTART)
    parser.add_argument(
        "--gate", type=Path, required=True, help="the box gate's gate.json (round 0)"
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, help="default ../.cache/lama_ckl/dreamer/<arm>")
    parser.add_argument("--pool-k", type=int, default=POOL_K)
    parser.add_argument("--dreams-n", type=int, default=DREAMS_N)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--tokens", type=int, default=DREAM_TOKENS, help="dream tokens")
    parser.add_argument("--fp32-forward", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-name", default="mamba2_2_7b")
    parser.add_argument(
        "--batch-size", type=int, default=POOL_K * DREAMS_N, help="dreams per launch"
    )
    parser.add_argument("--sft-batch", type=int, default=SFT_BATCH)
    parser.add_argument(
        "--eval-batch-size", type=int, default=4, help="rows per gradient-off forward"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.arm == "b":
        raise SystemExit(
            "arm (b) is not built: its mechanism spec block is still to be written and "
            "signed off (DISCUSSION-20261024 sec 4.52, sec 6.6)"
        )
    if not torch.cuda.is_available():
        raise SystemExit("the dreamer requires a GPU")
    args.output = args.output or Path("../.cache/lama_ckl/dreamer") / args.arm
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"[{ts()}] dreamer arm {args.arm}: {args.rounds} rounds from {args.init_adapter}, "
        f"P {args.rows}, pool {args.pool_k}x{args.dreams_n}, {args.tokens}-token dreams, "
        f"{'fp32' if args.fp32_forward else 'bf16 with fp32 masters (not the sec 3.2 forward)'}"
        f" forward, output {args.output}",
        flush=True,
    )
    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    fixed = json.loads(args.fixed_sets.read_text())
    stems = json.loads(args.abstentions.read_text())["stems"]
    for r in range(args.rounds):
        if round_complete(args.output, r):
            print(f"[{ts()}] round {r} complete, skipped", flush=True)
            continue
        run_round(args, r, corpus, fixed, stems)
        free()
    run_final(args, fixed)


if __name__ == "__main__":
    main()
