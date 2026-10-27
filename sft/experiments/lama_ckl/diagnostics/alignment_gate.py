"""Experiment: lama-ckl

Alignment gate (DISCUSSION-20261024 sec 3.5 item 2, sec 6.5) and the
double-backward check (sec 3.5 item 2b, the `double-backward` subcommand).

Order: (vii) the retained-pool pass (`dreamer_corpus retained-pool`, reused
when its file exists), the fit probe, the R draw (34/33/33 from retained-pool
known facts, abstention rows, warm-start rows) and g_old per source, then per
gate row: wake from a drawn frame, nine pool samples (the row's fillers),
teacher rows for the hand-written dreams forced from S_W, g_new in both forms,
the per-dream gradients, (i) the lr sweep, (ii) the in-process floor repeats,
(iii) the per-dream scalars, (iv) discrimination, (vi) norms by kind; then the
floor's second process, (v) the h fit and kappa, the cell alpha and c, the
verdict. `--instrument-only` runs (i), (ii), (v), (vii) and gives no verdict.
Every measured drop is on the true-reply loss L(C); the contrast drop is
logged beside it.

`gate_dreams.json` (analyst-written, committed beside this module):

    {"<gate row passage_id>": [{"kind": "revisit" | "acknowledgement" | "unrelated",
                                "text": "<dream body>"}, ...]}

Nine entries per gate row, three per kind; the first revisit is the row's
sweep batch. `text` is the dream as the model writes it after the EOC prompt:
role markers and `<|endoftext|>` spelled as their token strings, tokenized
with add_special_tokens=False, no `<|endofconversation|>` (a forced dream ends
as an EOC-stopped dream does, its last position predicting the unscored EOC),
at most 128 tokens. A revisit does not contain the row's answer string.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F

from experiments.dream_distillation import dream_batch_gradient, flat_grad
from experiments.dream_generation import _repeat_state, generate_replay_dreams
from experiments.dream_types import CachedDream
from experiments.lama_ckl.protocol import WAKE_FRAMES, dream_prompt_ids, run_conversational_wake
from experiments.lama_ckl.runner import EVIDENCE_TOKENS, KL_TEMPERATURE, REPLY_TOKENS
from experiments.lama_ckl.split import DEFAULT_WARMSTART
from progress import fmt_duration, heartbeat, ts

LRS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
HALF = 0.5
TOP_LABELLINGS = 4
PASS_ROWS = 4
KINDS = ("revisit", "acknowledgement", "unrelated")
PER_KIND = 3
HAND_TOKENS = 128
FILLERS = 9
REPEATS = 3
SAMPLE_TEMPERATURE = 1.0
R_DRAW = {"facts": 34, "abstention": 33, "warm_start": 33}
# Dolly plain : UltraChat about 1:4 by conversation count, the warm start's proportion.
R_WARM_DOLLY = round(R_DRAW["warm_start"] / 5)
GATE_DREAMS = Path(__file__).with_name("gate_dreams.json")

Seq = tuple[list[int], list[bool]]


# ---- arithmetic (no model) ----


def target_mask(
    ids: Sequence[int], *, user_id: int, asst_id: int, eos_id: int, eoc_id: int
) -> list[bool]:
    """True on the tokens of an assistant turn's content: no marker, `<eos>` or `<eoc>`."""
    mask, in_reply = [], False
    for token in ids:
        if token in (asst_id, user_id, eos_id, eoc_id):
            in_reply = token == asst_id
            mask.append(False)
        else:
            mask.append(in_reply)
    return mask


def alpha_rule(
    lrs: Sequence[float], drops: Sequence[float], predictions: Sequence[float]
) -> tuple[float | None, bool]:
    """(alpha, edge_found): the largest lr whose drop is at least HALF a positive
    prediction, None when none passes; no edge when the top rung passes."""
    passing = [
        lr
        for lr, drop, predicted in zip(lrs, drops, predictions, strict=True)
        if predicted > 0 and drop >= HALF * predicted
    ]
    alpha = max(passing, default=None)
    return alpha, alpha != max(lrs)


def cell_constants(
    alphas: Sequence[float | None], norms: Sequence[float]
) -> tuple[float | None, float]:
    """The cell's alpha (min over passing rows) and c (max first-batch ||G||)."""
    return min((a for a in alphas if a is not None), default=None), max(norms)


def permutation_test(drops: Sequence[float], revisit: Sequence[int]) -> tuple[float, int, float]:
    """(statistic, rank, p): mean drop of the `revisit` batches minus the mean of
    the others; rank counts the labellings of len(revisit) batches whose statistic
    is at or above the observed one (1 is the top)."""

    def statistic(chosen: tuple[int, ...]) -> float:
        inside = [drops[i] for i in chosen]
        outside = [d for i, d in enumerate(drops) if i not in chosen]
        return statistics.fmean(inside) - statistics.fmean(outside)

    observed = statistic(tuple(sorted(revisit)))
    labellings = list(itertools.combinations(range(len(drops)), len(revisit)))
    rank = sum(statistic(labelling) >= observed for labelling in labellings)
    return observed, rank, rank / len(labellings)


def fit_h(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Least squares y = h*x + b; returns (h, b)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("h fit needs at least two distinct x")
    h = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return h, my - h * mx


def kappa_rule(
    h: float, alpha: float, *, edge_found: bool, last_good_h: float | None = None
) -> tuple[float, float, str]:
    """(kappa, h used, status) by the sec 3.2 h rule; status "stop" stops the session."""
    if h > 0:
        return alpha * h, h, "fit"
    if not edge_found:
        return 0.0, h, "no-edge-zero"
    if last_good_h is not None:
        return alpha * last_good_h, last_good_h, "reused"
    return 0.0, h, "stop"


def score_terms(
    batch: Sequence[int],
    *,
    gram: torch.Tensor,
    new: torch.Tensor,
    old: torch.Tensor,
    alpha: float,
    kappa: float,
    c: float,
) -> tuple[float, float, float, float]:
    """(learn, quadratic, hinge sum, clip factor) of score(B) for G_B the mean of
    the batch's per-dream gradients, from the Gram matrix, the dots with g_new
    (P,) and the dots with each g_old^k (P, K)."""
    index = torch.tensor(list(batch))
    n = len(batch)
    norm2 = float(gram[index][:, index].sum()) / n**2
    clip = min(1.0, c / norm2**0.5) if norm2 > 0 else 1.0
    learn = alpha * clip * float(new[index].sum()) / n
    quadratic = 0.5 * alpha * kappa * clip**2 * norm2
    hinge = sum(max(0.0, -alpha * clip * float(v) / n) for v in old[index].sum(0))
    return learn, quadratic, hinge, clip


def score(batch: Sequence[int], **scalars: torch.Tensor | float) -> float:
    learn, quadratic, hinge, _ = score_terms(batch, **scalars)
    return learn - quadratic - hinge


def assemble(
    candidates: Iterable[int], n: int, **scalars: torch.Tensor | float
) -> tuple[list[int] | None, float | None]:
    """Greedy to exactly n dreams, then one pass of swaps (sec 3.2); (None, None)
    when fewer than n candidates. Candidates exclude dreams with no scored position."""
    pool = list(candidates)
    if len(pool) < n:
        return None, None
    batch: list[int] = []
    while len(batch) < n:
        batch.append(
            max((i for i in pool if i not in batch), key=lambda i: score([*batch, i], **scalars))
        )
    best = score(batch, **scalars)
    for slot in range(n):
        for other in pool:
            if other in batch:
                continue
            trial = [*batch[:slot], other, *batch[slot + 1 :]]
            value = score(trial, **scalars)
            if value > best:
                batch, best = trial, value
    return batch, best


def bf16_spacing(x: torch.Tensor) -> torch.Tensor:
    """The bf16 spacing at each value (8 significand bits); zeros give 2**-8."""
    _, exponent = torch.frexp(x.float())
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), exponent - 8)


def bf16_change_fraction(theta: torch.Tensor, delta: torch.Tensor) -> float:
    """Fraction of coordinates whose bf16 cast changes when theta moves by -delta."""
    return float(((theta - delta).bfloat16() != theta.bfloat16()).float().mean())


def theta_stats(trainable_pt: Path) -> dict[str, float]:
    """Coordinate RMS, median |theta| and median bf16 spacing (nonzero coordinates)."""
    theta = torch.cat(
        [t.float().flatten() for t in torch.load(trainable_pt, weights_only=True).values()]
    )
    nonzero = theta[theta != 0]
    return {
        "coordinates": theta.numel(),
        "rms": float(theta.pow(2).mean().sqrt()),
        "median_abs": float(theta.abs().median()),
        "median_bf16_spacing": float(bf16_spacing(nonzero).median()),
        "zeros": theta.numel() - nonzero.numel(),
    }


# ---- data ----


def load_gate_dreams(path: Path, gate_rows: Sequence[Mapping[str, object]]) -> dict[str, list]:
    """The hand-written dreams per gate row id, checked against the schema above."""
    dreams = json.loads(path.read_text())
    for row in gate_rows:
        row_id = str(row["passage_id"])
        entries = dreams.get(row_id, [])
        kinds = [d["kind"] for d in entries]
        if len(entries) != len(KINDS) * PER_KIND or any(kinds.count(k) != PER_KIND for k in KINDS):
            raise ValueError(f"{row_id}: need {PER_KIND} of each of {KINDS}, got {kinds}")
        answer = str(row["question"]["answer"]).lower()
        for d in entries:
            if d["kind"] == "revisit" and answer in d["text"].lower():
                raise ValueError(f"{row_id}: a revisit quotes the answer {answer!r}")
    return dreams


def _render(messages: Sequence[Mapping[str, str]], tokenizer, mod: ModuleType) -> Seq:
    from preparation.conversations import format_conversation

    ids, mask, _ = format_conversation(
        [dict(m) for m in messages],
        tokenizer,
        10**9,
        mod.USER_OPEN,
        mod.ASST_OPEN,
        mod.EOC,
        train_user_turns=True,
    )
    return ids, mask


def markers(tokenizer, mod: ModuleType) -> dict[str, int]:
    ids_of = tokenizer.convert_tokens_to_ids
    return {
        "user_id": int(ids_of(mod.USER_OPEN)),
        "asst_id": int(ids_of(mod.ASST_OPEN)),
        "eos_id": int(tokenizer.eos_token_id),
        "eoc_id": int(ids_of(mod.EOC)),
    }


def reply_seq(messages: Sequence[Mapping[str, str]], tokenizer, mod: ModuleType) -> Seq:
    """C or C-: the rendered row, loss on the reply's content tokens (sec 3.2 row 3)."""
    ids, _ = _render(messages, tokenizer, mod)
    return ids, target_mask(ids, **markers(tokenizer, mod))


def draw_reference(
    corpus: Sequence[Mapping[str, object]],
    retained: Sequence[Mapping[str, object]],
    tokenizer,
    mod: ModuleType,
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    """Round 0's R (sec 3.2 row 4). Facts: retained-pool known rows, loss on their
    currently-correct answer tokens and both stop tokens (sec 4.51). Abstention
    rows: loss on the reply and both stop tokens. Warm start: R_WARM_DOLLY Dolly
    plain rows and the rest trained-on UltraChat rows, a short pool filled from the
    other, loss as the warm start trained. Each row carries its id, ids and mask."""
    rng = random.Random(seed)
    ids = markers(tokenizer, mod)
    stops = (ids["eos_id"], ids["eoc_id"])
    by_kind: dict[str, list[Mapping[str, object]]] = {}
    for row in corpus:
        by_kind.setdefault(str(row["kind"]), []).append(row)

    def take(pool: list, count: int) -> list:
        return rng.sample(pool, count)

    facts = []
    for r in take([r for r in retained if r["side"] == "known"], R_DRAW["facts"]):
        messages = [
            {"role": "user", "content": str(r["question"])},
            {"role": "assistant", "content": str(r["answer"])},
        ]
        seq_ids, _ = _render(messages, tokenizer, mod)
        if tuple(seq_ids[-2:]) != stops or not r["correct_positions"]:
            raise ValueError(f"retained row {r['squad_id']}: no stop tokens or no correct token")
        targets = {*r["correct_positions"], len(seq_ids) - 2, len(seq_ids) - 1}
        mask = [i in targets for i in range(len(seq_ids))]
        facts.append({"squad_id": r["squad_id"], "ids": seq_ids, "mask": mask})

    abstention = []
    for r in take(by_kind.get("abstention", []), R_DRAW["abstention"]):
        seq_ids, _ = _render(r["messages"], tokenizer, mod)
        reply = target_mask(seq_ids, **ids)
        mask = [m or t in stops for m, t in zip(reply, seq_ids, strict=True)]
        abstention.append({"squad_id": r["squad_id"], "ids": seq_ids, "mask": mask})

    dolly = [({"dolly_line": r["dolly_line"]}, r) for r in by_kind.get("dolly_plain", [])]
    ultrachat = [
        ({"ultrachat_index": r["ultrachat_index"]}, r)
        for r in by_kind.get("warm_start", [])
        if not r["packer_dropped"]
    ]
    n_dolly = min(len(dolly), max(R_WARM_DOLLY, R_DRAW["warm_start"] - len(ultrachat)))
    n_ultrachat = R_DRAW["warm_start"] - n_dolly
    if (n_dolly, n_ultrachat) != (R_WARM_DOLLY, R_DRAW["warm_start"] - R_WARM_DOLLY):
        print(
            f"[{ts()}] WARNING: warm-start R pools short (Dolly plain {len(dolly)}, trained-on "
            f"UltraChat {len(ultrachat)}): drawing {n_dolly} Dolly and {n_ultrachat} UltraChat",
            flush=True,
        )
    warm = []
    for key, r in take(dolly, n_dolly) + take(ultrachat, n_ultrachat):
        seq_ids, mask = _render(r["messages"], tokenizer, mod)
        warm.append({**key, "ids": seq_ids, "mask": mask})
    return {"facts": facts, "abstention": abstention, "warm_start": warm}


def retained_pool_pass(
    init_adapter: Path, corpus: Path, model_name: str, batch_size: int, fp32: bool
) -> Path:
    """Step (vii): `dreamer_corpus retained-pool` on the checkpoint in the gate's
    forward, reused when that mode's file exists. Loads and frees its own model, so
    call it before loading the gate's."""
    from experiments.lama_ckl.dreamer_corpus import retained_pool, retained_pool_path

    path = retained_pool_path(init_adapter, corpus, fp32)
    if path.exists():
        print(f"[{ts()}] (vii) retained pool exists, reused: {path}", flush=True)
        return path
    retained_pool(
        argparse.Namespace(
            init_adapter=init_adapter,
            corpus=corpus,
            output=path,
            exclude_passages=None,
            size=2000,
            model_name=model_name,
            batch_size=batch_size,
            fp32_forward=fp32,
        )
    )
    torch.cuda.empty_cache()
    return path


# ---- model ----


def load_model(model_name: str, adapter: Path, fp32: bool, device: torch.device):
    """(model, MasterOptimizer over SGD, tokenizer, model module), fp32 before the load."""
    from models.common import build_tokenizer

    from training.checkpoints import load_checkpoint
    from training.master_weights import MasterOptimizer, fp32_forward

    mod = importlib.import_module(f"models.{model_name}")
    hooks = importlib.import_module(f"models.{model_name}.train_hooks")
    config = json.loads((adapter / "lora_config.json").read_text())
    model, trainable = hooks.setup_training(
        device, int(config["rank"]), float(config["alpha"]), 0.0
    )
    if fp32:
        fp32_forward(model)
    load_checkpoint(model, adapter)
    if hasattr(model, "set_grad_checkpoint"):
        model.set_grad_checkpoint(True, 64)
    opt = MasterOptimizer(trainable, torch.optim.SGD, lr=LRS[0], momentum=0.0, weight_decay=0.0)
    return model, opt, build_tokenizer(mod), mod


def row_losses(model, seqs: Sequence[Seq]) -> torch.Tensor:
    """Per-row mean cross-entropy over each row's masked target tokens, from empty
    state, in one right-padded forward."""
    device = next(model.parameters()).device
    width = max(len(ids) for ids, _ in seqs)
    tokens = torch.tensor([ids + [ids[-1]] * (width - len(ids)) for ids, _ in seqs], device=device)
    mask = torch.tensor(
        [mask[1:] + [False] * (width - len(mask)) for _, mask in seqs], device=device
    )
    logits, _ = model(tokens, state=None)
    nll = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), tokens[:, 1:], reduction="none")
    counts = mask.sum(1)
    if bool((counts == 0).any()):
        raise ValueError("a row has no target token")
    return (nll * mask).sum(1) / counts


def losses(model, seqs: Sequence[Seq]) -> list[float]:
    with torch.no_grad():
        return row_losses(model, seqs).tolist()


def mean_loss_grad(model, seqs: Sequence[Seq], batch_size: int, label: str = "") -> torch.Tensor:
    """The flat gradient of the mean over rows of each row's per-token mean loss."""
    started = time.time()
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start : start + batch_size]
        (row_losses(model, chunk).sum() / len(seqs)).backward()
        if label:
            heartbeat()
            done = start + len(chunk)
            print(
                f"[{ts()}] {label} {done}/{len(seqs)} rows, "
                f"ETA {fmt_duration((time.time() - started) / done * (len(seqs) - done))}",
                flush=True,
            )
    grad = flat_grad(model).float()
    model.zero_grad(set_to_none=True)
    return grad


def g_new(model, c: Seq, c_negative: Seq) -> tuple[torch.Tensor, torch.Tensor]:
    """(contrast, plain): grad L(C) - grad L(C-), and grad L(C)."""
    plain = mean_loss_grad(model, [c], 1)
    return plain - mean_loss_grad(model, [c_negative], 1), plain


def reference_gradients(
    model, reference: Mapping[str, Sequence[Mapping[str, object]]], batch_size: int
) -> dict[str, torch.Tensor]:
    """g_old^k per source: the gradient of the mean over its rows (sec 3.2 row 4)."""
    return {
        name: mean_loss_grad(
            model, [(r["ids"], r["mask"]) for r in rows], batch_size, f"g_old {name}"
        )
        for name, rows in reference.items()
    }


def dream_gradient(model, dream: CachedDream, ids: Mapping[str, int]) -> torch.Tensor | None:
    """The cell's per-dream gradient (uniform weights), None with no scored position."""
    out = dream_batch_gradient(model, [dream], KL_TEMPERATURE, **ids, loss_weights="uniform")
    grad = flat_grad(model).float() if out["scored_dreams"] else None
    model.zero_grad(set_to_none=True)
    return grad


def dream_scalars(
    grads: torch.Tensor, g_new: torch.Tensor, g_old: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Gram matrix (P, P), dots with g_new (P,), dots with each g_old^k (P, K) and
    norms (P,) for stacked per-dream gradients (P, d); score() takes the first three."""
    gram = (grads @ grads.T).double().cpu()
    old = (
        torch.stack([grads @ v for v in g_old.values()], 1)
        if g_old
        else grads.new_zeros(len(grads), 0)
    )
    return {
        "gram": gram,
        "new": (grads @ g_new).double().cpu(),
        "old": old.double().cpu(),
        "norms": gram.diagonal().sqrt(),
    }


def forced_dreams(
    model, wake_state, prompt: Sequence[int], bodies: Sequence[Sequence[int]], decode
) -> list[CachedDream]:
    """Teacher rows for hand-written dreams: prompt + body forced from a copy of the
    wake state in one batch, logits kept at every position as generation keeps them."""
    rows = [list(prompt) + list(body) for body in bodies]
    width = max(map(len, rows))
    device = next(model.parameters()).device
    tokens = torch.tensor([r + [r[-1]] * (width - len(r)) for r in rows], device=device)
    with torch.no_grad():
        logits, _ = model(tokens, state=_repeat_state(wake_state, len(rows)))
    return [
        CachedDream(
            dream_ids=row,
            token_texts=[decode(t) for t in row],
            teacher_logits=logits[i, : len(row)].float().cpu(),
            cue_flags=[False] * len(row),
            prefix_len=len(prompt),
            stop_reason="eoc",
            divergence=[],
            gate_positions=[],
            queries=[],
            spectra=[],
            ranks={},
            bases={},
        )
        for i, row in enumerate(rows)
    ]


def stepped_losses(model, opt, grad: torch.Tensor, lr: float, seqs: Sequence[Seq]) -> list[float]:
    """One unclipped SGD step of size lr along grad, the rows' losses, then theta restored."""
    saved = {
        "masters": [m.detach().clone() for m in opt.masters],
        "optimizer": opt.optimizer.state_dict(),
    }
    offset = 0
    for param in opt.params:
        param.grad = grad[offset : offset + param.numel()].view_as(param).to(param.dtype)
        offset += param.numel()
    assert offset == grad.numel(), "gradient and trainable set differ in length"
    opt.optimizer.param_groups[0]["lr"] = lr
    opt.step()
    opt.zero_grad()
    after = losses(model, seqs)
    opt.load_state_dict(saved)
    return after


def fit_probe(model, tokens: int) -> float:
    """Peak GB of one forward and backward over a `tokens`-long row at batch 1."""
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(1000, 30000, (tokens,), generator=torch.Generator().manual_seed(0))
    row_losses(model, [(ids.tolist(), [True] * tokens)]).sum().backward()
    model.zero_grad(set_to_none=True)
    return torch.cuda.max_memory_allocated() / 2**30


# ---- the gate ----


def prepare_row(
    model,
    tokenizer,
    mod,
    row: Mapping[str, object],
    dreams: Sequence[Mapping[str, str]],
    *,
    tokens: int,
    seed: int,
    dream_batch_size: int,
) -> dict[str, object]:
    """Wake (sec 3.2 row 1), the nine fillers, the forced hand-written dreams, C and C-."""
    device = next(model.parameters()).device
    ids = markers(tokenizer, mod)
    row_id = str(row["passage_id"])
    frame = random.Random(f"gate-frame-{row_id}").choice(WAKE_FRAMES)
    wake, state = run_conversational_wake(
        model,
        tokenizer,
        [str(row["passage"])],
        None,
        mod.USER_OPEN,
        mod.ASST_OPEN,
        mod.EOC,
        reply_tokens=REPLY_TOKENS,
        evidence_tokens=EVIDENCE_TOKENS,
        device=device,
        temperature=SAMPLE_TEMPERATURE,
        seed=2 * seed,
        frame=frame,
    )
    prompt = dream_prompt_ids(tokenizer, mod.USER_OPEN, mod.ASST_OPEN, mod.EOC, device)

    def decode(token: int) -> str:
        return tokenizer.decode([token])

    fillers, _ = generate_replay_dreams(
        model,
        state,
        prompt,
        count=FILLERS,
        batch_size=dream_batch_size,
        seed=2 * seed + 1,
        n_tokens=tokens,
        temperature=SAMPLE_TEMPERATURE,
        decode_token=decode,
        stop_id=ids["eoc_id"],
        turn_id=ids["eos_id"],
    )
    bodies = [list(tokenizer(d["text"], add_special_tokens=False)["input_ids"]) for d in dreams]
    for d, body in zip(dreams, bodies, strict=True):
        if len(body) > HAND_TOKENS or ids["eoc_id"] in body:
            raise ValueError(f"{row_id}: a {d['kind']} dream has {len(body)} tokens or an EOC")
    hand = forced_dreams(model, state, prompt[0].tolist(), bodies, decode)
    kinds = [d["kind"] for d in dreams]
    for kind in KINDS:
        text = "".join(hand[kinds.index(kind)].token_texts)
        print(f"[{ts()}] {row_id} forced {kind}: {text[:400]!r}", flush=True)
    for dream in fillers[:2]:
        print(f"[{ts()}] {row_id} filler: {''.join(dream.token_texts)[:400]!r}", flush=True)
    return {
        "row_id": row_id,
        "frame": frame,
        "reply": wake["turns"][0]["assistant"],
        "fillers": fillers,
        "hand": hand,
        "kinds": kinds,
        "c": reply_seq(row["question"]["messages"], tokenizer, mod),
        "c_negative": reply_seq(row["negative"]["messages"], tokenizer, mod),
    }


def batch_grad(hand: torch.Tensor, filler_sum: torch.Tensor, fillers: int) -> torch.Tensor:
    """G of one hand-written dream plus the scored fillers: the mean over scored dreams."""
    return (hand.to(filler_sum.device) + filler_sum) / (1 + fillers)


def _rung_below(alpha: float) -> float:
    return LRS[max(LRS.index(alpha) - 1, 0)]


def instrument(
    model,
    opt,
    tokenizer,
    mod,
    gate_rows: Sequence[Mapping[str, object]],
    gate_dreams: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    tokens: int,
    model_name: str,
    init_adapter: Path,
    fp32: bool,
    seed: int = 0,
    dream_batch_size: int = FILLERS,
    verdict: bool = False,
    g_old: Mapping[str, torch.Tensor] | None = None,
    last_good_h: float | None = None,
) -> dict[str, object]:
    """Steps (i), (ii), (v) on every gate row, and with `verdict` (iii), (iv), (vi)
    and the pass (g_old required). Returns the gate.json body without R and (vii)."""
    ids = markers(tokenizer, mod)
    rows_out: list[dict[str, object]] = []
    jobs: list[dict[str, object]] = []
    for index, row in enumerate(gate_rows):
        started = time.time()
        row_id = str(row["passage_id"])
        dreams = gate_dreams[row_id]
        prep = prepare_row(
            model,
            tokenizer,
            mod,
            row,
            dreams,
            tokens=tokens,
            seed=seed * 100 + index,
            dream_batch_size=dream_batch_size,
        )
        c, c_negative = prep["c"], prep["c_negative"]
        contrast, plain = g_new(model, c, c_negative)
        filler_sum, scored_fillers = torch.zeros_like(plain), 0
        for dream in prep["fillers"]:
            grad = dream_gradient(model, dream, ids)
            if grad is not None:
                filler_sum += grad
                scored_fillers += 1

        first_revisit = prep["kinds"].index("revisit")
        wanted = range(len(dreams)) if verdict else [first_revisit]
        hand_grads: dict[int, torch.Tensor | None] = {}
        for done, i in enumerate(wanted, 1):
            grad = dream_gradient(model, prep["hand"][i], ids)
            hand_grads[i] = None if grad is None else grad.cpu()
            heartbeat()
            print(f"[{ts()}] {row_id} hand-written gradients {done}/{len(wanted)}", flush=True)
        if None in hand_grads.values():
            raise ValueError(f"{row_id}: a hand-written dream has no scored position")
        out: dict[str, object] = {
            "row_id": row_id,
            "question": row["question"]["question"],
            "answer": row["question"]["answer"],
            "frame": prep["frame"],
            "wake_reply": prep["reply"],
            "scored_fillers": scored_fillers,
            "filler_lengths": [len(d.dream_ids) for d in prep["fillers"]],
        }

        # (i) the sweep on the first revisit batch.
        g0 = batch_grad(hand_grads[first_revisit], filler_sum, scored_fillers)
        dot_plain, dot_contrast = float(g0 @ plain), float(g0 @ contrast)
        theta = torch.cat([m.detach().float().flatten() for m in opt.masters])
        before = losses(model, [c, c_negative])
        sweep = []
        for lr in LRS:
            after = stepped_losses(model, opt, g0, lr, [c, c_negative])
            point = {
                "lr": lr,
                "predicted": lr * dot_plain,
                "drop": before[0] - after[0],
                "contrast_predicted": lr * dot_contrast,
                "contrast_drop": (before[0] - before[1]) - (after[0] - after[1]),
                "bf16_changed": bf16_change_fraction(theta, lr * g0),
            }
            sweep.append(point)
            print(
                f"[{ts()}] {row_id} (i) lr {lr:.0e}: L(C) drop {point['drop']:+.4e} predicted "
                f"{point['predicted']:+.4e}; contrast {point['contrast_drop']:+.4e} predicted "
                f"{point['contrast_predicted']:+.4e}; bf16 changed {point['bf16_changed']:.4f}",
                flush=True,
            )
        del theta
        alpha, edge = alpha_rule(LRS, [p["drop"] for p in sweep], [p["predicted"] for p in sweep])
        out.update(
            {
                "L_C": before[0],
                "L_C_negative": before[1],
                "grad_norm": float(g0.norm()),
                "sweep": sweep,
                "alpha": alpha,
                "edge_found": edge,
            }
        )
        print(
            f"[{ts()}] {row_id} (i) alpha {alpha} ({'edge' if edge else 'no edge found'}), "
            f"||G|| {out['grad_norm']:.4e}, L(C) {before[0]:.4f} nats",
            flush=True,
        )

        # (ii) the in-process half of the floor.
        if alpha is not None:
            rung = _rung_below(alpha)
            repeats = []
            for _ in range(REPEATS):
                fresh = losses(model, [c])[0]
                repeats.append(fresh - stepped_losses(model, opt, g0, rung, [c])[0])
            out["floor_rung"], out["floor_predicted"] = rung, rung * dot_plain
            out["floor_in_process"] = repeats
            jobs.append({"row_id": row_id, "lr": rung, "grad": g0.cpu(), "c": c})
            print(f"[{ts()}] {row_id} (ii) in-process drops at {rung:.0e}: {repeats}", flush=True)

        if verdict and alpha is not None:
            assert g_old is not None, "the verdict needs g_old"
            rung = _rung_below(alpha)
            stacked = torch.stack([g.to(plain.device) for g in hand_grads.values()])
            scalars = dream_scalars(stacked, contrast, g_old)
            plain_dots = (stacked @ plain).double().cpu()
            per_dream = []
            for k, i in enumerate(hand_grads):
                norm = float(scalars["norms"][k])
                per_dream.append(
                    {
                        "index": i,
                        "kind": prep["kinds"][i],
                        "s_contrast": alpha * float(scalars["new"][k]),
                        "s_plain": alpha * float(plain_dots[k]),
                        "o": {
                            name: alpha * float(scalars["old"][k, j])
                            for j, name in enumerate(g_old)
                        },
                        "norm": norm,
                        "cos_contrast": float(scalars["new"][k]) / (norm * float(contrast.norm())),
                        "cos_plain": float(plain_dots[k]) / (norm * float(plain.norm())),
                    }
                )
            del stacked
            out["dreams"] = per_dream
            for d in per_dream:
                print(
                    f"[{ts()}] {row_id} (iii) {d['kind']:15s} s(contrast) {d['s_contrast']:+.4e} "
                    f"s(plain) {d['s_plain']:+.4e} |grad| {d['norm']:.4e} "
                    f"cos {d['cos_contrast']:+.4f} o {d['o']}",
                    flush=True,
                )

            # (iv) discrimination: nine batches, one rung below alpha, shared fresh before-loss.
            before_iv = losses(model, [c, c_negative])
            drops, contrast_drops = [], []
            for i in range(len(dreams)):
                grad = batch_grad(hand_grads[i], filler_sum, scored_fillers)
                after = stepped_losses(model, opt, grad, rung, [c, c_negative])
                drops.append(before_iv[0] - after[0])
                contrast_drops.append((before_iv[0] - before_iv[1]) - (after[0] - after[1]))
                print(
                    f"[{ts()}] {row_id} (iv) {prep['kinds'][i]:15s} L(C) drop {drops[-1]:+.4e} "
                    f"contrast drop {contrast_drops[-1]:+.4e}",
                    flush=True,
                )
            revisits = [i for i, k in enumerate(prep["kinds"]) if k == "revisit"]
            statistic, rank, p = permutation_test(drops, revisits)
            s = [d["s_contrast"] for d in per_dream]
            ranked = min(s[i] for i in revisits) > max(
                v for i, v in enumerate(s) if i not in revisits
            )
            acks = [i for i, k in enumerate(prep["kinds"]) if k == "acknowledgement"]
            unrelated = [i for i, k in enumerate(prep["kinds"]) if k == "unrelated"]
            out["discrimination"] = {
                "lr": rung,
                "before": before_iv,
                "drops": drops,
                "contrast_drops": contrast_drops,
                "statistic": statistic,
                "rank": rank,
                "p": p,
                "s_ranks_revisits_first": ranked,
                "ack_minus_unrelated": statistics.fmean(drops[i] for i in acks)
                - statistics.fmean(drops[i] for i in unrelated),
                "pass": rank <= TOP_LABELLINGS and ranked,
            }
            # (vi) gradient norms by kind.
            out["norms_by_kind"] = {
                kind: statistics.fmean(d["norm"] for d in per_dream if d["kind"] == kind)
                for kind in KINDS
            }
            print(
                f"[{ts()}] {row_id} (iv) statistic {statistic:+.4e} rank {rank}/84 p {p:.4f} "
                f"s ranks revisits first {ranked}; (vi) norms {out['norms_by_kind']}",
                flush=True,
            )
        del contrast, plain, filler_sum, hand_grads, prep
        torch.cuda.empty_cache()
        rows_out.append(out)
        heartbeat()
        print(
            f"[{ts()}] row {index + 1}/{len(gate_rows)} {row_id} done in "
            f"{fmt_duration(time.time() - started)}",
            flush=True,
        )

    # (ii) the second process.
    worker_drops = run_worker(model, jobs, model_name=model_name, adapter=init_adapter, fp32=fp32)
    for out in rows_out:
        if "floor_in_process" in out:
            out["floor_worker"] = worker_drops[out["row_id"]]
            out["floor"] = statistics.stdev([*out["floor_in_process"], *out["floor_worker"]])
            out["above_floor"] = out["floor_predicted"] > out["floor"]

    # (v) h on the rungs at or below each passing row's alpha.
    xs, ys = [], []
    for out in rows_out:
        if out["alpha"] is None:
            continue
        for point in out["sweep"]:
            if point["lr"] <= out["alpha"]:
                xs.append(-0.5 * point["lr"] ** 2 * out["grad_norm"] ** 2)
                ys.append(point["drop"] - point["predicted"])
    alpha, c = cell_constants([o["alpha"] for o in rows_out], [o["grad_norm"] for o in rows_out])
    edge = any(o["edge_found"] for o in rows_out if o["alpha"] is not None)
    result: dict[str, object] = {"rows": rows_out, "alpha": alpha, "c": c, "edge_found": edge}
    if alpha is not None and len(set(xs)) > 1:
        h, intercept = fit_h(xs, ys)
        kappa, h_used, status = kappa_rule(h, alpha, edge_found=edge, last_good_h=last_good_h)
        result.update(
            {
                "h_fit": h,
                "intercept": intercept,
                "h": h_used,
                "kappa": kappa,
                "kappa_status": status,
            }
        )
    else:
        result.update({"h_fit": None, "h": None, "kappa": None, "kappa_status": "no-fit"})
    if verdict:
        for out in rows_out:
            out["pass"] = bool(
                out["alpha"] is not None
                and out.get("above_floor")
                and out.get("discrimination", {}).get("pass")
            )
        passed = sum(o["pass"] for o in rows_out)
        result["rows_passed"] = passed
        result["verdict"] = (
            "stop"
            if result["kappa_status"] == "stop"
            else "pass"
            if passed >= PASS_ROWS
            else "fail"
        )
    return result


def run_worker(
    model, jobs: Sequence[Mapping[str, object]], *, model_name: str, adapter: Path, fp32: bool
) -> dict[str, list[float]]:
    """Run the floor's second process on `jobs`; the gate's model waits on the CPU."""
    if not jobs:
        return {}
    device = next(model.parameters()).device
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / "floor.pt"
        torch.save(
            {"model_name": model_name, "adapter": str(adapter), "fp32": fp32, "jobs": list(jobs)},
            job,
        )
        model.cpu()
        torch.cuda.empty_cache()
        print(f"[{ts()}] (ii) spawning the second process for {len(jobs)} rows", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "experiments.lama_ckl.diagnostics.alignment_gate",
                "--worker",
                str(job),
            ],
            check=True,
        )
        model.to(device)
        return json.loads(job.with_suffix(".json").read_text())


def worker(job_path: Path) -> None:
    job = torch.load(job_path, weights_only=True)
    model, opt, _, _ = load_model(
        job["model_name"], Path(job["adapter"]), job["fp32"], torch.device("cuda")
    )
    drops: dict[str, list[float]] = {}
    for row in job["jobs"]:
        grad = row["grad"].cuda()
        c = (row["c"][0], row["c"][1])
        drops[row["row_id"]] = [
            losses(model, [c])[0] - stepped_losses(model, opt, grad, row["lr"], [c])[0]
            for _ in range(REPEATS)
        ]
        heartbeat()
        print(f"[{ts()}] worker {row['row_id']} drops {drops[row['row_id']]}", flush=True)
    job_path.with_suffix(".json").write_text(json.dumps(drops))


def summary_table(result: Mapping[str, object]) -> str:
    lines = [
        "| row | alpha | edge | ‖G‖ | L(C) | floor | pred@rung | stat | rank/84 | s order | pass |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for o in result["rows"]:
        d = o.get("discrimination", {})
        lines.append(
            f"| {o['row_id']} | {o['alpha']} | {o['edge_found']} | {o['grad_norm']:.3e} | "
            f"{o['L_C']:.3f} | {o.get('floor', float('nan')):.2e} | "
            f"{o.get('floor_predicted', float('nan')):.2e} | {d.get('statistic', float('nan')):.2e} | "
            f"{d.get('rank', '-')} | {d.get('s_ranks_revisits_first', '-')} | {o.get('pass', '-')} |"
        )
    lines.append(
        f"cell alpha {result['alpha']}, c {result['c']:.4e}, h {result['h']}, "
        f"kappa {result['kappa']} ({result['kappa_status']}), verdict {result.get('verdict', '-')}"
    )
    return "\n".join(lines)


# ---- 2b: the double backward ----


def double_backward(model, params: Sequence[torch.Tensor], wake, dream, qa, block: int):
    """Research note sec 6 on one dream: L_gm = -cos(grad L_in(theta; pi_theta), g),
    backpropagated into the dreamer's logits and through them into theta, every
    forward on the manual mixer under non-reentrant checkpointing. Also returns
    the double backward's d<G, g>/d pi against a finite difference along g."""
    from models.common import blockwise_checkpoint

    dtype = model.embedding.weight.dtype

    def forward(ids: torch.Tensor, state):
        state = model._init_state(ids.shape[0], dtype) if state is None else state
        return blockwise_checkpoint(model._forward_tokens, ids, state, block)

    def flat(grads: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([g.flatten() for g in grads])

    logits, _ = forward(qa, None)
    g = flat(torch.autograd.grad(F.cross_entropy(logits[0, :-1].float(), qa[0, 1:]), params))
    g = g.detach()
    with torch.no_grad():
        _, state = model._forward_tokens(wake, model._init_state(1, dtype))
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    dreamer_logits, _ = forward(dream, state.detach())
    pi = torch.softmax(dreamer_logits[0, :-1].float(), -1)
    pi_leaf = pi.detach().requires_grad_()

    def inner(p: torch.Tensor) -> torch.Tensor:
        student, _ = forward(dream, None)
        return -(p * torch.log_softmax(student[0, :-1].float(), -1)).sum(-1).mean()

    grads = torch.autograd.grad(inner(pi), params, create_graph=True)
    loss = -F.cosine_similarity(flat(grads), g, dim=0)
    loss.backward()
    peak = torch.cuda.max_memory_allocated() - base
    theta_grad = float(torch.cat([p.grad.flatten() for p in params if p.grad is not None]).norm())

    # d<G, g>/d pi by the double backward, against -(1/N) * the directional derivative
    # of the student's log-probs along g by central difference.
    leaf_grads = torch.autograd.grad(inner(pi_leaf), params, create_graph=True)
    dot = (flat(leaf_grads) * g).sum()
    analytic = (
        torch.autograd.grad(dot, pi_leaf, allow_unused=True)[0] if dot.requires_grad else None
    )
    eps = 1e-3 / float(g.norm())
    with torch.no_grad():
        logps = []
        for sign in (1, -1):
            offset = 0
            for p in params:
                p.add_(sign * eps * g[offset : offset + p.numel()].view_as(p).to(p.dtype))
                offset += p.numel()
            student, _ = forward(dream, None)
            logps.append(torch.log_softmax(student[0, :-1].float(), -1))
            offset = 0
            for p in params:
                p.sub_(sign * eps * g[offset : offset + p.numel()].view_as(p).to(p.dtype))
                offset += p.numel()
        expected = -(logps[0] - logps[1]) / (2 * eps) / pi.shape[0]
    error = (
        float("inf") if analytic is None else float((analytic - expected).norm() / expected.norm())
    )
    return {
        "dream_tokens": dream.shape[1],
        "peak_bytes": peak,
        "loss_gm": float(loss.detach()),
        "theta_grad_norm": theta_grad,
        "dpi_relative_error": error,
        "dpi_reaches_pi": analytic is not None,
    }


def double_backward_main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(description="sec 3.5 item 2b: the double backward on 780M")
    parser.add_argument("--model-name", default="mamba2_780m")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--block", type=int, default=64, help="checkpoint block, tokens")
    args = parser.parse_args(argv)
    device = torch.device("cuda")
    hooks = importlib.import_module(f"models.{args.model_name}.train_hooks")
    model, params = hooks.setup_training(device, 16, 32.0, 0.0)
    model.torch_norms = True
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(std=1e-3)  # a zero B leaves lora_A without gradient
    generator = torch.Generator().manual_seed(0)

    def ids() -> torch.Tensor:
        return torch.randint(1000, 30000, (1, args.tokens), generator=generator).to(device)

    started = time.time()
    try:
        out = double_backward(model, params, ids(), ids(), ids(), args.block)
    except RuntimeError as error:
        print(f"[{ts()}] 2b FAIL: {error}", flush=True)
        raise SystemExit(1) from error
    per_token = out["peak_bytes"] / args.tokens
    print(
        f"[{ts()}] 2b {args.model_name}: {args.tokens}-token dream peak "
        f"{out['peak_bytes'] / 2**30:.2f} GB above the model, extrapolated to 512 tokens "
        f"{per_token * 512 / 2**30:.2f} GB (linear); L_gm {out['loss_gm']:+.4e}, "
        f"|d theta| {out['theta_grad_norm']:.3e}, d pi relative error "
        f"{out['dpi_relative_error']:.3e} (reaches pi: {out['dpi_reaches_pi']}); "
        f"{fmt_duration(time.time() - started)}",
        flush=True,
    )
    ok = out["dpi_reaches_pi"] and out["dpi_relative_error"] < 1e-2
    print(f"[{ts()}] 2b {'PASS' if ok else 'FAIL'}", flush=True)
    raise SystemExit(0 if ok else 1)


# ---- CLI ----


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--corpus", type=Path, default=Path("data/dreamer_corpus.jsonl"))
    parser.add_argument("--fixed-sets", type=Path, default=Path("data/dreamer_fixed_sets.json"))
    parser.add_argument("--gate-dreams", type=Path, default=GATE_DREAMS)
    parser.add_argument("--init-adapter", type=Path, default=DEFAULT_WARMSTART)
    parser.add_argument("--tokens", type=int, choices=(128, 512), default=128)
    parser.add_argument("--fp32-forward", action="store_true")
    parser.add_argument("--instrument-only", action="store_true")
    parser.add_argument("--output", type=Path, help="directory for gate.json")
    parser.add_argument("--model-name", default="mamba2_2_7b")
    parser.add_argument("--dream-batch-size", type=int, default=FILLERS)
    parser.add_argument("--batch-size", type=int, default=4, help="rows per g_old forward")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    if sys.argv[1:2] == ["double-backward"]:
        double_backward_main(sys.argv[2:])
        return
    args = build_parser().parse_args()
    if args.worker:
        worker(args.worker)
        return
    if not torch.cuda.is_available():
        raise SystemExit("the alignment gate requires a GPU")
    print(
        f"[{ts()}] alignment gate: {args.init_adapter}, {args.tokens}-token dreams, "
        f"{'fp32' if args.fp32_forward else 'bf16 with fp32 masters'} forward, "
        f"{'instrument only' if args.instrument_only else 'verdict'}",
        flush=True,
    )
    fixed = json.loads(args.fixed_sets.read_text())
    gate_rows = fixed["gate"]
    gate_dreams = load_gate_dreams(args.gate_dreams, gate_rows)
    pool_path = retained_pool_pass(
        args.init_adapter, args.corpus, args.model_name, args.batch_size, args.fp32_forward
    )
    stats = theta_stats(args.init_adapter / "trainable.pt")
    print(f"[{ts()}] theta {stats}", flush=True)

    device = torch.device("cuda")
    model, opt, tokenizer, mod = load_model(
        args.model_name, args.init_adapter, args.fp32_forward, device
    )
    peak = fit_probe(model, args.tokens)
    print(f"[{ts()}] fit: {args.tokens} tokens, batch 1, forward+backward peak {peak:.2f} GB")

    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    retained = json.loads(pool_path.read_text())["rows"]
    reference = draw_reference(corpus, retained, tokenizer, mod, args.seed)
    del corpus
    g_old = None if args.instrument_only else reference_gradients(model, reference, args.batch_size)
    result = instrument(
        model,
        opt,
        tokenizer,
        mod,
        gate_rows,
        gate_dreams,
        tokens=args.tokens,
        model_name=args.model_name,
        init_adapter=args.init_adapter,
        fp32=args.fp32_forward,
        seed=args.seed,
        dream_batch_size=args.dream_batch_size,
        verdict=not args.instrument_only,
        g_old=g_old,
    )
    result.update(
        {
            "date": time.strftime("%Y-%m-%d"),
            "init_adapter": str(args.init_adapter),
            "trainable_sha256": hashlib.sha256(
                (args.init_adapter / "trainable.pt").read_bytes()
            ).hexdigest(),
            "tokens": args.tokens,
            "fp32_forward": args.fp32_forward,
            "fit_peak_gb": peak,
            "theta": stats,
            "retained_pool": str(pool_path),
            "R": {
                name: [{k: v for k, v in r.items() if k not in ("ids", "mask")} for r in rows]
                for name, rows in reference.items()
            },
        }
    )
    print(summary_table(result), flush=True)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "gate.json").write_text(json.dumps(result, indent=1))
        print(f"[{ts()}] wrote {args.output / 'gate.json'}", flush=True)


if __name__ == "__main__":
    main()
