"""Dream treatment operations: replay, drain, counterfactual, and SFT."""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from experiments.dreams.generation import copy_state, rehearsal_fraction, sample_next
from experiments.dreams.types import CachedDream, Dream, DreamCache
from experiments.erasure.operators import (
    deflate,
    erase_subspace,
    erase_subspace_scaled,
    rank1_erase,
    sigma_gammas,
    state_top_dirs,
)
from experiments.erasure.probe import group_by_layer
from experiments.inference import kl_loss, replay_step
from progress import fmt_duration, ts

if TYPE_CHECKING:
    import torch


GAMMA = 1.0
DEFLATE_K = 1
CONE_SKIP = 0.05
ERASE_OPS = ("raw", "deflated")
ERASE_OP = "deflated"
PRINT_EVERY = 16
SPINE_BLOCK = 32
CF_BATCH = 128


def erase_ssm(ssm_state: torch.Tensor, c: torch.Tensor, gamma: float = GAMMA,
              k: int = DEFLATE_K, op: str = ERASE_OP):
    import torch

    if op not in ERASE_OPS:
        raise ValueError(f"unknown erase op {op!r}; expected one of {ERASE_OPS}")
    c = c.to(ssm_state.device)
    if op == "raw":
        return rank1_erase(ssm_state, c, gamma), 0
    direction = deflate(c, state_top_dirs(ssm_state, k))
    skip = direction.float().norm(dim=-1) < CONE_SKIP * c.float().norm(dim=-1)
    if bool(skip.all()):
        return ssm_state, int(skip.sum())
    erased = rank1_erase(ssm_state, direction, gamma)
    if bool(skip.any()):
        erased = torch.where(skip.view(-1, *([1] * (ssm_state.dim() - 1))), ssm_state, erased)
    return erased, int(skip.sum())


def erase_state(state, queries: Sequence[torch.Tensor], gamma: float = GAMMA,
                k: int = DEFLATE_K, op: str = ERASE_OP) -> int:
    skipped = 0
    for i, query in enumerate(queries):
        state.ssm_states[i], hit = erase_ssm(state.ssm_states[i], query, gamma, k, op)
        skipped += hit
    return skipped


def make_erase_hook(erase_op: str):
    skipped = 0

    def hook(layer_idx: int, ssm_state, c):
        nonlocal skipped
        erased, was_skipped = erase_ssm(ssm_state, c, op=erase_op)
        skipped += was_skipped
        return erased

    return hook, lambda: skipped


def erased_start_scaled(wake_state, bases: Sequence[torch.Tensor], spectra: Sequence[Sequence[float]]):
    state = copy_state(wake_state)
    for layer, basis in enumerate(bases):
        if basis.numel() == 0:
            continue
        state.ssm_states[layer] = erase_subspace_scaled(
            state.ssm_states[layer], basis, sigma_gammas(spectra[layer], basis.shape[0]))
    return state


def erased_start(wake_state, bases: Sequence[torch.Tensor]):
    state = copy_state(wake_state)
    for layer, basis in enumerate(bases):
        state.ssm_states[layer] = erase_subspace(state.ssm_states[layer], basis)
    return state


def dream_from_cached(cached: CachedDream | DreamCache, device) -> Dream:
    import torch

    return Dream(tokens=torch.tensor([cached.dream_ids], dtype=torch.long, device=device),
                 logits=cached.teacher_logits, queries=cached.queries, final_state=None,
                 token_texts=cached.token_texts, skipped_cone=0, cue_flags=cached.cue_flags,
                 stop_reason=cached.stop_reason, prefix_len=cached.prefix_len)


def target_keep_mask(cue_flags: Sequence[bool]) -> list[bool]:
    return [index + 1 >= len(cue_flags) or not cue_flags[index + 1]
            for index in range(len(cue_flags))]


def scored_keep(cue_flags: Sequence[bool], prefix_len: int) -> list[bool]:
    masked = [cue or index < prefix_len for index, cue in enumerate(cue_flags)]
    return target_keep_mask(masked)


def distill_replay(model, opt, dream: Dream, steps: int, chunk_len: int, kl_temp: float, on_step,
                   fresh_state: bool = False, keep: Sequence[bool] | None = None, ce: bool = False,
                   init_state=None) -> int:
    import torch.nn.functional as F

    length = dream.tokens.shape[1]
    n_chunks = (length + chunk_len - 1) // chunk_len
    state = None
    tokens = 0
    for step in range(steps):
        chunk, reset = replay_step(step, n_chunks, fresh_state)
        if reset:
            state = None if init_state is None else copy_state(init_state)
        lo, hi = chunk * chunk_len, min((chunk + 1) * chunk_len, length)
        scored = [t for t in range(lo, hi) if (keep is None or keep[t]) and not (ce and t + 1 >= length)]
        if not scored:
            on_step(step, 0.0)
            continue
        logits, state = model(dream.tokens[:, lo:hi], state=state)
        state = state.detach()
        sel = [t - lo for t in scored]
        if ce:
            loss = F.cross_entropy(logits[0, sel].float(), dream.tokens[0, [t + 1 for t in scored]])
        else:
            loss = kl_loss(dream.logits[scored].unsqueeze(0).to(logits.device), logits[:, sel], kl_temp)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens += len(scored)
        on_step(step, loss.item())
    return tokens


def distill_dream_set(model, opt, dreams: Sequence[CachedDream], wake_state, variant: str | None,
                      epochs: int, kl_temp: float, on_step, on_boundary, sigma_scaled: bool = False) -> int:
    tokens = step = 0
    for epoch in range(epochs):
        for i, cached in enumerate(dreams):
            dream = dream_from_cached(cached, wake_state.ssm_states[0].device)
            if variant is None:
                start = None
            elif sigma_scaled:
                start = erased_start_scaled(wake_state, cached.bases[variant], cached.spectra)
            else:
                start = erased_start(wake_state, cached.bases[variant])
            tokens += distill_replay(
                model, opt, dream, 1, dream.tokens.shape[1], kl_temp,
                lambda s, loss, base=step: on_step(base + s, loss),
                keep=scored_keep(cached.cue_flags, cached.prefix_len), init_state=start)
            step += 1
            on_boundary(i, epoch, step)
    return tokens


def distill_counterfactual(model, opt, dream: Dream, wake_state, steps: int, kl_temp: float, accum: int,
                           in_place: bool, deep: bool, on_step, keep: Sequence[bool] | None = None,
                           erase_op: str = ERASE_OP):
    import torch

    if deep and in_place:
        raise ValueError("deep gradients are measured on B2 only -- deep-B1 is structurally confounded (sec 6)")
    hook, skipped = make_erase_hook(erase_op)
    length = dream.tokens.shape[1]
    step = tokens = 0
    state = None
    while step < steps:
        state = copy_state(wake_state)
        losses: list[torch.Tensor] = []
        for t in range(length):
            if step >= steps:
                break
            token = dream.tokens[:, t:t + 1]
            masked = keep is not None and not keep[t]
            target_state = state if in_place else copy_state(state)
            model.erase_hook = hook
            try:
                if masked:
                    with torch.no_grad():
                        _, advanced = model(token, state=target_state)
                else:
                    logits, advanced = model(token, state=target_state)
            finally:
                model.erase_hook = None
            if not masked:
                loss = kl_loss(dream.logits[t].view(1, 1, -1).to(logits.device), logits, kl_temp)
                if deep:
                    losses.append(loss)
                else:
                    (loss / accum).backward()
                    if (step + 1) % accum == 0:
                        opt.step(); opt.zero_grad(set_to_none=True)
                    on_step(step, loss.item())
                tokens += 1; step += 1
            if in_place:
                state = advanced if deep else advanced.detach()
            elif deep:
                _, state = model(token, state=state)
            else:
                with torch.no_grad():
                    _, state = model(token, state=state)
                state = state.detach()
        if deep and losses:
            total = torch.stack(losses).sum()
            total.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            on_step(step - 1, total.item() / len(losses)); state = state.detach()
    print(f"\n[{ts()}]  near-cone erases skipped: {skipped()} of "
          f"{tokens * max(1, len(getattr(model, 'layers', [1])))}")
    return tokens, state


def _state_layers(state, attr: str) -> list:
    return list(getattr(state, attr))


def _state_attrs(state) -> tuple[str, ...]:
    return tuple(attr for attr in ("conv_states", "ssm_states") if hasattr(state, attr))


def spine_states(model, tokens: torch.Tensor, wake_state, block: int) -> dict[str, list[torch.Tensor]]:
    import torch

    attrs = _state_attrs(wake_state)
    length = tokens.shape[1]
    block = max(1, min(block, length))
    n_blocks = (length + block - 1) // block
    padded = tokens if n_blocks * block == length else torch.cat(
        [tokens, tokens.new_zeros(1, n_blocks * block - length)], dim=1)
    starts = []
    state = copy_state(wake_state)
    for index in range(n_blocks):
        starts.append({attr: [tensor.clone() for tensor in _state_layers(state, attr)] for attr in attrs})
        _, state = model(padded[:, index * block:(index + 1) * block], state=state)
    batched = copy.copy(wake_state)
    for attr in attrs:
        setattr(batched, attr, [torch.cat([start[attr][i] for start in starts], dim=0)
                                for i in range(len(starts[0][attr]))])
    block_tokens = padded.view(n_blocks, block)
    spine = {attr: [tensor.new_empty((n_blocks, block, *tensor.shape[1:]))
                    for tensor in _state_layers(batched, attr)] for attr in attrs}
    for position in range(block):
        for attr in attrs:
            for i, tensor in enumerate(_state_layers(batched, attr)):
                spine[attr][i][:, position] = tensor
        _, batched = model(block_tokens[:, position:position + 1], state=batched)
    return {attr: [tensor.reshape(-1, *tensor.shape[2:])[:length] for tensor in tensors]
            for attr, tensors in spine.items()}


def fused_pass(model, dream: Dream, wake_state, spine: dict[str, list[torch.Tensor]], scored: Sequence[int],
               kl_temp: float, erase_op: str, cf_batch: int, backward: bool, retain: bool = False):
    import torch

    hook, skipped = make_erase_hook(erase_op)
    total = 0.0
    for lo in range(0, len(scored), cf_batch):
        selected = list(scored[lo:lo + cf_batch])
        index = torch.tensor(selected, dtype=torch.long, device=dream.tokens.device)
        cf = copy.copy(wake_state)
        for attr, layers in spine.items():
            setattr(cf, attr, [tensor[index] for tensor in layers])
        model.erase_hook = hook
        try:
            logits, _ = model(dream.tokens[0, index].view(-1, 1), state=cf)
        finally:
            model.erase_hook = None
        loss = kl_loss(dream.logits[selected].unsqueeze(1).to(logits.device), logits, kl_temp) * len(selected)
        if backward:
            loss.backward(retain_graph=retain)
        total += float(loss.detach())
    return total, skipped()


def distill_fused(model, opt, dream: Dream, wake_state, steps: int, kl_temp: float, on_step,
                  keep: Sequence[bool] | None = None, erase_op: str = ERASE_OP, deep: bool = False,
                  block: int = SPINE_BLOCK, cf_batch: int = CF_BATCH,
                  frozen_spine: dict[str, list[torch.Tensor]] | None = None, check=None) -> int:
    import torch

    scored = [t for t in range(dream.tokens.shape[1]) if keep is None or keep[t]]
    tokens = 0
    for step in range(steps):
        if frozen_spine is not None:
            spine = frozen_spine
        elif deep:
            spine = spine_states(model, dream.tokens, wake_state, block)
        else:
            with torch.no_grad():
                spine = spine_states(model, dream.tokens, wake_state, block)
        loss, _ = fused_pass(model, dream, wake_state, spine, scored, kl_temp, erase_op, cf_batch,
                             backward=True, retain=deep)
        if step == 0 and frozen_spine is not None and check is not None:
            device = frozen_spine["ssm_states"][0].device
            parked = {attr: [tensor.to("cpu") for tensor in layers]
                      for attr, layers in frozen_spine.items()}
            for layers in frozen_spine.values():
                layers.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with torch.no_grad():
                live = spine_states(model, dream.tokens, wake_state, block)
                reference, _ = fused_pass(model, dream, wake_state, live, scored, kl_temp,
                                         erase_op, cf_batch, backward=False)
            del live
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for attr, layers in parked.items():
                frozen_spine[attr].extend(tensor.to(device) for tensor in layers)
            equivalent = abs(loss - reference) <= 1e-4 * max(1.0, abs(reference))
            check({"pass": 1, "b3_loss": loss, "b2_loss": reference,
                   "abs_diff": abs(loss - reference), "equivalent": equivalent})
            print(f"[{ts()}]  pass-1 equivalence B3-fused vs B2-fused-detached: "
                  f"{'OK' if equivalent else 'FAILED'} ({loss:.6f} vs {reference:.6f})", flush=True)
        opt.step(); opt.zero_grad(set_to_none=True)
        tokens += len(scored); on_step(step, loss / max(1, len(scored)))
    return tokens


def distill_live(model, opt, wake_state, seed_ids: torch.Tensor, n_tokens: int, temperature: float,
                 kl_temp: float, accum: int, decode_token, needles: Sequence[str], on_step,
                 erase_op: str = ERASE_OP) -> tuple[object, Dream]:
    import torch

    state = copy.deepcopy(wake_state); ids = [int(i) for i in seed_ids[0].tolist()]
    texts: list[str] = []; logits_cache: list[torch.Tensor] = []; skipped = 0; started = time.time()
    for t in range(n_tokens):
        token = torch.tensor([[ids[t]]], dtype=torch.long, device=seed_ids.device)
        model.c_capture = []
        with torch.no_grad():
            stored, _ = model(token, state=copy.deepcopy(state))
        per_layer = group_by_layer(model.c_capture, len(model.layers))[0]; model.c_capture = None
        skipped += erase_state(state, per_layer, op=erase_op)
        logits, state = model(token, state=state)
        loss = kl_loss(stored.detach(), logits, kl_temp) / accum; loss.backward()
        if (t + 1) % accum == 0: opt.step(); opt.zero_grad(set_to_none=True)
        state = state.detach(); on_step(t, loss.item() * accum)
        logits_cache.append(stored[0, -1].float().cpu()); texts.append(decode_token(ids[t]))
        if t + 1 >= len(ids): ids.append(int(sample_next(stored[:, -1], temperature).item()))
        if (t + 1) % PRINT_EVERY == 0 or t + 1 == n_tokens:
            frac, _ = rehearsal_fraction(texts, needles); rate = (t + 1) / (time.time() - started)
            print(f"[{ts()}]  live dream {t + 1}/{n_tokens} rehearsal {frac:.2f} {rate:.1f} tok/s "
                  f"ETA {fmt_duration((n_tokens - t - 1) / rate)} | {''.join(texts[-PRINT_EVERY:])!r}", flush=True)
    return state, Dream(tokens=torch.tensor([ids[:n_tokens]], dtype=torch.long, device=seed_ids.device),
                        logits=torch.stack(logits_cache), queries=[], final_state=state,
                        token_texts=texts, skipped_cone=skipped)


def sft_steps(token_count: int, chunk_len: int) -> int:
    return (max(0, token_count - 1) + chunk_len - 1) // chunk_len


def distill_sft(model, opt, ids: torch.Tensor, steps: int, chunk_len: int, on_step) -> int:
    import torch.nn.functional as F

    length = ids.shape[1] - 1; n_chunks = (length + chunk_len - 1) // chunk_len
    state = None; tokens = 0
    for step in range(steps):
        chunk = step % n_chunks
        if chunk == 0: state = None
        lo, hi = chunk * chunk_len, min((chunk + 1) * chunk_len, length)
        logits, state = model(ids[:, lo:hi], state=state); state = state.detach()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), ids[0, lo + 1:hi + 1])
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        tokens += hi - lo; on_step(step, loss.item())
    return tokens


__all__ = ["distill_counterfactual", "distill_dream_set", "distill_fused", "distill_live",
           "distill_replay", "distill_sft", "dream_from_cached", "erase_state", "erase_ssm",
           "erased_start", "erased_start_scaled", "fused_pass", "make_erase_hook", "sft_steps",
           "scored_keep", "spine_states", "target_keep_mask"]
