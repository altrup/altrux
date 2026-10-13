"""Replay one dream cycle's distillation locally from the saved wake transcript and dreams.

Rebuilds the open state from wake.json, recomputes each dream's teacher logits from
that state, reports the KL and LoRA gradient norm of every dream at fixed weights,
runs the cycle's optimizer steps as the runner did, and prints the wake frame's
log-probability after the dream prompt before and after.

    uv run --no-sync python -u -m experiments.lama_ckl.diagnostics.replay_cycle \
        --run-dir ../.cache/lama_ckl/runs/dream-seed-42 --epoch 1 --cycle 0
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import models  # noqa: F401 -- ROCm stubs before mamba_ssm

import torch

from experiments.dream_distillation import scored_keep
from experiments.dream_generation import copy_state
from experiments.dream_types import CachedDream
from experiments.inference import kl_loss
from experiments.lama_ckl.protocol import WAKE_FRAME, _encode
from experiments.lama_ckl.runner import KL_TEMPERATURE, LEARNING_RATE
from progress import ts

CHUNK = 128


@torch.no_grad()
def run_state(model, ids: list[int], state, device):
    """Feed ids from state in chunks; returns (logits over all positions, state)."""
    out = []
    for lo in range(0, len(ids), CHUNK):
        tokens = torch.tensor([ids[lo : lo + CHUNK]], dtype=torch.long, device=device)
        logits, state = model(tokens, state=state)
        out.append(logits[0].float().cpu())
    return torch.cat(out), state


def frame_logprob(model, prefix: list[int], frame: list[int], state, device) -> float:
    logits, _ = run_state(
        model, prefix + frame, None if state is None else copy_state(state), device
    )
    logp = torch.log_softmax(logits, dim=-1)
    positions = range(len(prefix) - 1, len(prefix) + len(frame) - 1)
    return sum(logp[p, frame[i]].item() for i, p in enumerate(positions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--model-name", default="mamba2_2_7b")
    args = parser.parse_args()

    device = torch.device("cuda")
    adapter = args.run_dir / f"epoch-{args.epoch - 1:02d}"
    cycle_dir = args.run_dir / f"epoch-{args.epoch:02d}" / f"cycle-{args.cycle:02d}"
    wake = json.loads((cycle_dir / "wake.json").read_text())
    dreams_json = json.loads((cycle_dir / "dreams.json").read_text())

    model_mod = importlib.import_module(f"models.{args.model_name}")
    hooks = importlib.import_module(f"models.{args.model_name}.train_hooks")
    from models.common import build_tokenizer

    from training.checkpoints import load_checkpoint

    config = json.loads((adapter / "lora_config.json").read_text())
    model, trainable = hooks.setup_training(
        device, int(config["rank"]), float(config["alpha"]), 0.0
    )
    load_checkpoint(model, adapter)
    model.set_grad_checkpoint(False)
    tokenizer = build_tokenizer(model_mod)
    print(f"[{ts()}] loaded {adapter}", flush=True)

    transcript = [int(t) for t in wake["transcript_token_ids"]]
    _, open_state = run_state(model, transcript, None, device)
    print(f"[{ts()}] open state from {len(transcript)} wake tokens", flush=True)

    cached: list[CachedDream] = []
    for i, dream in enumerate(dreams_json["dreams"]):
        ids = [int(t) for t in dream["token_ids"]]
        logits, _ = run_state(model, ids, copy_state(open_state), device)
        cached.append(
            CachedDream(
                dream_ids=ids,
                token_texts=[],
                teacher_logits=logits,
                cue_flags=[False] * len(ids),
                prefix_len=int(dream["prefix_tokens"]),
                stop_reason=dream["stop_reason"],
                divergence=[],
                gate_positions=[],
                queries=[],
                spectra=[],
                ranks={},
                bases={},
            )
        )
        print(
            f"[{ts()}] teacher logits dream {i + 1}/{len(dreams_json['dreams'])} ({len(ids)} tokens)",
            flush=True,
        )

    # The frame as the wake tokenizes it (" Remember" is one token); the dream prompt is EOC.
    prefix = _encode(tokenizer, model_mod.EOC) + [int(wake["turns"][0]["prompt_token_ids"][0])]
    frame = [int(t) for t in wake["turns"][0]["prompt_token_ids"][1:]]
    frame = frame[: len(_encode(tokenizer, " " + WAKE_FRAME.rstrip()))]
    print(f"[{ts()}] probe tokens {[tokenizer.decode([t]) for t in prefix + frame]}", flush=True)

    def probe(label: str) -> None:
        empty = frame_logprob(model, prefix, frame, None, device)
        opened = frame_logprob(model, prefix, frame, open_state, device)
        print(
            f"[{ts()}] {label}: log P(frame | EOC [USER]) empty-state {empty:.2f} "
            f"wake-state {opened:.2f}",
            flush=True,
        )

    probe("before")

    def backward_chunked(dream: CachedDream) -> tuple[float, int]:
        """KL over the scored positions, backward per 4-token chunk, state detached between.

        ponytail: truncated backprop (no gradient through the carried state across chunks)
        so a 512-token dream fits beside the weights in 8 GB; the box runs the full sequence.
        """
        keep = scored_keep(dream.cue_flags, dream.prefix_len)
        total = sum(keep)
        tokens = torch.tensor([dream.dream_ids], dtype=torch.long, device=device)
        state = None
        loss_sum = 0.0
        for lo in range(0, len(dream.dream_ids), 4):
            hi = min(lo + 4, len(dream.dream_ids))
            scored = [t for t in range(lo, hi) if keep[t]]
            logits, state = model(tokens[:, lo:hi], state=state)
            state = state.detach()
            if not scored:
                continue
            sel = [t - lo for t in scored]
            loss = kl_loss(
                dream.teacher_logits[scored].unsqueeze(0).to(device), logits[:, sel], KL_TEMPERATURE
            )
            (loss * len(scored) / total).backward()
            loss_sum += loss.item() * len(scored) / total
        return loss_sum, total

    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    for i, dream in enumerate(cached):
        loss, scored = backward_chunked(dream)
        norm = torch.sqrt(
            sum((p.grad.float() ** 2).sum() for p in trainable if p.grad is not None)
        ).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        text = dreams_json["dreams"][i]["text"][:70].replace("\n", " ")
        print(
            f"[{ts()}] step {i + 1}/{len(cached)}: KL {loss:.4f} grad-norm {norm:.4f} "
            f"tokens {scored} | {text}",
            flush=True,
        )
    probe("after")
    out = (
        args.run_dir / f"epoch-{args.epoch:02d}" / f"cycle-{args.cycle:02d}" / "replay-trainable.pt"
    )
    torch.save(
        {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}, out
    )
    print(f"[{ts()}] saved post-step trainable tensors to {out}", flush=True)


if __name__ == "__main__":
    main()
