"""Experiment: memory-model

Measure the memory front-end's data-dependent write knobs on real data."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "mamba2_780m")


def load_trainable(model: torch.nn.Module, ckpt: Path) -> None:
    state = torch.load(ckpt / "trainable.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.unexpected_keys)
    print(f"loaded {loaded}/{len(state)} trainable tensors from {ckpt}")
    if loaded == 0:
        raise RuntimeError("checkpoint keys don't match model structure")


def percentiles(t: torch.Tensor) -> str:
    qs = torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    vals = torch.quantile(t.float(), qs)
    return "  ".join(f"p{int(q * 100):<3}{value:.4f}" for q, value in zip(qs, vals))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint dir with trainable.pt (default: fresh init, no checkpoint)",
    )
    parser.add_argument(
        "--data", default="data/train_memory.pt", help="Prepared dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--example-idx", type=int, default=0, help="Which example to run (default: %(default)s)"
    )
    parser.add_argument(
        "--tokens", type=int, default=128, help="How many tokens to run (default: %(default)s)"
    )
    parser.add_argument(
        "--chunk-len", type=int, default=16, help="Tokens per forward call (default: %(default)s)"
    )
    parser.add_argument(
        "--memory-window",
        type=int,
        default=None,
        help="Tokens per memory write (default: checkpoint's lora_config.json value, else 1)",
    )
    args = parser.parse_args()

    importlib.invalidate_caches()
    hooks = importlib.import_module(f"models.{MODEL_NAME}.train_hooks")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    lora_rank, lora_alpha, memory_window = 16, 32.0, 1
    if args.ckpt:
        cfg = json.loads((Path(args.ckpt) / "lora_config.json").read_text())
        lora_rank, lora_alpha = cfg["rank"], cfg["alpha"]
        memory_window = cfg.get("memory_window", 1)
    if args.memory_window is not None:
        memory_window = args.memory_window
    if args.chunk_len % memory_window != 0:
        parser.error(
            f"--chunk-len {args.chunk_len} must be a multiple of the memory window {memory_window}"
        )

    model, _ = hooks.setup_training(device, lora_rank, lora_alpha, 0.0)
    if args.ckpt:
        load_trainable(model, Path(args.ckpt))
    model.set_memory_window(memory_window)
    model.set_beta_anneal(10**9)
    model.eval()

    records: list[dict[str, torch.Tensor]] = []
    front_end = model.front_end
    orig_observe = front_end.observe

    def spy(residual: torch.Tensor):
        out = orig_observe(residual)
        _, _, _, eta, theta, alpha = out
        with torch.no_grad():
            records.append(
                {
                    "res_rms": residual.float().pow(2).mean(dim=-1).sqrt().cpu(),
                    "pre": front_end.knob_proj(residual).float().cpu(),
                    "eta": eta.float().cpu(),
                    "theta": theta.float().cpu(),
                    "alpha": alpha.float().cpu(),
                }
            )
        return out

    front_end.observe = spy
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    ids = data["ids"][args.example_idx][: args.tokens].unsqueeze(0).to(device)
    print(
        f"example {args.example_idx}: running {ids.shape[1]} tokens, memory_window {memory_window}"
    )

    state = hooks.init_state(model, 1, device)
    nm = state.neural_memory
    rms = lambda tensor: tensor.detach().float().pow(2).mean().sqrt().item()
    w1_traj = [rms(nm.w1)]
    surprise_traj: list[float] = []
    with torch.no_grad():
        for start in range(0, ids.shape[1], args.chunk_len):
            chunk = ids[:, start : start + args.chunk_len]
            _, state = model(chunk, state=state)
            state = state.detach()
            nm = state.neural_memory
            w1_traj.append(rms(nm.w1))
            surprise_traj.append(state.last_surprise.mean().item())
            print(
                f"\r  token {min(start + args.chunk_len, ids.shape[1]):>5}/{ids.shape[1]}"
                f"  w1_rms {w1_traj[-1]:.3e}  surprise {surprise_traj[-1]:.4f}",
                end="",
                flush=True,
            )
    print()

    eta = torch.cat([record["eta"] for record in records])
    theta = torch.cat([record["theta"] for record in records])
    alpha = torch.cat([record["alpha"] for record in records])
    pre = torch.cat([record["pre"] for record in records])
    res_rms = torch.cat([record["res_rms"] for record in records])
    print(f"\nknob distributions over {len(records)} tokens:")
    print(f"  eta    {percentiles(eta)}")
    print(f"  theta  {percentiles(theta)}")
    print(f"  alpha  {percentiles(alpha)}")
    print("\nknob_proj pre-activation (per knob: eta, theta, alpha):")
    for index, name in enumerate(("eta", "theta", "alpha")):
        print(f"  {name:<6} {percentiles(pre[:, index])}")
    print(f"\nresidual rms into knob_proj: {percentiles(res_rms)}")
    mean_alpha = alpha.mean().item()
    half_life_windows = math.log(0.5) / math.log(1 - mean_alpha) if 0 < mean_alpha < 1 else math.inf
    print(
        f"\nmean alpha {mean_alpha:.4f} per window -> fast-weight half-life ~{half_life_windows:.0f} windows (~{half_life_windows * memory_window:.0f} tokens)"
    )
    print(
        f"w1 rms trajectory: start {w1_traj[0]:.3e} -> end {w1_traj[-1]:.3e} ({w1_traj[-1] / w1_traj[0]:.3f}x)"
    )
    print(
        f"raw surprise (pooled MSE, unit-rms v): start {surprise_traj[0]:.4f} -> end {surprise_traj[-1]:.4f}"
    )


__all__ = ["MODEL_NAME", "load_trainable", "main", "percentiles"]


if __name__ == "__main__":
    main()
