"""Per-token read diagnostic for the memory subsystem (stage-2 groundwork).

Runs a checkpoint over real sequences on the manual path and captures, at
EVERY token: the read magnitude ||o_t|| and surprise (both already computed
per-token by the model), plus the residual stream at a probe layer (default
21, the injection-free floor) and at READ_LAYER (42, the front-end's input).
Reports:

1. Whether ||o_t|| / surprise vary within a sequence (gate-signal viability
   for o_t-gated sparse injection), including where the peaks land in real
   text.
2. How much of the front-end's layer-42 input is linearly recoverable from
   the probe layer (ridge regression res_probe -> res_42, scored in q/k/v
   space) -- the go/no-go for moving the front-end below the injection
   floor. See notes/DISCUSSION-20260722-stage2-readout.md.

Usage (from sft/, env vars as in the Makefile):
    make read-diagnostic ARGS="--ckpt <dir> [--tokens N] [--examples N]"
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "mamba2_2_7b_memory")
hooks = importlib.import_module(f"models.{MODEL_NAME}.train_hooks")
mm = importlib.import_module(f"models.{MODEL_NAME}.model")


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
    return "  ".join(f"p{int(q * 100):<3}{v:.4f}" for q, v in zip(qs, vals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="Checkpoint dir with trainable.pt")
    parser.add_argument("--data", default="data/train_memory_longalign.pt", help="Prepared dataset (default: %(default)s)")
    parser.add_argument("--examples", type=int, default=4, help="How many examples to run (default: %(default)s)")
    parser.add_argument("--tokens", type=int, default=4096, help="Tokens per example (default: %(default)s)")
    parser.add_argument("--chunk-len", type=int, default=48, help="Tokens per forward call (default: %(default)s)")
    parser.add_argument("--probe-layer", type=int, default=21, help="Injection-free layer to test against READ_LAYER (default: %(default)s)")
    parser.add_argument("--save", default=None, help="Optional .pt path to dump raw captures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg = json.loads((Path(args.ckpt) / "lora_config.json").read_text())
    memory_window = cfg.get("memory_window", 1)
    model, _ = hooks.setup_training(device, cfg["rank"], cfg["alpha"], 0.0)
    load_trainable(model, Path(args.ckpt))
    model.set_memory_window(memory_window)
    model.set_beta_anneal(10**9)
    model.eval()

    probe_layer = model.layers[args.probe_layer]
    res_probe: list[torch.Tensor] = []
    res_read: list[torch.Tensor] = []
    o_norms: list[float] = []
    surprises: list[float] = []

    orig_prenorm = model._prenorm

    def prenorm_spy(layer, h, residual):
        out = orig_prenorm(layer, h, residual)
        if layer is probe_layer:
            res_probe.append(out[1].detach()[0].half().cpu())
        return out

    model._prenorm = prenorm_spy

    orig_observe = model.front_end.observe

    def observe_spy(residual: torch.Tensor):
        res_read.append(residual.detach()[0].half().cpu())
        return orig_observe(residual)

    model.front_end.observe = observe_spy

    orig_read, orig_surprise = mm._NeuralMemory.read, mm._NeuralMemory.surprise

    def read_spy(self, q):
        o = orig_read(self, q)
        o_norms.append(o.detach()[0].norm().item())
        return o

    def surprise_spy(self, k, v):
        s = orig_surprise(self, k, v)
        surprises.append(s.detach()[0].item())
        return s

    mm._NeuralMemory.read, mm._NeuralMemory.surprise = read_spy, surprise_spy

    data = torch.load(args.data, map_location="cpu", weights_only=True)
    n_examples = min(args.examples, len(data["ids"]))
    all_ids: list[torch.Tensor] = []
    example_lens: list[int] = []
    for idx in range(n_examples):
        ids = data["ids"][idx][: args.tokens].unsqueeze(0).to(device)
        all_ids.append(ids[0].cpu())
        example_lens.append(ids.shape[1])
        state = hooks.init_state(model, 1, device)
        with torch.no_grad():
            for start in range(0, ids.shape[1], args.chunk_len):
                _, state = model(ids[:, start : start + args.chunk_len], state=state)
                state = state.detach()
                done = min(start + args.chunk_len, ids.shape[1])
                print(f"\r  example {idx}: token {done:>5}/{ids.shape[1]}  o_norm {o_norms[-1]:.2f}", end="", flush=True)
        print()

    total = sum(example_lens)
    assert len(o_norms) == len(surprises) == len(res_probe) == len(res_read) == total, (
        f"capture mismatch: {len(o_norms)}/{len(surprises)}/{len(res_probe)}/{len(res_read)} vs {total} tokens"
    )

    o_t = torch.tensor(o_norms)
    sup = torch.tensor(surprises)
    if args.save:
        torch.save(
            {
                "o_norm": o_t, "surprise": sup, "example_lens": example_lens, "ids": all_ids,
                "res_probe": torch.stack(res_probe), "res_read": torch.stack(res_read),
                "probe_layer": args.probe_layer, "ckpt": args.ckpt,
            },
            args.save,
        )
        print(f"raw captures saved to {args.save}")

    # --- 1. gate-signal viability -------------------------------------------
    print(f"\n=== per-token read signal over {total} tokens ({n_examples} examples) ===")
    print(f"||o_t||   {percentiles(o_t)}")
    print(f"          mean {o_t.mean():.3f}  std {o_t.std():.3f}  CV {o_t.std() / o_t.mean():.3f}")
    print(f"surprise  {percentiles(sup)}")
    print(f"          mean {sup.mean():.3f}  std {sup.std():.3f}  CV {sup.std() / sup.mean():.3f}")
    corr = torch.corrcoef(torch.stack([o_t, sup]))[0, 1]
    print(f"corr(||o_t||, surprise) {corr:.3f}")

    pos = torch.arange(total)
    offsets = []
    for n in example_lens:
        offsets.append(pos[:n] % memory_window)
        pos = pos[n:]
    win_pos = torch.cat(offsets)
    by_pos = [o_t[win_pos == p].mean().item() for p in range(memory_window)]
    print(f"mean ||o_t|| by window position 0..{memory_window - 1}: " + "  ".join(f"{v:.2f}" for v in by_pos))

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        model_mod = importlib.import_module(f"models.{MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(model_mod.TOKENIZER_ID)
    except Exception as e:  # tokenizer is decoration; the numbers above stand alone
        print(f"(skipping peak decode, tokenizer unavailable: {e})")
    if tokenizer is not None:
        flat_ids = torch.cat(all_ids)
        # Skip each example's first 256 tokens: M ramps from cold there.
        warm = torch.ones(total, dtype=torch.bool)
        base = 0
        for n in example_lens:
            warm[base : base + min(256, n)] = False
            base += n
        warm_idx = torch.nonzero(warm).squeeze(-1)
        top = warm_idx[o_t[warm_idx].argsort(descending=True)[:8]]
        print("\ntop ||o_t|| tokens in context (marked with >><<):")
        for t in top.tolist():
            lo, hi = max(0, t - 12), min(total, t + 4)
            ctx = tokenizer.decode(flat_ids[lo:t]) + " >>" + tokenizer.decode(flat_ids[t : t + 1]) + "<< " + tokenizer.decode(flat_ids[t + 1 : hi])
            print(f"  [{t}] o_norm {o_t[t]:.2f}: {ctx!r}")

    # --- 2. probe-layer -> READ_LAYER linear recoverability -----------------
    print(f"\n=== ridge regression: layer {args.probe_layer} -> layer {mm.READ_LAYER} residual ===")
    X = torch.stack(res_probe).float()
    Y = torch.stack(res_read).float()
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(0))
    n_test = max(1, total // 5)
    test_i, train_i = perm[:n_test], perm[n_test:]
    Xm, Ym = X[train_i].mean(0), Y[train_i].mean(0)
    Xc, Yc = X[train_i] - Xm, Y[train_i] - Ym
    gram = Xc.T @ Xc
    lam = 1e-3 * gram.diagonal().mean()
    W = torch.linalg.solve(gram + lam * torch.eye(gram.shape[0]), Xc.T @ Yc)
    Y_hat = (X[test_i] - Xm) @ W + Ym
    Y_true = Y[test_i]
    ss_res = (Y_true - Y_hat).pow(2).sum()
    ss_tot = (Y_true - Y[train_i].mean(0)).pow(2).sum()
    print(f"held-out residual R^2: {1 - ss_res / ss_tot:.4f}  ({len(train_i)} train / {n_test} test tokens)")

    fe = model.front_end
    for name in ("q_proj", "k_proj", "v_proj"):
        proj = getattr(fe, name).to("cpu").float()
        with torch.no_grad():
            true_p = mm._rms_normalize(proj(Y_true))
            pred_p = mm._rms_normalize(proj(Y_hat))
            shuf_p = pred_p[torch.randperm(n_test, generator=torch.Generator().manual_seed(1))]
        cos = torch.nn.functional.cosine_similarity(true_p, pred_p, dim=-1)
        cos_shuf = torch.nn.functional.cosine_similarity(true_p, shuf_p, dim=-1)
        # Raw cosines are inflated by the projections' shared mean direction
        # (shuffled control ~0.9); centering isolates the per-token part.
        mu = true_p.mean(0, keepdim=True)
        cos_c = torch.nn.functional.cosine_similarity(true_p - mu, pred_p - mu, dim=-1)
        cos_c_shuf = torch.nn.functional.cosine_similarity(true_p - mu, shuf_p - mu, dim=-1)
        print(f"{name} cosine(true, predicted): {percentiles(cos)}   [shuffled-control mean {cos_shuf.mean():.3f}]")
        print(f"{name}   centered:              {percentiles(cos_c)}   [shuffled-control mean {cos_c_shuf.mean():.3f}]")


if __name__ == "__main__":
    main()
