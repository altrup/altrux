"""Dream-fidelity probe: can the neural memory M *generate* a faithful dream
of what it stored, once the short-term SSM state is wiped?

Tests the precondition for the M->weights consolidation sketch
(notes/RESEARCH-20260722-memory-consolidation-landscape.md): prime M on a real
text, wipe the SSM (sleep), then generate under M-primed vs a random-M control
and measure how much the generation reflects the primed text. Contrast (primed
vs random), not raw inspection, is the point -- a wiped SSM + gated injection
otherwise just emits generic backbone priors; the difference is M's signal.

We have only ever *scored* likelihoods from M (probe_recall); this is the
first probe that *generates* from it. Prediction from the gist-vs-verbatim
result: topically steered but factually generic (gist-faithful, fact-lossy).

Box tool (~17 GB, same class as probe_recall): priming the 2.7B + the memory
write's transient working set doesn't fit the 8 GB local card. Priming runs at
the checkpoint's trained memory-window (the fused kernel path engages on CUDA);
generation drops to window 1 because single-token autoregressive steps can't
close a larger window.

Usage (from sft/, env vars as in the Makefile):
    make dream-fidelity ARGS="--checkpoint <dir> [--n-probes N] [--prime N] [--gen N]"
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
train_hooks = importlib.import_module(f"models.{MODEL_NAME}.train_hooks")
model_mod = importlib.import_module(f"models.{MODEL_NAME}")
mmod = importlib.import_module(f"models.{MODEL_NAME}.model")
pr = importlib.import_module("probe_recall")


def load_trainable(model, ckpt: Path) -> None:
    state = torch.load(ckpt / "trainable.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if len(state) - len(result.unexpected_keys) == 0:
        raise RuntimeError("checkpoint keys don't match model structure")


@torch.no_grad()
def generate(model, state, n_tokens: int, first_token: torch.Tensor, temperature: float, top_p: float, label: str) -> torch.Tensor:
    """Autoregressive generation from `state` (already primed+wiped), one token
    per step (memory-window 1). Nucleus sampling per row; temperature 0 = greedy.
    Returns (n_probes, n_tokens) generated ids."""
    out = []
    tok = first_token
    for i in range(n_tokens):
        logits, state = model(tok, state=state)
        state = state.detach()
        nl = logits[:, -1].float()
        if temperature <= 0:
            tok = nl.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(nl / temperature, dim=-1)
            sp, si = torch.sort(probs, descending=True, dim=-1)
            drop = (sp.cumsum(-1) - sp) > top_p
            sp[drop] = 0.0
            probs = torch.zeros_like(probs).scatter_(-1, si, sp)
            probs /= probs.sum(-1, keepdim=True)
            tok = torch.multinomial(probs, num_samples=1)
        out.append(tok)
        print(f"\r  {label}: gen {i + 1}/{n_tokens}", end="", flush=True)
    print()
    return torch.cat(out, dim=1)


def overlap_with_prime(gen: torch.Tensor, prime: torch.Tensor, common: set[int]) -> float:
    """Jaccard overlap of content-token *types* between one generation and its
    prime, excluding the corpus-common tokens in `common`."""
    g = {t for t in gen.tolist() if t not in common}
    p = {t for t in prime.tolist() if t not in common}
    if not g or not p:
        return 0.0
    return len(g & p) / len(g | p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir with trainable.pt")
    parser.add_argument("--data", default="data/train_memory_longalign.pt", help="Prepared dataset (default: %(default)s)")
    parser.add_argument("--n-probes", type=int, default=4)
    parser.add_argument("--prime", type=int, default=2048, help="Priming prefix tokens (default: %(default)s)")
    parser.add_argument("--gen", type=int, default=128, help="Tokens to generate (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature; 0 = greedy (default: %(default)s)")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Memory writes need no autograd graph for a read-only probe (matches
    # probe_recall's eval path).
    orig_write = mmod._NeuralMemory.write
    mmod._NeuralMemory.write = lambda self, ks, vs, et, th, al, create_graph=True: orig_write(
        self, ks, vs, et, th, al, create_graph=False
    )

    ckpt = Path(args.checkpoint)
    lora_cfg = json.loads((ckpt / "lora_config.json").read_text())
    model, _ = train_hooks.setup_training(device, lora_cfg["rank"], lora_cfg["alpha"], 0.0)
    load_trainable(model, ckpt)
    model.eval()
    model.set_beta_anneal(10**9)

    from models.common import build_tokenizer

    tokenizer = build_tokenizer(model_mod)
    user_id = tokenizer.convert_tokens_to_ids(model_mod.USER_OPEN)
    asst_id = tokenizer.convert_tokens_to_ids(model_mod.ASST_OPEN)

    import random

    rng = random.Random(args.seed)
    prime_len = args.prime - args.prime % pr.CHUNK_LEN
    prefix, _, _ = pr.build_gist_rows(args.data, user_id, asst_id, args.n_probes, prime_len, pr.CHUNK_LEN, rng)
    prefix = prefix.to(device)
    prime_window = lora_cfg.get("memory_window", 1)
    print(
        f"priming {args.n_probes} rows x {prime_len} tokens from {args.data} "
        f"(window {prime_window}); generating {args.gen} each, temp {args.temperature}"
    )

    # Corpus-common token set to discount when scoring overlap (the top ~200
    # most frequent ids across all primes ~ stopwords/punctuation/markup).
    counts: dict[int, int] = {}
    for row in prefix.cpu().tolist():
        for t in row:
            counts[t] = counts.get(t, 0) + 1
    common = {t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:200]}

    mem_dtype = model.front_end.q_proj.weight.dtype
    seed_tok = prefix[:, -1:].clone()
    torch.manual_seed(args.seed)
    with torch.no_grad():
        model.set_memory_window(prime_window)
        _, primed_state = pr.run_chunks(model, prefix, None, "prime", keep_logits=False)

        model.set_memory_window(1)  # single-token generation needs window 1

        m_state = pr.clone_state(mmod, primed_state)
        for b in range(args.n_probes):
            model.sleep_slot(m_state, b)
        gen_primed = generate(model, m_state, args.gen, seed_tok, args.temperature, args.top_p, "M-primed")

        # Control: same wiped SSM, memory replaced with a fresh random init.
        r_state = pr.clone_state(mmod, primed_state)
        for b in range(args.n_probes):
            model.sleep_slot(r_state, b)
        r_state.neural_memory = model.front_end.init_memory(args.n_probes, device, mem_dtype)
        gen_random = generate(model, r_state, args.gen, seed_tok, args.temperature, args.top_p, "M-random")

    print("\n=== overlap with priming text (Jaccard, corpus-common tokens excluded) ===")
    ov_p = [overlap_with_prime(gen_primed[b].cpu(), prefix[b].cpu(), common) for b in range(args.n_probes)]
    ov_r = [overlap_with_prime(gen_random[b].cpu(), prefix[b].cpu(), common) for b in range(args.n_probes)]
    for b in range(args.n_probes):
        print(f"  row {b}: M-primed {ov_p[b]:.3f}  vs  M-random {ov_r[b]:.3f}   (delta {ov_p[b] - ov_r[b]:+.3f})")
    mp, mr = sum(ov_p) / len(ov_p), sum(ov_r) / len(ov_r)
    print(f"  mean:  M-primed {mp:.3f}  vs  M-random {mr:.3f}   (delta {mp - mr:+.3f})")
    print("  (positive delta = M steers generation toward the primed content)")

    print("\n=== decoded samples ===")
    for b in range(min(args.n_probes, 2)):
        print(f"\n row {b}")
        print(f"  PRIME tail : ...{tokenizer.decode(prefix[b, -60:].cpu())!r}")
        print(f"  M-primed   : {tokenizer.decode(gen_primed[b].cpu())!r}")
        print(f"  M-random   : {tokenizer.decode(gen_random[b].cpu())!r}")


if __name__ == "__main__":
    main()
