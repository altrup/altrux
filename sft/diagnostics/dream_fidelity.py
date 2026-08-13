"""Dream-fidelity probe for the trained neural memory."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()


def load_trainable(model, ckpt: Path) -> None:
    state = torch.load(ckpt / "trainable.pt", map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if len(state) - len(result.unexpected_keys) == 0:
        raise RuntimeError("checkpoint keys don't match model structure")


@torch.no_grad()
def generate(model, state, n_tokens: int, first_token: torch.Tensor, temperature: float,
             top_p: float, label: str) -> torch.Tensor:
    """Generate tokens from ``state`` with greedy or nucleus sampling."""
    out = []
    token = first_token
    for index in range(n_tokens):
        logits, state = model(token, state=state)
        state = state.detach()
        next_logits = logits[:, -1].float()
        if temperature <= 0:
            token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            drop = (sorted_probs.cumsum(-1) - sorted_probs) > top_p
            sorted_probs[drop] = 0.0
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
            probs /= probs.sum(-1, keepdim=True)
            token = torch.multinomial(probs, num_samples=1)
        out.append(token)
        print(f"\r  {label}: gen {index + 1}/{n_tokens}", end="", flush=True)
    print()
    return torch.cat(out, dim=1)


def overlap_with_prime(gen: torch.Tensor, prime: torch.Tensor, common: set[int]) -> float:
    """Return Jaccard overlap of non-common content-token types."""
    generated = {token for token in gen.tolist() if token not in common}
    primed = {token for token in prime.tolist() if token not in common}
    if not generated or not primed:
        return 0.0
    return len(generated & primed) / len(generated | primed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir with trainable.pt")
    parser.add_argument("--data", default="data/train_memory_longalign.pt",
                        help="Prepared dataset (default: data/train_memory_longalign.pt)")
    parser.add_argument("--n-probes", type=int, default=4)
    parser.add_argument("--prime", type=int, default=2048, help="Priming prefix tokens (default: %(default)s)")
    parser.add_argument("--gen", type=int, default=128, help="Tokens to generate (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature; 0 = greedy (default: %(default)s)")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    import probe_recall as pr

    model_name = os.getenv("MODEL_NAME", "mamba2_2_7b_memory")
    train_hooks = importlib.import_module(f"models.{model_name}.train_hooks")
    model_mod = importlib.import_module(f"models.{model_name}")
    mmod = importlib.import_module(f"models.{model_name}.model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
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
    prime_len = args.prime - args.prime % pr.CHUNK_LEN
    prefix, _, _ = pr.build_gist_rows(args.data, user_id, asst_id, args.n_probes, prime_len, pr.CHUNK_LEN, random.Random(args.seed))
    prefix = prefix.to(device)
    prime_window = lora_cfg.get("memory_window", 1)
    print(f"priming {args.n_probes} rows x {prime_len} tokens from {args.data} (window {prime_window}); generating {args.gen} each, temp {args.temperature}")

    counts: dict[int, int] = {}
    for row in prefix.cpu().tolist():
        for token in row:
            counts[token] = counts.get(token, 0) + 1
    common = {token for token, _ in sorted(counts.items(), key=lambda item: -item[1])[:200]}

    mem_dtype = model.front_end.q_proj.weight.dtype
    seed_token = prefix[:, -1:].clone()
    torch.manual_seed(args.seed)
    with torch.no_grad():
        model.set_memory_window(prime_window)
        _, primed_state = pr.run_chunks(model, prefix, None, "prime", keep_logits=False)
        model.set_memory_window(1)
        m_state = pr.clone_state(mmod, primed_state)
        for batch in range(args.n_probes):
            model.sleep_slot(m_state, batch)
        gen_primed = generate(model, m_state, args.gen, seed_token, args.temperature, args.top_p, "M-primed")
        r_state = pr.clone_state(mmod, primed_state)
        for batch in range(args.n_probes):
            model.sleep_slot(r_state, batch)
        r_state.neural_memory = model.front_end.init_memory(args.n_probes, device, mem_dtype)
        gen_random = generate(model, r_state, args.gen, seed_token, args.temperature, args.top_p, "M-random")

    print("\n=== overlap with priming text (Jaccard, corpus-common tokens excluded) ===")
    overlap_primed = [overlap_with_prime(gen_primed[batch].cpu(), prefix[batch].cpu(), common) for batch in range(args.n_probes)]
    overlap_random = [overlap_with_prime(gen_random[batch].cpu(), prefix[batch].cpu(), common) for batch in range(args.n_probes)]
    for batch in range(args.n_probes):
        print(f"  row {batch}: M-primed {overlap_primed[batch]:.3f}  vs  M-random {overlap_random[batch]:.3f}   (delta {overlap_primed[batch] - overlap_random[batch]:+.3f})")
    mean_primed = sum(overlap_primed) / len(overlap_primed)
    mean_random = sum(overlap_random) / len(overlap_random)
    print(f"  mean:  M-primed {mean_primed:.3f}  vs  M-random {mean_random:.3f}   (delta {mean_primed - mean_random:+.3f})")
    print("  (positive delta = M steers generation toward the primed content)")

    print("\n=== decoded samples ===")
    for batch in range(min(args.n_probes, 2)):
        print(f"\n row {batch}")
        print(f"  PRIME tail : ...{tokenizer.decode(prefix[batch, -60:].cpu())!r}")
        print(f"  M-primed   : {tokenizer.decode(gen_primed[batch].cpu())!r}")
        print(f"  M-random   : {tokenizer.decode(gen_random[batch].cpu())!r}")


__all__ = ["generate", "load_trainable", "main", "overlap_with_prime"]
