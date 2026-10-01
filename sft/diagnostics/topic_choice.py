"""Experiment: state-erasure

Inspect which fact a wake state selects as its next topic."""

from __future__ import annotations

import argparse
import sys

OPENERS = (
    "{user} What is the code for the",
    "{user} Can you provide me with the code for the",
    "{asst} The code for the",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", required=True, help="a dream cache (single or set)")
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--rebuild-state",
        action="store_true",
        help="Re-run the cache's wake transcript under THIS adapter instead of "
        "using the state the cache was built with. Required whenever the "
        "adapter differs from the cache's generator, which is the whole "
        "point when screening corpus variants.",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    args = parser.parse_args()

    import importlib
    import os

    from models.common import build_tokenizer

    import torch
    from dotenv import load_dotenv

    from adapters.lora import apply_lora
    from experiments.dreams.cache import load_dream_cache
    from experiments.dreams.cli import load_init_adapter
    from experiments.dreams.generation import state_to
    from experiments.dreams.types import CachedDream, DreamCache, DreamSetCache
    from experiments.facts import Fact
    from progress import ts

    load_dotenv()
    model_mod = importlib.import_module(f"models.{os.getenv('MODEL_NAME', 'mamba2_780m')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(model_mod)
    model = model_mod.load_inference(str(device))
    apply_lora(
        model,
        model_mod.TARGET_LORA_MODULES,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.0,
    )
    model.to(device)

    main_mod = sys.modules["__main__"]
    for name, value in (
        ("DreamSetCache", DreamSetCache),
        ("DreamCache", DreamCache),
        ("CachedDream", CachedDream),
        ("Fact", Fact),
    ):
        if not hasattr(main_mod, name):
            setattr(main_mod, name, value)

    load_init_adapter(model, args.init_adapter, args.lora_rank, args.lora_alpha)
    model.eval()

    cache = load_dream_cache(args.cache)
    facts = cache.fact_list
    state = cache.wake_state
    if args.rebuild_state:
        from experiments.inference import run_chunks

        transcript = torch.tensor([cache.transcript_ids], device=device)
        chunk = getattr(
            importlib.import_module(f"models.{os.getenv('MODEL_NAME', 'mamba2_780m')}.train_hooks"),
            "DEFAULT_CHUNK_LEN",
            512,
        )
        state = run_chunks(model, transcript, None, chunk, "wake", keep_logits=False)[1]
        print(
            f"[{ts()}] wake state rebuilt under this adapter ({transcript.shape[1]} transcript tokens)"
        )
    print(
        f"[{ts()}] {args.cache}: facts " + ", ".join(f"{fact.entity}={fact.code}" for fact in facts)
    )

    def first_id(entity: str) -> int:
        return int(tokenizer(f" {entity}", add_special_tokens=False)["input_ids"][0])

    ids = {fact.entity: first_id(fact.entity) for fact in facts}
    print(
        f"[{ts()}] first-token ids: "
        + ", ".join(f"{entity}={token}" for entity, token in ids.items())
    )

    for template in OPENERS:
        text = template.format(user=model_mod.USER_OPEN, asst=model_mod.ASST_OPEN)
        tokens = torch.tensor(
            [tokenizer(text, add_special_tokens=False)["input_ids"]], device=device
        )
        with torch.no_grad():
            live = state_to(state, device)
            logits, _ = model(tokens, state=live)
        probs = torch.softmax(logits[0, -1].float(), dim=-1)

        share = {entity: float(probs[token]) for entity, token in ids.items()}
        total = sum(share.values())
        print(f"\n[{ts()}] {text!r}")
        print(f"[{ts()}]   P(entity) and share of the four:")
        for entity, probability in sorted(share.items(), key=lambda item: -item[1]):
            bar = "#" * int(60 * (probability / total if total else 0))
            print(
                f"[{ts()}]     {entity:<10} p={probability:.5f}  {100 * probability / total if total else 0:5.1f}%  {bar}"
            )
        print(f"[{ts()}]   the four together hold {100 * total:.2f}% of the mass")
        top = torch.topk(probs, args.top_k)
        print(
            f"[{ts()}]   unrestricted top-{args.top_k}: "
            + ", ".join(
                f"{tokenizer.decode([int(index)])!r}={float(probability):.4f}"
                for probability, index in zip(top.values, top.indices, strict=True)
            )
        )


__all__ = ["OPENERS", "main"]


if __name__ == "__main__":
    main()
