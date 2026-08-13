"""Which fact does the state actually want to talk about?

The un-cued coverage failure (two of four facts never spontaneously rehearsed)
could be either of two very different things:

  (a) the state cannot serve the missing facts un-cued -- but the in-context
      probe says otherwise: all four return at margin +12 to +16 when asked;
  (b) the model's TOPIC SELECTION is lopsided -- at the position where a dream
      picks which entity to ask about, the distribution over the four names is
      concentrated on one or two.

This reads (b) directly: put the model at the wake state, feed the opener a
dream actually uses, and print the next-token distribution restricted to the
four entity names -- plus the unrestricted top-k, so "none of them, it wants to
talk about something else entirely" is visible rather than hidden by
normalising over a set the model never considered.

    MODEL_NAME=... python topic_choice.py --cache data/dream_set_r_uncued_s1234.pt \
        --init-adapter <ckpt>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# The openers a dream reaches this decision through. The first is the wake
# session's own phrasing; the others are what un-cued dreams were observed
# emitting on their own.
OPENERS = (
    "{user} What is the code for the",
    "{user} Can you provide me with the code for the",
    "{asst} The code for the",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, help="a dream cache (single or set)")
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rebuild-state", action="store_true",
                        help="Re-run the cache's wake transcript under THIS adapter instead of "
                             "using the state the cache was built with. Required whenever the "
                             "adapter differs from the cache's generator, which is the whole "
                             "point when screening corpus variants.")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    args = parser.parse_args()

    import importlib
    import os

    import torch
    from dotenv import load_dotenv

    from progress import ts
    from lora import apply_lora
    from models.common import build_tokenizer

    load_dotenv()
    model_mod = importlib.import_module(f"models.{os.getenv('MODEL_NAME', 'mamba2_780m')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(model_mod)
    model = model_mod.load_inference(str(device))
    apply_lora(model, model_mod.TARGET_LORA_MODULES, rank=args.lora_rank,
               alpha=args.lora_alpha, dropout=0.0)
    model.to(device)

    import dream_sleep
    from dream_sleep import load_dream_cache, state_to

    # The caches were pickled while dream_sleep.py was __main__, so their
    # classes resolve to __main__.DreamSetCache and load ONLY from that script
    # unless the names are re-bound here.
    main_mod = sys.modules["__main__"]
    for name in ("DreamSetCache", "DreamCache", "CachedDream", "MixerState", "Fact"):
        if not hasattr(main_mod, name) and hasattr(dream_sleep, name):
            setattr(main_mod, name, getattr(dream_sleep, name))

    from dream_sleep import load_init_adapter

    load_init_adapter(model, args.init_adapter, args.lora_rank, args.lora_alpha)
    model.eval()

    cache = load_dream_cache(args.cache)
    facts = cache.fact_list
    state = cache.wake_state
    if args.rebuild_state:
        from dream_sleep import run_chunks

        transcript = torch.tensor([cache.transcript_ids], device=device)
        chunk = getattr(
            importlib.import_module(f"models.{os.getenv('MODEL_NAME', 'mamba2_780m')}.train_hooks"),
            "DEFAULT_CHUNK_LEN", 512)
        state = run_chunks(model, transcript, None, chunk, "wake", keep_logits=False)[1]
        print(f"[{ts()}] wake state rebuilt under this adapter "
              f"({transcript.shape[1]} transcript tokens)")
    print(f"[{ts()}] {args.cache}: facts " + ", ".join(f"{f.entity}={f.code}" for f in facts))

    def first_id(entity: str) -> int:
        return int(tokenizer(f" {entity}", add_special_tokens=False)["input_ids"][0])

    ids = {f.entity: first_id(f.entity) for f in facts}
    print(f"[{ts()}] first-token ids: " + ", ".join(f"{e}={i}" for e, i in ids.items()))

    for template in OPENERS:
        text = template.format(user=model_mod.USER_OPEN, asst=model_mod.ASST_OPEN)
        tokens = torch.tensor(
            [tokenizer(text, add_special_tokens=False)["input_ids"]], device=device)
        with torch.no_grad():
            live = state_to(state, device)
            logits, _ = model(tokens, state=live)
        probs = torch.softmax(logits[0, -1].float(), dim=-1)

        share = {e: float(probs[i]) for e, i in ids.items()}
        total = sum(share.values())
        print(f"\n[{ts()}] {text!r}")
        print(f"[{ts()}]   P(entity) and share of the four:")
        for entity, p in sorted(share.items(), key=lambda kv: -kv[1]):
            bar = "#" * int(60 * (p / total if total else 0))
            print(f"[{ts()}]     {entity:<10} p={p:.5f}  {100 * p / total if total else 0:5.1f}%  {bar}")
        print(f"[{ts()}]   the four together hold {100 * total:.2f}% of the mass")
        top = torch.topk(probs, args.top_k)
        print(f"[{ts()}]   unrestricted top-{args.top_k}: " + ", ".join(
            f"{tokenizer.decode([int(i)])!r}={float(p):.4f}"
            for p, i in zip(top.values, top.indices, strict=True)))


if __name__ == "__main__":
    main()
