"""The warm-start acceptance check — DISCUSSION-20260808 §2.1, at the token level.

A warm start is accepted when its free-running dreams emit the REGISTERED role
markers rather than plain-text imitations of them, and no mojibake. Both
clauses are counted over token ids, never over the decoded string: `[USER]`
decodes identically whether it is special token 50277 or the four ordinary
tokens that spell it, and telling those apart is the entire point of the
check (the 08-08 run's finding).

Calibration on the 780M adapter (§2.1): 77% plain-bracket share at 400 steps
(fail), 26% at 800 (pass).

    MODEL_NAME=... python acceptance_check.py --cache data/dream_set_s1234.pt
"""

from __future__ import annotations

import argparse
from typing import Protocol, Sequence

# Plain-bracket tokens may be at most this share of marker-slot emissions in
# the free-running spans (§2.1, re-specified from the 08-08 run's request).
MARKER_SHARE_MAX = 0.35

# The plain ']' the mimicry spells the marker with, in the GPT-NeoX vocabulary.
PLAIN_BRACKET_ID = 62


class Dream(Protocol):
    dream_ids: list[int]
    token_texts: list[str]
    prefix_len: int


def acceptance(
    dreams: Sequence[Dream], user_id: int, asst_id: int, plain_id: int = PLAIN_BRACKET_ID
) -> dict[str, object]:
    """Both §2.1 clauses over the free-running spans of ``dreams``."""
    plain = markers = non_ascii = 0
    for dream in dreams:
        ids = dream.dream_ids[dream.prefix_len :]
        texts = dream.token_texts[dream.prefix_len :]
        plain += sum(1 for token in ids if token == plain_id)
        markers += sum(1 for token in ids if token in (user_id, asst_id))
        non_ascii += sum(1 for text in texts if not text.isascii())
    slots = plain + markers
    share = plain / slots if slots else 0.0
    passed = bool(slots) and share < MARKER_SHARE_MAX and non_ascii == 0
    return {
        "dreams": len(dreams),
        "plain": plain,
        "markers": markers,
        "share": share,
        "non_ascii": non_ascii,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache", required=True, help="Dream cache to score (single-dream or dream-set)"
    )
    args = parser.parse_args()

    import importlib
    import os

    from models.common import build_tokenizer

    from dotenv import load_dotenv

    from experiments.dreams.cache import load_dream_cache
    from progress import ts

    load_dotenv()
    model_mod = importlib.import_module(f"models.{os.getenv('MODEL_NAME', 'mamba2_780m')}")
    tokenizer = build_tokenizer(model_mod)
    cache = load_dream_cache(args.cache)
    dreams = list(getattr(cache, "dreams", [cache]))

    result = acceptance(
        dreams,
        user_id=tokenizer.convert_tokens_to_ids(model_mod.USER_OPEN),
        asst_id=tokenizer.convert_tokens_to_ids(model_mod.ASST_OPEN),
    )
    print(f"[{ts()}] acceptance check ({args.cache}, {result['dreams']} dreams, free spans only)")
    print(f"[{ts()}]   real markers (USER/ASST ids) : {result['markers']}")
    print(f"[{ts()}]   plain ']' (id {PLAIN_BRACKET_ID})            : {result['plain']}")
    print(
        f"[{ts()}]   plain share of marker slots  : {result['share']:.1%}  "
        f"(must be < {MARKER_SHARE_MAX:.0%})"
    )
    print(f"[{ts()}]   non-ASCII tokens (mojibake)  : {result['non_ascii']}  (must be 0)")
    print(f"[{ts()}] {'PASSED' if result['passed'] else 'FAILED'}")
    raise SystemExit(0 if result["passed"] else 1)


__all__ = ["Dream", "MARKER_SHARE_MAX", "PLAIN_BRACKET_ID", "acceptance", "main"]


if __name__ == "__main__":
    main()
