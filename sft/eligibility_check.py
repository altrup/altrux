"""Count gist-eval-eligible conversations in a prepare_data.py .pt file.

A conversation is eligible for (prefix P, cont C) when some turn-boundary
token (USER_OPEN/ASST_OPEN) has >= P tokens before it and >= C after —
build_gist_rows' criterion in probe_recall.py. Prints a P x C grid so the
probe config can be chosen per corpus before sweeping.

Usage: python eligibility_check.py data/eval_ultrachat.pt [--prefixes 1024,2048,...] [--conts 256,512]
"""

from preparation.eligibility import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
