#!/usr/bin/env python3
"""Normalize authenticated Claude Code or Codex CLI sessions as wake-user JSON."""

import sys

from providers.user_generator import main

__all__ = ["main"]


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
