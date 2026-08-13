"""Compatibility entry point for the fact-correction probe."""

from diagnostics.correction import build_prefix, build_query, main


__all__ = ["build_prefix", "build_query", "main"]


if __name__ == "__main__":
    main()
