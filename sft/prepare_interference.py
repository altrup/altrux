"""Interference-recall data preparation compatibility exports."""

from preparation.interference import COLORS, FACT_KINDS, LABEL_POOL, LABEL_SKIP, NAMES, WEEKDAYS, main

__all__ = ["LABEL_POOL", "LABEL_SKIP", "COLORS", "WEEKDAYS", "NAMES", "FACT_KINDS", "main"]


if __name__ == "__main__":
    main()
