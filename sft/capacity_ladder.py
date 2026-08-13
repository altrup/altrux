"""Compatibility entry point for the in-context capacity ladder."""

from experiments.consolidation.capacity import CONTROL_THRESHOLD, DEFAULT_GRID, main, parse_grid


__all__ = ["CONTROL_THRESHOLD", "DEFAULT_GRID", "main", "parse_grid"]


if __name__ == "__main__":
    main()
