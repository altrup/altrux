"""Compatibility wrapper for the adaptive CPU smoke test."""

import sys

from experiments.adaptive.runner import (
    FakeModel,
    FakeState,
    FakeTokenizer,
    fake_backend,
    generator,
    smoke_main,
)

main = smoke_main

__all__ = ["generator", "FakeState", "FakeTokenizer", "FakeModel", "fake_backend", "smoke_main", "main"]


if __name__ == "__main__":
    if "--generator" in sys.argv:
        generator()
    else:
        smoke_main()
