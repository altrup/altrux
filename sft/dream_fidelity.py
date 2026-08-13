"""Compatibility entry point for the dream-fidelity probe."""

from diagnostics.dream_fidelity import generate, load_trainable, main, overlap_with_prime


__all__ = ["generate", "load_trainable", "main", "overlap_with_prime"]


if __name__ == "__main__":
    main()
