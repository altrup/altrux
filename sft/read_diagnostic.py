"""Compatibility entry point for the per-token read diagnostic."""

from diagnostics.reads import MODEL_NAME, load_trainable, main, percentiles


__all__ = ["MODEL_NAME", "load_trainable", "main", "percentiles"]


if __name__ == "__main__":
    main()
