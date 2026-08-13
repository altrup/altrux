"""Compatibility entry point for the memory knob diagnostic."""

from diagnostics.knobs import MODEL_NAME, load_trainable, main, percentiles


__all__ = ["MODEL_NAME", "load_trainable", "main", "percentiles"]


if __name__ == "__main__":
    main()
