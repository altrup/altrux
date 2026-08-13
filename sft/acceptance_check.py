"""Compatibility entry point for the warm-start acceptance check."""

from diagnostics.acceptance import Dream, MARKER_SHARE_MAX, PLAIN_BRACKET_ID, acceptance, main


__all__ = ["Dream", "MARKER_SHARE_MAX", "PLAIN_BRACKET_ID", "acceptance", "main"]


if __name__ == "__main__":
    main()
