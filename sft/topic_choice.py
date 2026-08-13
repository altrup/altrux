"""Compatibility entry point for the topic-choice diagnostic."""

from diagnostics.topic_choice import OPENERS, main


__all__ = ["OPENERS", "main"]


if __name__ == "__main__":
    main()
