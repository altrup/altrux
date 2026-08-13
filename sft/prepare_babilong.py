"""Convert RMT-team/babilong records to messages-shaped JSONL."""

from preparation.babilong import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
