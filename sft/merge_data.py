"""Concatenates multiple prepare_data.py outputs (each {ids: [...], masks: [...]})
into one .pt file. train.py only accepts a single --data path, so mixing
sources (e.g. LongAlign-10k + babilong) needs this merge step after each is
tokenized separately.
"""

from preparation.merge import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
