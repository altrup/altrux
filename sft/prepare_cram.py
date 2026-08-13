"""Wikipedia IMR cram block generation compatibility exports."""

from preparation.cram import (
    _find,
    add_block_args,
    build_blocks,
    emit,
    find_subsequence,
    load_wikipedia_passages,
    main,
    make_items,
    split_articles,
    split_sentences,
    validate_blocks,
)

__all__ = [
    "add_block_args",
    "build_blocks",
    "emit",
    "find_subsequence",
    "load_wikipedia_passages",
    "make_items",
    "split_articles",
    "split_sentences",
    "validate_blocks",
    "_find",
    "main",
]


if __name__ == "__main__":
    main()
