"""babilong-style needle block generation compatibility exports."""

from preparation.needles import (
    add_block_args,
    babilong_items,
    build_blocks,
    emit,
    load_babilong,
    load_wikipedia_passages,
    main,
    split_articles,
)

__all__ = [
    "add_block_args",
    "babilong_items",
    "build_blocks",
    "emit",
    "load_babilong",
    "load_wikipedia_passages",
    "split_articles",
    "main",
]


if __name__ == "__main__":
    main()
