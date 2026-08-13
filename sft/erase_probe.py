"""Compatibility entry point for the erasure efficacy probe."""

from experiments.erasure.operators import (
    clear_state_top_dirs_cache,
    deflate,
    rank1_erase,
    state_top_dirs,
)
from experiments.erasure.probe import (
    FILLER_SPAN,
    UNBOUND_LOGPROB,
    build_sweep,
    group_by_layer,
    main,
)
from experiments.erasure.wake_items import (
    Bystander,
    assert_no_collisions,
    build_bystanders,
    build_dialogue,
    build_mixed_turns,
    build_nearcone,
    build_wake_items,
    collisions,
    report_distractors,
)

__all__ = [
    "Bystander",
    "FILLER_SPAN",
    "UNBOUND_LOGPROB",
    "assert_no_collisions",
    "build_bystanders",
    "build_dialogue",
    "build_mixed_turns",
    "build_nearcone",
    "build_sweep",
    "build_wake_items",
    "clear_state_top_dirs_cache",
    "collisions",
    "deflate",
    "group_by_layer",
    "main",
    "rank1_erase",
    "report_distractors",
    "state_top_dirs",
]


if __name__ == "__main__":
    main()
