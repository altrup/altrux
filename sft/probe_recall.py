"""Compatibility entry point for the recall diagnostic."""

from diagnostics.recall import (
    CHUNK_LEN,
    FILLER_SENTENCES,
    single_token_labels,
    build_probe_rows,
    build_gist_rows,
    score_continuation,
    run_gist,
    ablate_memory,
    ablation_scope,
    clone_state,
    run_chunks,
    score_targets,
    main,
)

__all__ = [
    "CHUNK_LEN",
    "FILLER_SENTENCES",
    "single_token_labels",
    "build_probe_rows",
    "build_gist_rows",
    "score_continuation",
    "run_gist",
    "ablate_memory",
    "ablation_scope",
    "clone_state",
    "run_chunks",
    "score_targets",
    "main",
]


if __name__ == "__main__":
    main()
