"""Compatibility entry point for dream-sleep grid reporting."""

from reporting.grid import (
    aggregate_bound,
    apply_floor,
    arm_and_seed,
    cell,
    check_hashes,
    check_init_adapter,
    cli_main,
    curve_rows,
    floor_deltas,
    load_cells,
    main,
    print_curves,
)


__all__ = [
    "aggregate_bound",
    "apply_floor",
    "arm_and_seed",
    "cell",
    "check_hashes",
    "check_init_adapter",
    "cli_main",
    "curve_rows",
    "floor_deltas",
    "load_cells",
    "main",
    "print_curves",
]


if __name__ == "__main__":
    cli_main()
