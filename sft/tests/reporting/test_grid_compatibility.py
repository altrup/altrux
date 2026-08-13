import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2]))

import summarize_grid
from reporting import grid


def test_summarize_grid_reexports_reporting_surface():
    names = (
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
    )

    assert all(getattr(summarize_grid, name) is getattr(grid, name) for name in names)
