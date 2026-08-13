import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2]))

import capacity_ladder
from experiments.consolidation import capacity


def test_capacity_ladder_reexports_experiment_surface():
    names = ("CONTROL_THRESHOLD", "DEFAULT_GRID", "main", "parse_grid")

    assert all(getattr(capacity_ladder, name) is getattr(capacity, name) for name in names)
