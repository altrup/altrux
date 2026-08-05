import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from capacity_ladder import parse_grid


def test_parse_grid():
    assert parse_grid("4x40, 24x800") == [(4, 40), (24, 800)]
