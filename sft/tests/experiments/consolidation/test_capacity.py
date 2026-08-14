import sys
from pathlib import Path


from experiments.consolidation.capacity import parse_grid


def test_parse_grid():
    assert parse_grid("4x40, 24x800") == [(4, 40), (24, 800)]
