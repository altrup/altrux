import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[2]))

import measure_knobs
from diagnostics import knobs


def test_measure_knobs_reexports_diagnostic_surface():
    for name in ("load_trainable", "main", "percentiles"):
        assert getattr(measure_knobs, name) is getattr(knobs, name)


def test_percentiles_reports_registered_quantiles():
    result = knobs.percentiles(torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))

    assert "p0  0.0000" in result
    assert "p50 2.0000" in result
    assert "p1004.0000" in result
