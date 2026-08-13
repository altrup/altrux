import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[2]))

import read_diagnostic
from diagnostics import reads


def test_read_diagnostic_reexports_diagnostic_surface():
    for name in ("load_trainable", "main", "percentiles"):
        assert getattr(read_diagnostic, name) is getattr(reads, name)


def test_percentiles_keeps_the_existing_format():
    result = reads.percentiles(torch.tensor([0.0, 1.0, 2.0]))

    assert result.startswith("p0  0.0000")
    assert "p50 1.0000" in result
