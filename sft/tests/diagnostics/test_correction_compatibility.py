import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2]))

from diagnostics import correction


def test_probe_correction_reexports_diagnostic_surface():
    import probe_correction

    names = ("build_prefix", "build_query", "main")

    assert all(getattr(probe_correction, name) is getattr(correction, name) for name in names)
