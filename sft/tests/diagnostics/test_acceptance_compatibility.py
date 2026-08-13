import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2]))

import acceptance_check
from diagnostics import acceptance


def test_acceptance_check_reexports_diagnostic_surface():
    for name in ("MARKER_SHARE_MAX", "PLAIN_BRACKET_ID", "Dream", "acceptance"):
        assert getattr(acceptance_check, name) is getattr(acceptance, name)
