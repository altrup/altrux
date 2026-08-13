import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

import probe_recall
import smoke_test
from diagnostics import recall, training_smoke


def test_diagnostic_compatibility_wrappers_reexport_their_implementations():
    for wrapper, implementation, names in (
        (probe_recall, recall, recall.__all__),
        (smoke_test, training_smoke, training_smoke.__all__),
    ):
        assert wrapper.__all__ == names
        for name in names:
            assert getattr(wrapper, name) is getattr(implementation, name)


def test_diagnostic_compatibility_scripts_delegate_cli_help():
    for script, option in (("probe_recall.py", "--gist"), ("smoke_test.py", "--batch-size")):
        result = subprocess.run(
            [sys.executable, str(ROOT / script), "--help"],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT.parent)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert option in result.stdout
