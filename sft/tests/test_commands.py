import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SFT = ROOT / "sft"


@pytest.mark.parametrize(
    ("module", "option"),
    (
        ("training.cli", "--resume"),
        ("experiments.dreams.cli", "--arm"),
        ("experiments.adaptive.cli", "--manifest"),
    ),
)
def test_primary_domain_commands_have_help(module: str, option: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=SFT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert option in result.stdout


@pytest.mark.parametrize(
    ("target", "module"),
    (
        ("train", "training.cli"),
        ("dream-sleep", "experiments.dreams.cli"),
        ("adaptive-multisleep", "experiments.adaptive.cli"),
    ),
)
def test_primary_make_targets_run_domain_modules(target: str, module: str) -> None:
    result = subprocess.run(
        ["make", "-n", target],
        cwd=SFT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"python -u -m {module}" in result.stdout
