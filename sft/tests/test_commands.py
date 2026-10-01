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


@pytest.mark.parametrize(
    "module",
    (
        "preparation.conversations",
        "preparation.babilong",
        "preparation.merge",
        "preparation.interference",
        "preparation.chains",
        "preparation.inspection",
        "preparation.cram",
        "preparation.needles",
        "preparation.filtering",
        "diagnostics.recall",
        "diagnostics.correction",
        "diagnostics.reads",
        "diagnostics.dream_fidelity",
        "diagnostics.training_smoke",
        "experiments.consolidation.null",
        "experiments.consolidation.capacity",
        "experiments.erasure.probe",
        "reporting.grid",
        "diagnostics.acceptance",
        "diagnostics.knobs",
        "diagnostics.topic_choice",
        "preparation.eligibility",
        "experiments.adaptive.runner",
        "experiments.erasure.pilot",
        "providers.user_generator",
    ),
)
def test_make_domain_commands_have_help(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=SFT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_make_does_not_run_top_level_python_files() -> None:
    makefile = (SFT / "Makefile").read_text()
    assert not any(
        word.endswith(".py")
        for line in makefile.splitlines()
        if "uv run" in line
        for word in line.split()
    )


@pytest.mark.parametrize(
    ("target", "module"),
    (
        ("acceptance-check", "diagnostics.acceptance"),
        ("eligibility-check", "preparation.eligibility"),
        ("measure-knobs", "diagnostics.knobs"),
        ("topic-choice", "diagnostics.topic_choice"),
        ("gate-pilot", "experiments.erasure.pilot"),
        ("adaptive-wake-smoke", "experiments.adaptive.runner"),
        ("user-generator", "providers.user_generator"),
    ),
)
def test_remaining_make_targets_run_domain_modules(target: str, module: str) -> None:
    result = subprocess.run(
        ["make", "-n", target],
        cwd=SFT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"-m {module}" in result.stdout


def test_sft_root_has_only_the_shared_progress_module() -> None:
    assert {path.name for path in SFT.glob("*.py")} == {"progress.py"}


def test_shell_drivers_live_under_scripts() -> None:
    names = {
        "_driver_common.sh",
        "cue_sweep.sh",
        "cue_sweep2.sh",
        "cue_sweep3.sh",
        "cue_sweep4.sh",
        "cue_sweep5.sh",
        "install_probe.sh",
        "run_grid.sh",
        "run_grid2.sh",
        "run_ladder.sh",
    }
    assert not list(SFT.glob("*.sh"))
    assert {path.name for path in (SFT / "scripts").glob("*.sh")} == names


@pytest.mark.parametrize(
    ("target", "script"),
    (
        ("cue-sweep", "cue_sweep.sh"),
        ("cue-sweep2", "cue_sweep2.sh"),
        ("cue-sweep3", "cue_sweep3.sh"),
        ("cue-sweep4", "cue_sweep4.sh"),
        ("cue-sweep5", "cue_sweep5.sh"),
        ("install-probe", "install_probe.sh"),
        ("grid", "run_grid.sh"),
        ("grid2", "run_grid2.sh"),
        ("ladder", "run_ladder.sh"),
    ),
)
def test_shell_drivers_are_exposed_by_make(target: str, script: str) -> None:
    result = subprocess.run(
        ["make", "-n", target, "ARGS=1234"],
        cwd=SFT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"bash scripts/{script}" in result.stdout
