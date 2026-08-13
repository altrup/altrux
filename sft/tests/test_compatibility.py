"""Characterization tests for the sft refactor compatibility surface."""

import importlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SFT = ROOT / "sft"


@pytest.mark.parametrize(
    ("module_name", "names"),
    {
        "adaptive_multisleep": (
            "artifact_transcript_ids",
            "manifest_sha",
            "run_registered_experiment",
        ),
        "adaptive_wake": (
            "AdaptiveWakeError",
            "ExperimentConfig",
            "ExperimentManifest",
            "WakePlan",
            "load_experiment_manifest",
            "load_wake_plan",
        ),
        "b4": ("erase_subspace", "erase_subspace_scaled", "sigma_gammas"),
        "consolidation_null": ("Fact", "exact_match", "generate", "ts"),
        "dream_sleep": (
            "CachedDream",
            "DreamCache",
            "DreamSetCache",
            "build_distractors",
            "load_dream_cache",
            "save_dream_cache",
        ),
        "erase_probe": ("Bystander", "deflate", "state_top_dirs"),
        "gate_pilot": ("PilotCapture", "PilotDream", "scheme_weights"),
        "prepare_cram": ("build_blocks", "_find"),
        "prepare_data": ("format_conversation", "format_pack", "pack_records"),
        "prepare_needles": ("babilong_items",),
        "probes_common": ("battery_summary", "code_margin", "perplexity"),
    }.items(),
)
def test_repository_used_imports_remain_available(module_name, names):
    module = importlib.import_module(module_name)
    assert all(hasattr(module, name) for name in names)


@pytest.mark.parametrize(
    ("script", "option"),
    (
        ("dream_sleep.py", "--arm"),
        ("adaptive_multisleep.py", "--manifest"),
        ("user_generator_cli.py", "--provider"),
    ),
)
def test_top_level_cli_help_remains_available(script, option):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, str(SFT / script), "--help"],
        cwd=SFT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert option in result.stdout


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    (
        ("dream_sleep", "Dream"),
        ("dream_sleep", "DreamCache"),
        ("dream_sleep", "CachedDream"),
        ("dream_sleep", "DreamSetCache"),
        ("consolidation_null", "Fact"),
    ),
)
def test_legacy_pickle_globals_still_resolve(module_name, class_name):
    module = importlib.import_module(module_name)
    legacy_global = f"c{module_name}\n{class_name}\n.".encode()
    assert pickle.loads(b"\x80\x04" + legacy_global) is getattr(module, class_name)
