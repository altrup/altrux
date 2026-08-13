import subprocess
import sys
from pathlib import Path


SFT = Path(__file__).parents[2]
ROOT = SFT.parent


def test_train_reexports_training_datasets_api():
    names = (
        "DataSpec",
        "parse_data_spec",
        "check_shares",
        "load_datasets",
        "group_specs",
        "pick_deficit",
        "resolve_share",
        "build_order",
        "recall_weight_at",
        "dataset_fingerprint",
        "datasets_fingerprint",
    )
    script = f"""
import os
import sys
import types

models = types.ModuleType("models")
models.__path__ = []
model = types.ModuleType("models.compat_test")
model.MODEL_ID = "compat-test"
hooks = types.ModuleType("models.compat_test.train_hooks")
sys.modules.update({{
    "models": models,
    "models.compat_test": model,
    "models.compat_test.train_hooks": hooks,
}})
os.environ["MODEL_NAME"] = "compat_test"
sys.path[:0] = [{str(SFT)!r}, {str(ROOT)!r}]

import train
from training import datasets

names = {names!r}
assert all(getattr(train, name) is getattr(datasets, name) for name in names)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
