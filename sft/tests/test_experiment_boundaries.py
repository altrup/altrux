import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "sft" / "experiments"


def _imported_experiment_folders(path: Path) -> set[str]:
    folders = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
        elif isinstance(node, ast.Import):
            parts = node.names[0].name.split(".")
        else:
            continue
        if parts[0] == "experiments" and len(parts) > 2 and (EXPERIMENTS / parts[1]).is_dir():
            folders.add(parts[1])
    return folders


@pytest.mark.xfail(strict=True, reason="cross-experiment imports remain until the folder split")
def test_experiment_folders_only_import_shared_modules():
    crossing = []
    for folder in sorted(p for p in EXPERIMENTS.iterdir() if p.is_dir()):
        for path in sorted(folder.rglob("*.py")):
            for other in _imported_experiment_folders(path) - {folder.name}:
                crossing.append(f"{path.relative_to(ROOT)} -> experiments/{other}")
    assert not crossing, "\n".join(crossing)
