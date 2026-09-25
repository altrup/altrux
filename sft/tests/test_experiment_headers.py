import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = re.compile(r"Experiment: (lama-ckl|state-erasure|memory-model|shared)$")


def test_every_module_names_its_owning_experiment():
    protected = set((ROOT / "PROTECTED_PATHS").read_text().split())
    missing = []
    for top in ("sft", "models"):
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT)
            if {".venv", "tests"} & set(rel.parts) or path.name == "__init__.py" or str(rel) in protected:
                continue
            doc = ast.get_docstring(ast.parse(path.read_text())) or ""
            if not HEADER.fullmatch(doc.partition("\n")[0]):
                missing.append(str(rel))
    assert not missing
