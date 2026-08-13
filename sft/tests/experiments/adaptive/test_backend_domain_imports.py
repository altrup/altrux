import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

from experiments.adaptive.backend import DreamSleepBackend


def test_backend_uses_domain_dream_modules_for_moved_sleep_apis():
    source = inspect.getsource(DreamSleepBackend.sleep)

    assert "from experiments.dreams.cache import" in source
    assert "from experiments.dreams.distillation import" in source
    assert "from experiments.dreams.generation import" in source
    assert "from experiments.dreams.probes import" in source
    assert "from dream_sleep import (" not in source
