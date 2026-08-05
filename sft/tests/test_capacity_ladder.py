import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from capacity_ladder import parse_grid, set_memory_injection


class _Injecting:
    def __init__(self) -> None:
        self.injection_enabled = True


class _Plain:
    pass


def test_parse_grid():
    assert parse_grid("4x40, 24x800") == [(4, 40), (24, 800)]


def test_disabling_injection_turns_the_memory_path_off():
    model = _Injecting()
    assert set_memory_injection(model, False) is True
    assert model.injection_enabled is False


def test_enabling_injection_leaves_the_memory_path_on():
    model = _Injecting()
    assert set_memory_injection(model, True) is True
    assert model.injection_enabled is True


def test_a_model_without_a_memory_path_reports_nothing_to_disable():
    model = _Plain()
    assert set_memory_injection(model, False) is False
    assert not hasattr(model, "injection_enabled")
