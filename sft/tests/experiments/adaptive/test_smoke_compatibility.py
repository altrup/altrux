import importlib


PUBLIC = ("generator", "FakeState", "FakeTokenizer", "FakeModel", "fake_backend", "smoke_main", "main")


def test_adaptive_wake_smoke_reexports_runner_implementation():
    legacy = importlib.import_module("adaptive_wake_smoke")
    domain = importlib.import_module("experiments.adaptive.runner")

    for name in PUBLIC:
        assert getattr(legacy, name) is getattr(domain, name)
