import importlib


def test_top_level_lora_reexports_adapter_api():
    adapter = importlib.import_module("adapters.lora")
    legacy = importlib.import_module("lora")

    assert legacy.__all__ == adapter.__all__
    for name in adapter.__all__:
        assert getattr(legacy, name) is getattr(adapter, name)
