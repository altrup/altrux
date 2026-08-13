import importlib


def test_top_level_wrapper_reexports_provider_main():
    provider = importlib.import_module("providers.user_generator")
    legacy = importlib.import_module("user_generator_cli")

    assert legacy.__all__ == ["main"]
    assert legacy.main is provider.main
