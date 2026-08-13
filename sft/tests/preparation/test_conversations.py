import importlib


def test_top_level_prepare_data_wrapper_reexports_public_api():
    implementation = importlib.import_module("preparation.conversations")
    legacy = importlib.import_module("prepare_data")

    expected = [
        "RECAP_QUOTE_CHARS",
        "format_conversation",
        "recap_messages",
        "pack_records",
        "format_pack",
        "iter_records",
        "report_packing",
        "main",
    ]
    assert legacy.__all__ == expected
    for name in expected:
        assert getattr(legacy, name) is getattr(implementation, name)
