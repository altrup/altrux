import importlib


def test_top_level_chains_wrapper_reexports_public_api():
    implementation = importlib.import_module("preparation.chains")
    legacy = importlib.import_module("prepare_chains")

    expected = ["sample_log_uniform", "build_chains", "SUSPENDED", "UU_SILENT", "AA", "validate", "main"]
    assert legacy.__all__ == expected
    for name in expected:
        assert getattr(legacy, name) is getattr(implementation, name)
