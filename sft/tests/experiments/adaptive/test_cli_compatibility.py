import importlib


def test_adaptive_cli_domain_exposes_main_and_registered_options():
    cli = importlib.import_module("experiments.adaptive.cli")

    assert callable(cli.main)
    assert cli.PARSER_OPTIONS == ("--manifest", "--seed", "--aggregate-only")
