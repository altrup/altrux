"""Characterization tests for the dream CLI extraction boundary."""

import importlib


def test_cli_parser_and_main_have_a_domain_owner_and_legacy_exports():
    cli = importlib.import_module("experiments.dreams.cli")
    legacy = importlib.import_module("dream_sleep")

    assert legacy.build_parser is cli.build_parser
    assert legacy.main is cli.main
    assert cli.build_parser().parse_args([]).arm == "replay"
