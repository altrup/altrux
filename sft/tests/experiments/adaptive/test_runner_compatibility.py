"""Compatibility checks for adaptive run persistence."""


def test_adaptive_multisleep_reexports_registered_run_implementation():
    import adaptive_multisleep
    from experiments.adaptive import runner

    assert adaptive_multisleep.run_registered_experiment is runner.run_registered_experiment
