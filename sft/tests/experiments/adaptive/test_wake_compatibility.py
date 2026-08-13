"""Compatibility checks for adaptive live-wake extraction."""


def test_adaptive_wake_reexports_live_wake_implementation():
    import adaptive_wake
    from experiments.adaptive import wake

    assert adaptive_wake.CommandUserGenerator is wake.CommandUserGenerator
    assert adaptive_wake.LiveWakeHarness is wake.LiveWakeHarness


def test_adaptive_wake_reexports_the_coordinator_implementation():
    import adaptive_wake
    from experiments.adaptive import coordinator

    assert adaptive_wake.MultiSleepCoordinator is coordinator.MultiSleepCoordinator
    assert adaptive_wake.ExperimentRuntime is coordinator.ExperimentRuntime
