"""Compatibility checks for adaptive live-wake extraction."""


def test_adaptive_wake_reexports_live_wake_implementation():
    import adaptive_wake
    from experiments.adaptive import wake

    assert adaptive_wake.CommandUserGenerator is wake.CommandUserGenerator
    assert adaptive_wake.LiveWakeHarness is wake.LiveWakeHarness
