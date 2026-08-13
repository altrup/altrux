"""Compatibility checks for adaptive manifest extraction."""


def test_adaptive_wake_reexports_manifest_api():
    import adaptive_wake
    from experiments.adaptive import manifest

    for name in (
        "AdaptiveWakeError",
        "WakePlan",
        "ExperimentConfig",
        "WakeSpec",
        "ExperimentManifest",
        "token_sha",
        "load_wake_plan",
        "load_experiment_manifest",
        "counterbalanced_facts",
    ):
        assert getattr(adaptive_wake, name) is getattr(manifest, name)
