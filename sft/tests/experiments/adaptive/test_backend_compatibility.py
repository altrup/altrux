import importlib


PUBLIC = (
    "RuntimeBackend", "artifact_transcript_ids", "dream_cache_identity",
    "battery_collision_terms", "manifest_payload", "manifest_sha",
    "bound_rehearsal_counts", "DreamSleepBackend",
)


def test_adaptive_multisleep_reexports_backend_surface():
    legacy = importlib.import_module("adaptive_multisleep")
    domain = importlib.import_module("experiments.adaptive.backend")

    assert tuple(domain.__all__) == PUBLIC
    for name in PUBLIC:
        assert getattr(legacy, name) is getattr(domain, name)
