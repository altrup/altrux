import importlib
import pickle


PUBLIC = (
    "PilotDream", "PilotCapture", "QUANTILES", "FAMILIES", "MIN_AUC",
    "CLIP_QUANTILE", "auc", "eligible_positions", "oracle_positions",
    "separability", "scheme_weights", "quantile", "readout_removals",
    "_mean", "score_scheme", "recommend", "HEADER", "format_row", "main",
)


def test_gate_pilot_explicitly_reexports_the_domain_api_and_legacy_pickle_globals():
    legacy = importlib.import_module("gate_pilot")
    domain = importlib.import_module("experiments.erasure.pilot")

    assert tuple(legacy.__all__) == PUBLIC
    assert tuple(domain.__all__) == PUBLIC
    for name in PUBLIC:
        assert getattr(legacy, name) is getattr(domain, name)

    assert pickle.loads(b"\x80\x04cgate_pilot\nPilotDream\n.") is legacy.PilotDream
    assert pickle.loads(b"\x80\x04cgate_pilot\nPilotCapture\n.") is legacy.PilotCapture
