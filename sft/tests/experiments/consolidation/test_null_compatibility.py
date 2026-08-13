import importlib
import pickle


def test_top_level_consolidation_wrapper_reexports_domain_api():
    implementation = importlib.import_module("experiments.consolidation.null")
    legacy = importlib.import_module("consolidation_null")

    expected = [
        "CODE_DIGITS",
        "ENTITY_POOL",
        "FILLER_SENTENCES",
        "Fact",
        "Turn",
        "build_facts",
        "build_turns",
        "contains_code",
        "cue_rungs",
        "digits",
        "exact_match",
        "extract_answer",
        "fact_turns",
        "first_success",
        "normalize",
        "pass_at_k",
        "render_turns",
        "role_adjacency_violations",
        "generate",
        "kl_loss",
        "replay_step",
        "run_chunks",
        "target_logprob",
        "fmt_duration",
        "ts",
        "PASS_MATCH_RATE",
        "UNDERPOWERED_DELTA_NATS",
        "GEN_TOKENS",
        "verdict",
        "report_transcript",
        "main",
    ]
    assert legacy.__all__ == expected
    for name in expected:
        assert getattr(legacy, name) is getattr(implementation, name)


def test_legacy_fact_pickle_global_resolves():
    legacy = importlib.import_module("consolidation_null")
    payload = b"\x80\x04cconsolidation_null\nFact\n."
    assert pickle.loads(payload) is legacy.Fact
