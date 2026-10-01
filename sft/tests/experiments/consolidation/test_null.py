"""CPU tests for experiments/consolidation/null.py's pure logic: transcript assembly,
answer grading, the cue ladder, and the pre-registered verdict thresholds.

Nothing here loads a model or a tokenizer -- consolidation_null keeps every
`models.*` import inside main(), so importing the module is CPU-safe, and the
transcript builder takes a token-length callable instead of a tokenizer.
"""

import itertools
import random
import sys
from pathlib import Path

from experiments.consolidation.null import (
    FILLER_SENTENCES,
    PASS_MATCH_RATE,
    UNDERPOWERED_DELTA_NATS,
    build_facts,
    build_turns,
    contains_code,
    exact_match,
    extract_answer,
    first_success,
    pass_at_k,
    render_turns,
    replay_step,
    role_adjacency_violations,
    verdict,
)


def words(text: str) -> int:
    return len(text.split())


def transcript(n_facts: int = 6, filler: int = 20, seed: int = 7):
    rng = random.Random(seed)
    facts = build_facts(n_facts, rng)
    turns = build_turns(facts, filler, words, rng)
    return facts, turns


def test_every_fact_is_stated_exactly_once():
    facts, turns = transcript()
    text = render_turns(turns, "[USER]", "[ASSISTANT]")
    for fact in facts:
        assert text.count(fact.code) == 1
        assert text.count(fact.entity) == 2  # the question turn and its answer


def test_entities_are_distinct():
    facts, _ = transcript(n_facts=12)
    assert len({f.entity for f in facts}) == 12


def test_roles_strictly_alternate():
    _, turns = transcript(n_facts=8, filler=50)
    assert role_adjacency_violations(turns) == 0
    assert turns[0][0] == "user"


def test_filler_reaches_its_token_budget_between_every_pair_of_facts():
    facts, turns = transcript(n_facts=4, filler=40)
    codes = [t for t in turns if any(f.code in t[1] for f in facts)]
    assert len(codes) == 4
    gaps = []
    idx = [i for i, t in enumerate(turns) if any(f.code in t[1] for f in facts)]
    for a, b in itertools.pairwise(idx):
        gaps.append(sum(words(turns[i][1]) for i in range(a + 1, b)))
    assert all(g >= 40 for g in gaps)


def test_filler_carries_no_digits_that_could_collide_with_a_code():
    assert not any(c.isdigit() for s in FILLER_SENTENCES for c in s)


def test_same_seed_rebuilds_the_same_transcript():
    a = render_turns(transcript(seed=3)[1], "[USER]", "[ASSISTANT]")
    b = render_turns(transcript(seed=3)[1], "[USER]", "[ASSISTANT]")
    assert a == b
    c = render_turns(transcript(seed=4)[1], "[USER]", "[ASSISTANT]")
    assert a != c


def test_extract_answer_drops_trailing_punctuation_and_whitespace():
    assert extract_answer("  4 8 2 1 3 .  \n") == "4 8 2 1 3"
    assert extract_answer(" 4 8 2 1 3. The code for the heron") == "4 8 2 1 3"


def test_extract_answer_stops_at_a_role_marker():
    assert extract_answer(" 4 8 2 1 3[USER] hello", stops=("[USER]",)) == "4 8 2 1 3"


def test_exact_match_ignores_digit_spacing_but_not_the_digits():
    assert exact_match(" 4 8 2 1 3.", "4 8 2 1 3")
    assert exact_match(" 48213.", "4 8 2 1 3")
    assert not exact_match(" 4 8 2 1 9.", "4 8 2 1 3")
    assert not exact_match("", "4 8 2 1 3")


def test_exact_match_rejects_an_answer_that_only_contains_the_code_later():
    assert not exact_match(" I think. The code is 4 8 2 1 3.", "4 8 2 1 3")
    assert contains_code(" I think. The code is 4 8 2 1 3.", "4 8 2 1 3")


def test_pass_at_k_is_the_fraction_of_samples_containing_the_code():
    samples = [" 4 8 2 1 3.", " no idea", " maybe 48213 then", " 1 1 1 1 1"]
    assert pass_at_k(samples, "4 8 2 1 3") == 0.5
    assert pass_at_k([], "4 8 2 1 3") == 0.0


def test_cue_ladder_stops_at_the_first_rung_that_succeeds():
    probed = []

    def probe(prompt: str) -> bool:
        probed.append(prompt)
        return prompt == "rung2"

    assert first_success(["rung1", "rung2", "rung3"], probe) == 2
    assert probed == ["rung1", "rung2"]


def test_cue_ladder_reports_zero_when_no_rung_succeeds():
    assert first_success(["a", "b"], lambda _: False) == 0


def test_verdict_passes_at_the_registered_match_rate():
    assert verdict(PASS_MATCH_RATE, -5.0) == "PASS"
    assert verdict(1.0, 0.0) == "PASS"


def test_verdict_is_underpowered_when_only_the_logprob_moved():
    assert verdict(0.0, UNDERPOWERED_DELTA_NATS) == "FAIL-UNDERPOWERED"
    assert verdict(PASS_MATCH_RATE - 0.01, 3.0) == "FAIL-UNDERPOWERED"


def test_verdict_is_dead_when_neither_threshold_is_met():
    assert verdict(0.0, 0.0) == "FAIL-DEAD"
    assert verdict(PASS_MATCH_RATE - 0.01, UNDERPOWERED_DELTA_NATS - 0.01) == "FAIL-DEAD"


def test_replay_schedule_carries_state_within_a_pass():
    """Default (carried) schedule: state resets only at a pass boundary, so
    chunk 0 is the only chunk the student ever sees from a fresh state."""
    sched = [replay_step(i, n_chunks=3, fresh_state_replay=False) for i in range(7)]
    assert [c for c, _ in sched] == [0, 1, 2, 0, 1, 2, 0]
    assert [reset for _, reset in sched] == [True, False, False, True, False, False, True]


def test_fresh_state_replay_resets_before_every_chunk():
    """--fresh-state-replay: every chunk is distilled from a fresh state, the
    same condition the post-distillation probes run under."""
    sched = [replay_step(i, n_chunks=3, fresh_state_replay=True) for i in range(7)]
    assert [c for c, _ in sched] == [0, 1, 2, 0, 1, 2, 0]
    assert all(reset for _, reset in sched)
