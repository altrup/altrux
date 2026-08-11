"""CPU tests for the shared dream cache (DISCUSSION-20260806 sec 3/4): the
round-trip and its hash check, the distractor codes the margin metric needs,
cue target-masking, binding-aware coverage, and the dream seed separator.

The cache is the machine check behind the "one dream per seed, byte-identical
across arms" invariant the 08-06 grid lost in the harness->driver composition.
"""

import pytest
import torch

from consolidation_null import Fact
from dream_sleep import (
    CachedDream,
    DreamCache,
    DreamSetCache,
    aggregate_binding,
    assert_aggregate_binding,
    dream_bases,
    report_dream_set,
    binding_coverage,
    build_distractors,
    dream_seed_text,
    dream_sidecar_text,
    load_dream_cache,
    save_dream_cache,
    scored_keep,
    write_dream_set_sidecar,
    sidecar_path,
    target_keep_mask,
    token_sha,
)

FACTS = [Fact("osprey", "bird", "5 9 7 9 7"), Fact("heron", "bird", "1 2 3 4 5")]


class FakeState:
    def __init__(self, ssm_states):
        self.ssm_states = ssm_states


def _cache(dream_ids=(3, 4, 5, 6)) -> DreamCache:
    return DreamCache(
        seed=1234,
        transcript_ids=[1, 2, 3],
        dream_ids=list(dream_ids),
        wake_state=FakeState([torch.zeros(1, 1, 1, 2)]),
        teacher_logits=torch.randn(len(dream_ids), 5),
        queries=[[torch.randn(1, 2)] for _ in dream_ids],
        token_texts=[f"<{i}>" for i in dream_ids],
        cue_flags=[False] * len(dream_ids),
        distractors={f.entity: "0 0 0 0 0" for f in FACTS},
        facts=[(f.entity, f.category, f.code) for f in FACTS],
    )


def test_cache_round_trips_through_disk(tmp_path):
    path = tmp_path / "dream_cache_s1234.pt"
    cache = _cache()
    save_dream_cache(cache, path)
    loaded = load_dream_cache(path)

    assert loaded.dream_ids == cache.dream_ids
    assert loaded.transcript_ids == cache.transcript_ids
    assert loaded.dream_sha == cache.dream_sha and loaded.transcript_sha == cache.transcript_sha
    torch.testing.assert_close(loaded.teacher_logits, cache.teacher_logits)


def test_cache_hashes_are_over_the_tokens_not_the_object():
    assert _cache().dream_sha == token_sha([3, 4, 5, 6])
    assert _cache([3, 4, 5, 7]).dream_sha != _cache().dream_sha


def test_load_rejects_a_cache_whose_tokens_no_longer_match_their_hash(tmp_path):
    path = tmp_path / "dream_cache_s1234.pt"
    save_dream_cache(_cache(), path)
    corrupted = torch.load(path, weights_only=False)
    corrupted.dream_ids[2] = 99
    torch.save(corrupted, path)

    with pytest.raises(SystemExit):
        load_dream_cache(path)


def test_distractors_are_one_per_fact_and_never_a_real_code():
    distractors = build_distractors(FACTS, seed=1234)

    assert set(distractors) == {f.entity for f in FACTS}
    real = {f.code for f in FACTS}
    for code in distractors.values():
        assert code not in real
        assert len(code.split()) == len(FACTS[0].code.split())


def test_distractors_have_their_own_rng_stream():
    """Drawn from Random(seed ^ const), so adding them cannot shift the wake
    transcript's own draws -- prior runs stay bit-identical."""
    assert build_distractors(FACTS, 1234) == build_distractors(FACTS, 1234)
    assert build_distractors(FACTS, 1234) != build_distractors(FACTS, 2345)


def test_cue_tokens_are_masked_as_targets_only():
    # positions:      0      1     2     3      4
    # cue span covers tokens 1 and 2; token 3 is the first post-cue answer token.
    keep = target_keep_mask([False, True, True, False, False])

    assert keep == [False, False, True, True, True]


def test_binding_coverage_counts_a_code_only_beside_its_own_entity():
    bound, misbound = binding_coverage("The code for the osprey is 5 9 7 9 7.", FACTS)
    assert bound["osprey"] == 1 and misbound["osprey"] == 0


def test_binding_coverage_reports_the_08_06_misbinding_separately():
    """A's seed-1234 dream opened with osprey's code attributed to the heron;
    the old substring coverage counted it as rehearsal."""
    bound, misbound = binding_coverage("The code for the heron is 5 9 7 9 7.", FACTS)

    assert bound["osprey"] == 0 and misbound["osprey"] == 1
    assert bound["heron"] == 0 and misbound["heron"] == 0


def test_binding_coverage_does_not_reach_across_a_sentence_boundary():
    bound, misbound = binding_coverage("The osprey is a bird. The code is 5 9 7 9 7.", FACTS)
    assert bound["osprey"] == 0 and misbound["osprey"] == 0


def test_dream_seed_keeps_the_format_separator():
    """[ASSISTANT] + a literal space is the trained chat format; the old
    .rstrip() dropped it and started every dream one token off-format."""
    assert dream_seed_text("[ASSISTANT]", "") == "[ASSISTANT] "
    assert dream_seed_text("[ASSISTANT]", "a bird") == "[ASSISTANT] a bird"


def test_sidecar_marks_the_cue_spans_in_the_decoded_dream():
    cache = _cache()
    cache.token_texts = ["The", " code", " is", " 5"]
    cache.cue_flags = [False, True, True, False]
    text = dream_sidecar_text(cache)

    assert "The" in text and " 5" in text
    assert " code is" in text.replace("[CUE]", "").replace("[/CUE]", "")
    assert text.count("[CUE]") == 1 and text.count("[/CUE]") == 1


def test_the_sidecar_follows_the_cache_filename(tmp_path):
    """Two caches in one directory must not clobber each other's sidecar --
    the multi-sleep cache sits beside the single-sleep one."""
    assert sidecar_path(tmp_path / "dream_cache_s1234.pt") == tmp_path / "dream_s1234.txt"
    assert sidecar_path(tmp_path / "dream_cache_w4_s1234.pt") == tmp_path / "dream_w4_s1234.txt"
    assert sidecar_path(tmp_path / "picker.pt") == tmp_path / "picker.txt"


# ---- the multi-dream cache (DISCUSSION-20260808 sec 2.10.4) ----------------


def _cached_dream(dream_ids=(3, 4, 5, 6), texts=None) -> CachedDream:
    return CachedDream(
        dream_ids=list(dream_ids),
        token_texts=texts or [f"<{i}>" for i in dream_ids],
        teacher_logits=torch.randn(len(dream_ids), 5),
        cue_flags=[False] * len(dream_ids),
        prefix_len=1,
        stop_reason="eoc",
        divergence=[0.0] * len(dream_ids),
        gate_positions=[1, 2],
        queries=[[torch.randn(1, 2)], [torch.randn(1, 2)]],
        spectra=[[1.0, 0.1]],
        ranks={"ratio-gap": [1], "median": [1]},
        bases={v: [torch.eye(2)[:1]] for v in ("raw", "deflated", "qcm")},
    )


def _set(dream_ids=((3, 4, 5, 6), (7, 8, 9, 10))) -> DreamSetCache:
    return DreamSetCache(
        seed=1234,
        transcript_ids=[1, 2, 3],
        wake_state=FakeState([torch.zeros(1, 1, 1, 2)]),
        dreams=[_cached_dream(ids) for ids in dream_ids],
        distractors={f.entity: "0 0 0 0 0" for f in FACTS},
        facts=[(f.entity, f.category, f.code) for f in FACTS],
        dream_prompt="",
        gate_threshold=1.0,
    )


def test_the_dream_set_round_trips_through_disk(tmp_path):
    path = tmp_path / "dream_set_s1234.pt"
    cache = _set()
    save_dream_cache(cache, path)
    loaded = load_dream_cache(path)

    assert isinstance(loaded, DreamSetCache)
    assert [d.dream_ids for d in loaded.dreams] == [d.dream_ids for d in cache.dreams]
    assert loaded.set_sha == cache.set_sha
    assert loaded.dreams[0].bases.keys() == {"raw", "deflated", "qcm"}


def test_the_set_hash_covers_every_dream_and_their_order():
    """Sec 3: the *set* hash is what a result jsonl asserts, so no cell can
    silently distil a different collection of dreams."""
    assert _set().set_sha == _set().set_sha
    assert _set(((3, 4, 5, 6), (7, 8, 9, 11))).set_sha != _set().set_sha
    assert _set(((7, 8, 9, 10), (3, 4, 5, 6))).set_sha != _set().set_sha


def test_loading_rejects_a_set_whose_dream_tokens_no_longer_match(tmp_path):
    path = tmp_path / "dream_set_s1234.pt"
    save_dream_cache(_set(), path)
    corrupted = torch.load(path, weights_only=False)
    corrupted.dreams[1].dream_ids[0] = 99
    torch.save(corrupted, path)

    with pytest.raises(SystemExit):
        load_dream_cache(path)


def test_aggregate_binding_counts_the_dreams_each_fact_binds_in():
    dreams = [
        _cached_dream(texts=["The code for the osprey is 5 9 7 9 7."]),
        _cached_dream(texts=["The code for the osprey is 5 9 7 9 7. And again 5 9 7 9 7 osprey."]),
        _cached_dream(texts=["Nothing about birds here."]),
    ]

    counts = aggregate_binding(dreams, FACTS)

    assert counts["osprey"] == 2  # dreams, not rehearsals
    assert counts["heron"] == 0


def test_the_aggregate_gate_refuses_a_set_a_fact_is_underbound_in():
    """Sec 4: the per-dream gate is retired for these caches and the aggregate
    one GATES -- today's single-dream report only warns."""
    bound = [_cached_dream(texts=[f"The code for the {f.entity} is {f.code}."]) for f in FACTS]
    with pytest.raises(SystemExit):
        assert_aggregate_binding(bound, FACTS, min_dreams=2)

    assert_aggregate_binding(bound * 2, FACTS, min_dreams=2)


def test_the_steer_prefix_is_excluded_from_the_scored_positions():
    """Sec 4: prefix tokens influence the dream through state only, in EVERY
    arm. Position prefix_len-1 predicts the first free token and is kept."""
    keep = scored_keep([False] * 5, prefix_len=2)

    assert keep == [False, True, True, True, True]


def test_the_prefix_and_the_cue_mask_compose():
    keep = scored_keep([False, False, True, False], prefix_len=1)

    # position 0 is the last prefix token and predicts the first free one.
    assert keep == [True, False, True, True]


def test_dream_bases_share_one_svd_per_layer_across_all_three_variants():
    """Sec 2.7: both rank rules are computed and printed for every layer, the
    spectrum is kept, and the three variants come from the SAME shared basis."""
    torch.manual_seed(0)
    state = FakeState([torch.randn(1, 2, 2, 16) for _ in range(2)])
    queries = [[torch.randn(1, 16) for _ in range(2)] for _ in range(6)]

    spectra, ranks, bases = dream_bases(queries, gate=[0, 2, 4], wake_state=state,
                                        rank_rule="ratio-gap")

    assert len(spectra) == 2 and set(ranks) == {"ratio-gap", "median"}
    assert all(len(r) == 2 for r in ranks.values())
    for variant in ("raw", "deflated", "qcm"):
        for basis in bases[variant]:
            torch.testing.assert_close(basis @ basis.T, torch.eye(basis.shape[0]), atol=1e-5, rtol=0)


def test_the_rank_rule_flag_picks_which_rule_truncates():
    torch.manual_seed(0)
    state = FakeState([torch.randn(1, 2, 2, 64)])
    # Six near-parallel queries: one dominant direction, the rest noise, so the
    # two rules can disagree on where to cut.
    base = torch.randn(1, 64)
    queries = [[base + 0.01 * torch.randn(1, 64)] for _ in range(6)]

    _, ranks, _ = dream_bases(queries, list(range(6)), state, "ratio-gap")

    assert all(r >= 1 for rule in ranks.values() for r in rule)


def test_an_empty_variant_basis_is_a_note_not_a_refusal(capsys):
    """The gate selects B4's queries; it does not validate dreams (altrup,
    2026-08-10). One gated query leaves qcm empty -- that dream's qcm eraser
    removes nothing at this layer, loudly, and the build continues."""
    torch.manual_seed(0)
    state = FakeState([torch.randn(1, 2, 2, 16)])
    queries = [[torch.randn(1, 16)]]

    _, _, bases = dream_bases(queries, gate=[0], wake_state=state, rank_rule="ratio-gap")

    assert bases["qcm"][0].shape[0] == 0
    assert bases["raw"][0].shape[0] == 1
    assert "empty" in capsys.readouterr().out


def _bound_set() -> DreamSetCache:
    cache = _set()
    for dream in cache.dreams:
        dream.token_texts = [f"The code for the {f.entity} is {f.code}." for f in FACTS]
        dream.gate_positions = [0, 1]
    return cache


def test_the_set_report_prints_the_artifact_and_passes_a_bound_set(capsys):
    """Root CLAUDE.md's sanity rule for a dream set: termination reasons,
    per-dream basis sizes, cross-dream V-overlap, the gate's agreement with the
    binding scan, within-dream repeats, and a decoded dream start."""
    report_dream_set(_bound_set(), min_dreams=2, rank_rule="ratio-gap")
    out = capsys.readouterr().out

    for expected in ("termination reasons", "within-dream repeats", "gate vs binding scan",
                     "cross-dream V-overlap", ">>PREFIX>>", ">>FREE>>",
                     "aggregate binding gate PASSED"):
        assert expected in out


def test_the_set_report_refuses_an_underbound_set():
    with pytest.raises(SystemExit):
        report_dream_set(_set(), min_dreams=2, rank_rule="ratio-gap")


def test_the_set_sidecar_carries_the_prefix_the_spectra_and_both_rank_rules(tmp_path):
    path = tmp_path / "dream_set_s1234.txt"
    write_dream_set_sidecar(_bound_set(), path)
    text = path.read_text()

    assert "steer prefix" in text and "gate threshold" in text
    assert "per-layer spectra" in text
    assert "ratio-gap=" in text and "median=" in text
    assert "dreams bound per fact" in text


def test_the_set_builder_passes_cue_splicing_through_to_generation():
    """A dream SET built with --cue-every must splice cues, exactly as the
    single-dream cache does. The set path once accepted the cue flags and
    dropped them on the floor: argparse took them, generation never saw them,
    and the build came out free-running with the coverage to match.
    """
    import inspect

    from dream_sleep import build_dream_set

    source = inspect.getsource(build_dream_set)
    call = source[source.index("teacher_dream(") :]
    call = call[: call.index(")\n")]

    assert "cues=cues" in call
    assert "cue_every=args.cue_every" in call
    assert "cue_greedy=args.cue_greedy" in call


def test_copy_fraction_separates_a_quoting_dream_from_an_original_one():
    """A dream trained on a recall-heavy corpus can degenerate into REPLAYING
    the wake transcript verbatim instead of dreaming about it. Rehearsal
    counting cannot see that -- a verbatim copy scores perfect coverage -- so
    copying needs its own number."""
    from dream_sleep import copy_fraction

    transcript = ("the lighthouse keeper kept meticulous logs of every passing storm "
                  "the bakery on the corner sells out of rye bread before noon").split()
    verbatim = transcript[:14]
    original = "what is the code for the clove i think it was mentioned earlier today".split()

    assert copy_fraction(verbatim, transcript, n=6) == 1.0
    assert copy_fraction(original, transcript, n=6) == 0.0


def test_copy_fraction_counts_only_runs_at_least_n_long():
    from dream_sleep import copy_fraction

    transcript = "alpha beta gamma delta epsilon zeta eta theta".split()
    # A 3-gram overlap is ordinary language reuse, not regurgitation.
    short_overlap = "alpha beta gamma nothing further here at all".split()

    assert copy_fraction(short_overlap, transcript, n=6) == 0.0
    assert copy_fraction(short_overlap, transcript, n=3) > 0.0


def test_copy_fraction_is_zero_for_an_empty_dream():
    from dream_sleep import copy_fraction

    assert copy_fraction([], "a b c".split(), n=3) == 0.0


def test_copy_fraction_ignores_spliced_cue_tokens():
    """Cue splicing injects the wake session's own question phrasing, so those
    tokens match the transcript BY CONSTRUCTION. Counting them as copying
    inflates a cued set's score for a reason that has nothing to do with what
    the model generated."""
    from dream_sleep import copy_fraction

    transcript = "what is the code for the clove the code for the clove is one two".split()
    dream = "what is the code for the clove the code for the clove is one two".split()
    cue = [True] * len(dream)

    assert copy_fraction(dream, transcript, n=6) == 1.0
    assert copy_fraction(dream, transcript, n=6, cue_flags=cue) == 0.0


def test_longest_verbatim_run_separates_regurgitation_from_phrase_reuse():
    """Two different failures wear the same name. A dream that reuses the
    wake's phrasing sentence by sentence can score a high copied-token
    FRACTION while never reproducing more than a line; a dream that replays
    the transcript wholesale is a different object. The run length is what
    distinguishes them."""
    from dream_sleep import copy_fraction, longest_verbatim_run

    transcript = [i for i in range(200)]
    wholesale = transcript[10:150]                       # one long replay
    scattered = (transcript[0:13] + [900 + i for i in range(40)]
                 + transcript[50:63] + [800 + i for i in range(40)])

    assert longest_verbatim_run(wholesale, transcript) == len(wholesale)
    assert longest_verbatim_run(scattered, transcript) == 13
    # the fraction cannot tell them apart nearly as well
    assert copy_fraction(scattered, transcript) > 0.2


def _set_cache(seed, transcript, dreams, **over):
    from dream_sleep import DreamSetCache

    fields = dict(seed=seed, transcript_ids=transcript, wake_state=None, dreams=dreams,
                  distractors={"clove": "9 9 9"}, facts=[("clove", "spice", "1 2 3")],
                  dream_prompt="", generator="abc123")
    fields.update(over)
    return DreamSetCache(**fields)


def _dream(ids):
    return CachedDream(dream_ids=list(ids), token_texts=[str(i) for i in ids],
                       teacher_logits=torch.zeros(len(ids), 4), cue_flags=[False] * len(ids),
                       prefix_len=0, stop_reason="eoc", divergence=[0.0] * len(ids),
                       gate_positions=[], queries=[], spectra=[], ranks={}, bases={})


def test_merging_sets_concatenates_dreams_and_rehashes():
    """100+ dreams on one wake state are generated by several processes with
    disjoint --dream-seed-offset. Training needs them as ONE set: the arms
    share a dream set by registration, and a set's hash is what the summarizer
    checks that sharing against."""
    from dream_sleep import merge_dream_sets

    a = _set_cache(1234, [1, 2, 3], [_dream([10, 11]), _dream([12, 13])])
    b = _set_cache(1234, [1, 2, 3], [_dream([14, 15])])

    merged = merge_dream_sets([a, b])

    assert len(merged.dreams) == 3
    assert [d.dream_ids for d in merged.dreams] == [[10, 11], [12, 13], [14, 15]]
    assert merged.set_sha not in (a.set_sha, b.set_sha)
    assert merged.transcript_ids == a.transcript_ids


def test_merging_refuses_sets_from_different_wake_states():
    """Dreams from a different transcript were generated from a different
    state: pooling them would silently mix two experiments."""
    from dream_sleep import merge_dream_sets

    a = _set_cache(1234, [1, 2, 3], [_dream([10, 11])])
    b = _set_cache(1234, [9, 9, 9], [_dream([12, 13])])

    with pytest.raises(SystemExit, match="transcript"):
        merge_dream_sets([a, b])


def test_merging_refuses_sets_from_different_generators():
    from dream_sleep import merge_dream_sets

    a = _set_cache(1234, [1, 2, 3], [_dream([10, 11])])
    b = _set_cache(1234, [1, 2, 3], [_dream([12, 13])], generator="different")

    with pytest.raises(SystemExit, match="generator"):
        merge_dream_sets([a, b])


def test_merging_refuses_duplicate_dreams():
    """Two processes given the same offset produce byte-identical dreams;
    merging them would double-count a dream as coverage."""
    from dream_sleep import merge_dream_sets

    a = _set_cache(1234, [1, 2, 3], [_dream([10, 11])])
    b = _set_cache(1234, [1, 2, 3], [_dream([10, 11])])

    with pytest.raises(SystemExit, match="identical"):
        merge_dream_sets([a, b])
