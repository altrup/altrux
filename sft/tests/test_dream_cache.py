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
    DreamCache,
    binding_coverage,
    build_distractors,
    dream_seed_text,
    dream_sidecar_text,
    load_dream_cache,
    save_dream_cache,
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
