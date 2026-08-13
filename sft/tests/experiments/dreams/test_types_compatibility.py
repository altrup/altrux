import pickle
import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[3]))

import dream_sleep
from experiments.dreams import types


def test_dream_sleep_reexports_cache_types():
    for name in ("Dream", "DreamCache", "CachedDream", "DreamSetCache", "dream_set_sha", "token_sha"):
        assert getattr(dream_sleep, name) is getattr(types, name)


def test_cache_type_pickle_round_trip_keeps_the_new_global():
    cache = types.DreamCache(
        seed=1,
        transcript_ids=[1, 2],
        dream_ids=[3, 4],
        wake_state=None,
        teacher_logits=torch.zeros(2, 3),
        queries=[],
        token_texts=["a", "b"],
        cue_flags=[False, False],
        distractors={},
        facts=[],
    )

    loaded = pickle.loads(pickle.dumps(cache))

    assert type(loaded) is types.DreamCache
    assert loaded.transcript_sha == types.token_sha([1, 2])
    assert loaded.dream_sha == types.token_sha([3, 4])
