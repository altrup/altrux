"""CPU tests for erase_probe's pure pieces: the rank-1 erase math and the
capture bookkeeping."""

import torch

from experiments.erasure.operators import deflate, rank1_erase, state_top_dirs
from experiments.erasure.probe import group_by_layer


def read(ssm_state: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhpn,bn->bhp", ssm_state.float(), c.float())


def test_full_erase_zeroes_the_same_read():
    torch.manual_seed(0)
    s = torch.randn(1, 4, 8, 16)
    c = torch.randn(1, 16)
    erased = rank1_erase(s, c, gamma=1.0)
    assert read(erased, c).abs().max().item() < 1e-5


def test_partial_erase_attenuates_by_one_minus_gamma():
    torch.manual_seed(1)
    s = torch.randn(1, 4, 8, 16)
    c = torch.randn(1, 16)
    erased = rank1_erase(s, c, gamma=0.25)
    torch.testing.assert_close(read(erased, c), 0.75 * read(s, c), rtol=1e-5, atol=1e-5)


def test_orthogonal_read_is_untouched():
    torch.manual_seed(2)
    s = torch.randn(1, 4, 8, 16)
    c = torch.randn(1, 16)
    other = torch.randn(1, 16)
    other = other - (other * c).sum() / (c * c).sum() * c
    erased = rank1_erase(s, c, gamma=1.0)
    torch.testing.assert_close(read(erased, other), read(s, other), rtol=1e-5, atol=1e-5)


def test_erase_preserves_dtype_and_shape():
    s = torch.randn(1, 4, 8, 16).to(torch.bfloat16)
    c = torch.randn(1, 16).to(torch.bfloat16)
    erased = rank1_erase(s, c, gamma=0.5)
    assert erased.dtype == s.dtype and erased.shape == s.shape


def test_zero_query_is_a_noop():
    s = torch.randn(1, 4, 8, 16)
    c = torch.zeros(1, 16)
    torch.testing.assert_close(rank1_erase(s, c, gamma=1.0), s)


def test_deflate_removes_basis_components():
    torch.manual_seed(3)
    basis = torch.linalg.qr(torch.randn(16, 2))[0].T  # (2, 16) orthonormal rows
    c = torch.randn(1, 16)
    out = deflate(c, basis)
    assert torch.einsum("bn,kn->bk", out, basis).abs().max().item() < 1e-5


def test_erase_along_deflated_direction_spares_basis_reads():
    torch.manual_seed(4)
    s = torch.randn(1, 4, 8, 16)
    basis = torch.linalg.qr(torch.randn(16, 1))[0].T
    c = torch.randn(1, 16)
    erased = rank1_erase(s, deflate(c, basis), gamma=1.0)
    torch.testing.assert_close(read(erased, basis[:1]), read(s, basis[:1]), rtol=1e-4, atol=1e-5)


def test_state_top_dirs_finds_the_dominant_key():
    torch.manual_seed(5)
    key = torch.nn.functional.normalize(torch.randn(16), dim=0)
    vals = torch.randn(4, 8, 1)
    s = (vals * key).unsqueeze(0) + 0.01 * torch.randn(1, 4, 8, 16)
    top = state_top_dirs(s, k=1)
    assert top.shape == (1, 1, 16)  # one basis per batch element
    assert torch.nn.functional.cosine_similarity(top[0, 0], key, dim=0).abs().item() > 0.99


def test_state_top_dirs_gives_each_batch_element_its_own_basis():
    torch.manual_seed(6)
    keys = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
    s = (torch.randn(2, 4, 8, 1) * keys.view(2, 1, 1, 16)) + 0.01 * torch.randn(2, 4, 8, 16)

    top = state_top_dirs(s, k=1)

    for b in range(2):
        assert torch.nn.functional.cosine_similarity(top[b, 0], keys[b], dim=0).abs().item() > 0.99


def test_group_by_layer_orders_capture_token_major():
    # _mixer_step appends one C per layer per token: [t0l0, t0l1, t1l0, t1l1]
    flat = ["t0l0", "t0l1", "t1l0", "t1l1"]
    assert group_by_layer(flat, n_layers=2) == [["t0l0", "t0l1"], ["t1l0", "t1l1"]]


def test_group_by_layer_rejects_partial_tokens():
    try:
        group_by_layer(["a", "b", "c"], n_layers=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError on a capture not divisible by n_layers")


def test_the_erase_direction_carries_gradient_back_to_the_query():
    """DISCUSSION-20260807 sec 3.4: the cut follows the query, so the direction
    is differentiable -- a detached one makes the loss landscape treat the cut
    as fixed and points the read gradient at dodging via remnants."""
    s = torch.randn(1, 2, 4, 8, requires_grad=True)
    c = torch.randn(1, 8, requires_grad=True)

    rank1_erase(s, c, gamma=1.0).float().sum().backward()

    assert c.grad is not None and c.grad.abs().max().item() > 0
    assert s.grad is not None


def test_the_deflation_basis_is_detached_from_the_state_it_is_read_from():
    s = torch.randn(1, 2, 4, 8, requires_grad=True)
    c = torch.randn(1, 8, requires_grad=True)

    basis = state_top_dirs(s, 1)
    assert not basis.requires_grad
    assert deflate(c, basis).requires_grad  # through c only


# --- the rich wake transcript (DISCUSSION-20260808 sec 2.10.11) --------------

import random

import pytest

from experiments.erasure.wake_items import (
    build_bystanders,
    build_dialogue,
    build_mixed_turns,
    build_wake_items,
    collisions,
)
from experiments.facts import Fact

USER, ASST = "[USER]", "[ASSISTANT]"
FACTS = [Fact("osprey", "bird", "1 2 3 4 5"), Fact("heron", "bird", "5 9 7 9 7")]
BATTERY = ["Paris", "Tokyo", "Tuesday", "honey"]


def test_bystander_pools_drop_the_values_they_are_told_to_avoid():
    plain = build_bystanders(6, random.Random(0), USER, ASST)
    assert any("Paris" in b.statement for b in plain)  # sanity: the pool has it

    avoided = build_bystanders(6, random.Random(0), USER, ASST, avoid=BATTERY)
    for b in avoided:
        assert not collisions(f"{b.question} {b.statement}", FACTS, BATTERY)


def test_collisions_names_every_overlapping_string():
    assert collisions("The parcel left Paris on Tuesday.", FACTS, BATTERY) == ["Paris", "Tuesday"]
    assert collisions("The osprey was seen at dawn.", FACTS, BATTERY) == ["osprey"]
    assert collisions("The reading was 1 2 3 4 5 exactly.", FACTS, BATTERY) == ["1 2 3 4 5"]
    assert collisions("A quiet afternoon by the water.", FACTS, BATTERY) == []


def test_dialogue_slice_skips_conversations_that_collide():
    records = [
        {
            "messages": [
                {"role": "user", "content": "Where is the Louvre?"},
                {"role": "assistant", "content": "It is in Paris."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How do I store bread?"},
                {"role": "assistant", "content": "Keep it in a cloth bag."},
            ]
        },
    ]
    items = build_dialogue(records, 1, random.Random(0), USER, ASST, FACTS, BATTERY)

    assert len(items) == 1
    assert items[0].cls == "dialogue"
    assert "bread" in items[0].question
    assert items[0].prompt.startswith(USER) and ASST in items[0].prompt


def test_the_dialogue_slice_refuses_rather_than_under_deliver():
    records = [
        {
            "messages": [
                {"role": "user", "content": "Where is the Louvre?"},
                {"role": "assistant", "content": "It is in Paris."},
            ]
        }
    ]
    with pytest.raises(SystemExit):
        build_dialogue(records, 1, random.Random(0), USER, ASST, FACTS, BATTERY)


def test_wake_items_carry_facts_and_distractors_and_never_collide():
    records = [
        {
            "messages": [
                {"role": "user", "content": f"How do I store bread {i}?"},
                {"role": "assistant", "content": f"Keep it in a cloth bag {i}."},
            ]
        }
        for i in range(20)
    ]
    items, distractors = build_wake_items(
        FACTS,
        bystanders=3,
        nearcone=2,
        dialogue=2,
        dialogue_records=records,
        rng=random.Random(0),
        user_open=USER,
        asst_open=ASST,
        battery_answers=BATTERY,
    )

    assert len(items) == len(FACTS) + 7
    assert len(distractors) == 7
    assert [i for i in items if isinstance(i, Fact)] == FACTS or set(FACTS) <= set(items)
    for d in distractors:
        assert not collisions(f"{d.question} {d.statement}", FACTS, BATTERY)

    turns = build_mixed_turns(items, 4, lambda s: len(s.split()), random.Random(0))
    text = " ".join(t for _, t in turns)
    for fact in FACTS:
        assert text.count(fact.code) == 1


def test_a_colliding_distractor_stops_the_build():
    from experiments.erasure.wake_items import Bystander, assert_no_collisions

    bad = Bystander("x", "Where is it?", "The parcel left Paris.", "p", " Paris")
    with pytest.raises(SystemExit):
        assert_no_collisions([bad], FACTS, BATTERY)
    assert_no_collisions([], FACTS, BATTERY)


def test_state_top_dirs_is_memoised_per_state_and_recomputed_after_mutation(monkeypatch):
    """The wake state is fixed across a whole gate sweep, but its top
    directions were re-derived per dream per scheme row -- a (h*p, n) SVD per
    layer, which was most of the sweep's runtime. The cache must still notice
    an in-place edit, or a multi-sleep caller would silently reuse the previous
    sleep's directions."""
    import torch

    from experiments.erasure import operators
    from experiments.erasure.operators import state_top_dirs

    calls = []
    real_svd = torch.linalg.svd

    def counting_svd(*args, **kwargs):
        calls.append(1)
        return real_svd(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counting_svd)
    operators.clear_state_top_dirs_cache()

    torch.manual_seed(0)
    state = torch.randn(1, 4, 3, 8)

    first = state_top_dirs(state, 1)
    assert len(calls) == 1
    again = state_top_dirs(state, 1)
    assert len(calls) == 1  # served from the cache
    assert torch.equal(first, again)

    with torch.no_grad():
        state[0, 0, 0, :] = torch.tensor([5.0, 0, 0, 0, 0, 0, 0, 0])
    after = state_top_dirs(state, 1)

    assert len(calls) == 2  # an in-place edit invalidates
    assert not torch.equal(first, after)
