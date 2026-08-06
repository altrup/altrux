"""CPU tests for dream_sleep's pure pieces: the frozen-teacher bypass, the
sampling ban, the state erase applied per layer, the paraphrase probes, and
the dream's fact-rehearsal accounting."""

import pytest
import torch
import torch.nn as nn

from consolidation_null import Fact
from dream_sleep import (
    ARM_CARRY,
    distill_counterfactual,
    distill_live,
    distill_replay,
    distill_sft,
    erase_state,
    frozen_teacher,
    paraphrase_prompts,
    rehearsal_fraction,
    sample_next,
    teacher_dream,
)
from erase_probe import deflate, state_top_dirs
from lora import LoRALinear

USER, ASST = "[USER]", "[ASSISTANT]"


class FakeState:
    """Stands in for MixerState -- erase_state only touches ssm_states."""

    def __init__(self, ssm_states):
        self.ssm_states = ssm_states


def read(ssm_state: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhpn,bn->bhp", ssm_state.float(), c.float())


def test_frozen_teacher_bypasses_the_adapters_and_restores_them():
    base = nn.Linear(4, 4)
    layer = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
    with torch.no_grad():
        layer.lora_B += 0.5
    x = torch.randn(1, 4)
    trained = layer(x)

    with frozen_teacher(nn.Sequential(layer)):
        torch.testing.assert_close(layer(x), base(x))
        torch.testing.assert_close(layer.weight, base.weight)

    torch.testing.assert_close(layer(x), trained)


def test_frozen_teacher_zeroes_and_restores_the_marker_delta():
    model = nn.Module()
    model.marker_delta = nn.Module()
    model.marker_delta.delta = nn.Parameter(torch.full((2, 3), 0.25))

    with frozen_teacher(model):
        assert model.marker_delta.delta.abs().max().item() == 0.0

    assert model.marker_delta.delta.abs().max().item() == 0.25


def test_sample_next_never_emits_a_banned_token():
    logits = torch.full((1, 5), -20.0)
    logits[0, 3] = 20.0  # EOS would dominate every sample
    for _ in range(20):
        assert sample_next(logits, temperature=1.0, banned=(3,)).item() != 3


def test_sample_next_is_greedy_at_zero_temperature():
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    assert sample_next(logits, temperature=0.0, banned=()).item() == 1


def test_erase_state_zeroes_each_layer_s_own_deflated_read():
    torch.manual_seed(0)
    state = FakeState([torch.randn(1, 2, 4, 8) for _ in range(3)])
    queries = [torch.randn(1, 8) for _ in range(3)]
    directions = [deflate(c, state_top_dirs(s, 1)) for s, c in zip(state.ssm_states, queries, strict=True)]

    erase_state(state, queries)

    for ssm, direction in zip(state.ssm_states, directions, strict=True):
        assert read(ssm, direction).abs().max().item() < 1e-4


def test_erase_state_leaves_the_state_s_dominant_direction_alone():
    # Deflation (k=1) protects the shared interference cone -- the direction
    # the state concentrates its energy in.
    torch.manual_seed(1)
    cone = torch.nn.functional.normalize(torch.randn(8), dim=0)
    values = torch.randn(2, 4, 1)
    ssm = (values * cone).unsqueeze(0) + 0.01 * torch.randn(1, 2, 4, 8)
    state = FakeState([ssm.clone()])

    erase_state(state, [cone.unsqueeze(0) + 0.05 * torch.randn(1, 8)])

    before, after = read(ssm, cone.unsqueeze(0)), read(state.ssm_states[0], cone.unsqueeze(0))
    torch.testing.assert_close(after, before, rtol=1e-2, atol=1e-2)


def test_erase_state_skips_directions_that_are_pure_cone():
    torch.manual_seed(2)
    cone = torch.nn.functional.normalize(torch.randn(8), dim=0)
    ssm = (torch.randn(2, 4, 1) * cone).unsqueeze(0)
    state = FakeState([ssm.clone()])

    skipped = erase_state(state, [cone.unsqueeze(0)])

    assert skipped == 1
    torch.testing.assert_close(state.ssm_states[0], ssm)


def test_paraphrase_prompts_reword_the_question_but_keep_the_answer_stem():
    fact = Fact("osprey", "bird", "1 2 3 4 5")
    prompts = paraphrase_prompts(fact, USER, ASST)

    assert len(prompts) == 4 and len(set(prompts)) == 4
    for prompt in prompts:
        assert fact.entity in prompt
        assert prompt.endswith(f"{ASST} The code for the {fact.entity} is")
        assert "What is the code for the osprey?" not in prompt  # the trained phrasing


def test_rehearsal_fraction_counts_tokens_covering_fact_content():
    tokens = ["The", " code", " for", " the", " osprey", " is", " 1", " 2"]
    fraction, counts = rehearsal_fraction(tokens, ["osprey", "1 2"])

    assert counts == {"osprey": 1, "1 2": 1}
    assert fraction == 3 / 8  # " osprey", " 1", " 2"


def test_rehearsal_fraction_is_zero_for_a_wandering_dream():
    fraction, counts = rehearsal_fraction(["the", " weather", " is", " mild"], ["osprey", "1 2"])
    assert fraction == 0.0 and counts == {"osprey": 0, "1 2": 0}


class TinyModel(nn.Module):
    """Minimal stand-in for the 780M wrapper: one trainable head, a per-layer
    additive `ssm_state`, and the same `c_capture` / `(logits, state)` contract
    the sleep loops drive. Enough to exercise the loops' wiring on CPU."""

    def __init__(self, vocab: int = 7, dim: int = 4, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)
        self.layers = list(range(n_layers))
        self.c_capture = None
        self.erase_hook = None
        self.marker_delta = None

    def forward(self, ids, state=None):
        dim = self.embedding.embedding_dim
        if state is None:
            state = FakeState([torch.zeros(1, 1, 1, dim) for _ in self.layers])
        out = []
        for t in range(ids.shape[1]):
            h = self.embedding(ids[:, t])
            for i in range(len(self.layers)):
                if self.c_capture is not None:
                    self.c_capture.append(h.detach())
                if self.erase_hook is not None:
                    state.ssm_states[i] = self.erase_hook(i, state.ssm_states[i], h)
                state.ssm_states[i] = state.ssm_states[i] + h.view(1, 1, 1, dim)
            read = torch.einsum("bhpn,bn->bn", state.ssm_states[0], h)
            out.append(self.head(h + read))
        return torch.stack(out, dim=1), state


def _detachable(state: FakeState) -> FakeState:
    return FakeState([s.detach() for s in state.ssm_states])


FakeState.detach = _detachable


def _tiny_setup():
    torch.manual_seed(0)
    model = TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    wake = FakeState([torch.randn(1, 1, 1, 4) for _ in model.layers])
    seed = torch.tensor([[1]])
    return model, opt, wake, seed


def _sentences(token_id: int) -> str:
    """Decoded text where every token ends a sentence, so a fired cue timer
    splices at the next token rather than waiting for a boundary."""
    return f"{token_id}."


def _moved(model, before) -> bool:
    return any(not torch.equal(p, b) for p, b in zip(model.parameters(), before, strict=True))


def test_teacher_dream_caches_a_position_per_token_and_bans_eos():
    model, _, wake, seed = _tiny_setup()

    dream = teacher_dream(model, wake, seed, n_tokens=8, temperature=1.0, banned=(0,),
                          drain=False, decode_token=lambda i: f"<{i}>", needles=["<1>"])

    assert dream.tokens.shape == (1, 8)
    assert dream.logits.shape[0] == 8
    assert all(len(q) == len(model.layers) for q in dream.queries) and len(dream.queries) == 8
    assert 0 not in dream.tokens.tolist()[0]
    assert len(dream.token_texts) == 8


def test_cue_schedule_forces_every_cue_into_the_dream_in_rotation():
    model, _, wake, seed = _tiny_setup()

    dream = teacher_dream(model, wake, seed, n_tokens=24, temperature=1.0, banned=(),
                          drain=False, decode_token=_sentences, needles=[],
                          cues=[[3, 4], [5, 6]], cue_every=6)

    emitted = dream.tokens.tolist()[0]
    windows = [emitted[i:i + 2] for i in range(len(emitted) - 1)]
    assert [3, 4] in windows and [5, 6] in windows


def test_cue_schedule_leaves_the_answer_slot_free_to_come_from_the_state():
    """The cue supplies the question; the tokens after it must still be
    sampled, or the dream would be teacher-forced text rather than recall."""
    model, _, wake, seed = _tiny_setup()

    a = teacher_dream(model, wake, seed, 20, 1.0, (), False, _sentences, [], cues=[[3, 4]], cue_every=5)
    b = teacher_dream(model, wake, seed, 20, 1.0, (), False, _sentences, [], cues=[[3, 4]], cue_every=5)

    assert a.tokens.tolist() != b.tokens.tolist()


def test_cue_greedy_decodes_the_answer_span_deterministically():
    """The digits after a cue are the recall being rehearsed: sampled at
    temperature they almost never come out right, so they decode greedily
    while the rest of the dream stays sampled."""
    model, _, wake, seed = _tiny_setup()

    dream = teacher_dream(model, wake, seed, n_tokens=18, temperature=1.0, banned=(),
                          drain=False, decode_token=_sentences, needles=[],
                          cues=[[3, 4]], cue_every=4, cue_greedy=3)

    emitted = dream.tokens.tolist()[0]
    start = next(j for j in range(len(emitted) - 1) if emitted[j:j + 2] == [3, 4]) + 2
    span = range(start, min(start + 3, len(emitted)))
    assert len(span) > 0
    for j in span:
        assert emitted[j] == int(dream.logits[j - 1].argmax())


def test_no_cues_leaves_generation_untouched():
    model, _, wake, seed = _tiny_setup()
    kwargs = dict(seed_ids=seed, n_tokens=8, temperature=0.0, banned=(), drain=False,
                  decode_token=str, needles=[])

    torch.manual_seed(0)
    plain = teacher_dream(model, wake, **kwargs)
    torch.manual_seed(0)
    empty = teacher_dream(model, wake, cues=[], cue_every=4, **kwargs)

    assert plain.tokens.tolist() == empty.tokens.tolist()


def test_teacher_dream_drains_the_state_it_generates_from():
    model, _, wake, seed = _tiny_setup()
    kwargs = dict(seed_ids=seed, n_tokens=6, temperature=0.0, banned=(), decode_token=str, needles=[])

    intact = teacher_dream(model, wake, drain=False, **kwargs)
    drained = teacher_dream(model, wake, drain=True, **kwargs)

    assert not torch.allclose(intact.final_state.ssm_states[0], drained.final_state.ssm_states[0])


def test_distill_replay_and_sft_take_their_optimizer_steps():
    model, opt, wake, seed = _tiny_setup()
    dream = teacher_dream(model, wake, seed, 6, 0.0, (), False, str, [])
    before = [p.detach().clone() for p in model.parameters()]

    distill_replay(model, opt, dream, steps=4, chunk_len=3, kl_temp=1.0, on_step=lambda s, l: None)
    assert _moved(model, before)

    before = [p.detach().clone() for p in model.parameters()]
    distill_sft(model, opt, torch.tensor([[1, 2, 3, 4, 5, 6]]), steps=3, chunk_len=2, on_step=lambda s, l: None)
    assert _moved(model, before)


def test_distill_live_trains_online_and_returns_the_drained_state():
    model, opt, wake, seed = _tiny_setup()
    before = [p.detach().clone() for p in model.parameters()]
    wake_before = wake.ssm_states[0].clone()

    state, dream = distill_live(model, opt, wake, seed, n_tokens=5, temperature=1.0, banned=(0,),
                                kl_temp=1.0, accum=1, decode_token=str, needles=[],
                                on_step=lambda s, l: None)

    assert _moved(model, before)
    assert dream.tokens.shape == (1, 5) and state is dream.final_state
    torch.testing.assert_close(wake.ssm_states[0], wake_before)  # the arm works on a copy


def _fixed_dream(model, wake, seed, n_tokens=4):
    return teacher_dream(model, wake, seed, n_tokens, 0.0, (), False, str, [])


def _counterfactual_run(in_place: bool, steps: int = 1, deep: bool = False):
    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed)
    losses: list[float] = []
    distill_counterfactual(model, opt, dream, wake, steps=steps, kl_temp=1.0, accum=1,
                           in_place=in_place, deep=deep, on_step=lambda s, l: losses.append(l))
    return [p.detach().clone() for p in model.parameters()], losses


def test_b1_and_b2_are_identical_at_token_one_including_gradients():
    """Same state, same query, same ablation, same logit -- so the same
    optimizer step. They may only diverge from token 2, through the carry."""
    b1_params, b1_losses = _counterfactual_run(in_place=True)
    b2_params, b2_losses = _counterfactual_run(in_place=False)

    assert b1_losses == b2_losses
    for a, b in zip(b1_params, b2_params, strict=True):
        assert torch.equal(a, b)


def test_b1_and_b2_diverge_once_the_carry_matters():
    b1_params, _ = _counterfactual_run(in_place=True, steps=4)
    b2_params, _ = _counterfactual_run(in_place=False, steps=4)

    assert any(not torch.equal(a, b) for a, b in zip(b1_params, b2_params, strict=True))


def test_counterfactual_ablates_through_the_model_s_erase_hook():
    """The erase is per-layer and interleaved inside the forward (sec 3's
    micro-order), so it must arrive via the hook rather than a wrapper."""
    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed)
    seen: list[int] = []
    original = TinyModel.forward

    def spy(self, ids, state=None):
        if self.erase_hook is not None:
            seen.append(1)
        return original(self, ids, state=state)

    TinyModel.forward = spy
    try:
        distill_counterfactual(model, opt, dream, wake, steps=2, kl_temp=1.0, accum=1,
                               in_place=False, deep=False, on_step=lambda s, l: None)
    finally:
        TinyModel.forward = original
    assert seen and model.erase_hook is None  # always unset again


def test_counterfactual_leaves_the_wake_state_untouched():
    for in_place in (True, False):
        model, opt, wake, seed = _tiny_setup()
        dream = _fixed_dream(model, wake, seed)
        before = wake.ssm_states[0].clone()
        before_params = [p.detach().clone() for p in model.parameters()]

        distill_counterfactual(model, opt, dream, wake, steps=5, kl_temp=1.0, accum=1,
                               in_place=in_place, deep=False, on_step=lambda s, l: None)

        torch.testing.assert_close(wake.ssm_states[0], before)
        assert _moved(model, before_params)


def test_counterfactual_reports_one_token_gradient_per_step():
    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed)
    tokens, carried = distill_counterfactual(model, opt, dream, wake, steps=6, kl_temp=1.0, accum=1,
                                             in_place=False, deep=False, on_step=lambda s, l: None)
    assert tokens == 6 and carried is not None


def test_deep_b2_takes_one_optimizer_step_per_pass():
    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed)
    steps: list[int] = []

    distill_counterfactual(model, opt, dream, wake, steps=8, kl_temp=1.0, accum=1,
                           in_place=False, deep=True, on_step=lambda s, l: steps.append(s))

    assert len(steps) == 2  # two passes over a 4-token dream, one step each


def test_deep_mode_is_not_available_for_b1():
    """Sec 6: deep-B1's cross-token gradient passes through every ablation
    projector, so its depth is pre-aimed at the compensation harbor."""
    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed)
    with pytest.raises(ValueError):
        distill_counterfactual(model, opt, dream, wake, steps=2, kl_temp=1.0, accum=1,
                               in_place=True, deep=True, on_step=lambda s, l: None)


def test_replay_can_reset_the_state_before_every_chunk():
    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed, n_tokens=6)

    carried, _ = _replay_params(model, opt, dream, fresh_state=False)
    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed, n_tokens=6)
    fresh, _ = _replay_params(model, opt, dream, fresh_state=True)

    assert any(not torch.equal(a, b) for a, b in zip(carried, fresh, strict=True))


def _replay_params(model, opt, dream, **kwargs):
    tokens = distill_replay(model, opt, dream, steps=4, chunk_len=3, kl_temp=1.0,
                            on_step=lambda s, l: None, **kwargs)
    return [p.detach().clone() for p in model.parameters()], tokens


def test_replay_masks_cue_targets_out_of_the_loss():
    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed, n_tokens=6)
    keep = [True, False, True, True, True, True]

    _, tokens = _replay_params(model, opt, dream, keep=keep)

    assert tokens == 10  # chunks of 3 over 6 tokens: 2 kept then 3 kept, twice


def test_a_fully_masked_chunk_takes_no_optimizer_step():
    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed, n_tokens=6)
    before = [p.detach().clone() for p in model.parameters()]

    tokens = distill_replay(model, opt, dream, steps=1, chunk_len=6, kl_temp=1.0,
                            on_step=lambda s, l: None, keep=[False] * 6)

    assert tokens == 0 and not _moved(model, before)


def test_ce_on_dream_trains_on_the_dream_tokens_instead_of_the_teacher_logits():
    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed, n_tokens=6)
    kl, _ = _replay_params(model, opt, dream)
    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed, n_tokens=6)
    ce, _ = _replay_params(model, opt, dream, ce=True)

    assert any(not torch.equal(a, b) for a, b in zip(kl, ce, strict=True))


def test_arm_carry_matches_the_registered_sequences():
    assert ARM_CARRY == {
        "replay": "none",
        "ce-on-dream": "none",
        "drain": "drained",
        "counterfactual": "intact",
        "drain-live": "drained",
        "sft-ref": "none",
        "no-sleep": "intact",
    }


class _FakeTokenizer:
    eos_token_id = 0


def _fake_io():
    def encode(text: str) -> torch.Tensor:
        return torch.tensor([[1 + (sum(map(ord, word)) % 5) for word in text.split()][:4] or [1]])

    def decode(ids) -> str:
        return "".join(f"{int(i)}." for i in ids)

    return encode, decode


def _sleep_args(**overrides):
    from types import SimpleNamespace

    args = dict(seed=1234, dream_tokens=8, dream_temp=0.0, dream_prompt="", cue_every=3, cue_greedy=1,
                chunk_len=None, distill_steps=4, kl_temp=1.0, accum_window=1, probe_every=0,
                fresh_state_replay=False, deep=False, ce_on_dream=False, n_facts=2, filler_tokens=4)
    return SimpleNamespace(**{**args, **overrides})


def _built_cache(tmp_path, args):
    from dream_sleep import build_cache

    model, opt, wake, seed = _tiny_setup()
    encode, decode = _fake_io()
    facts = [Fact("osprey", "bird", "1 2 3 4 5"), Fact("heron", "bird", "5 9 7 9 7")]
    transcript = torch.tensor([[1, 2, 3, 4, 5, 6]])
    cache = build_cache(model, args, tmp_path / "dream_cache_s1234.pt", transcript, facts, 3,
                        encode, decode, _FakeTokenizer(), USER, ASST)
    return model, opt, wake, cache, facts, transcript, encode, decode


def test_build_cache_writes_a_loadable_cache_and_its_sidecar(tmp_path):
    from dream_sleep import load_dream_cache

    args = _sleep_args()
    _, _, _, cache, facts, transcript, _, _ = _built_cache(tmp_path, args)

    loaded = load_dream_cache(tmp_path / "dream_cache_s1234.pt")
    assert loaded.dream_ids == cache.dream_ids and len(loaded.dream_ids) == args.dream_tokens
    assert loaded.transcript_ids == [1, 2, 3, 4, 5, 6]
    assert set(loaded.distractors) == {f.entity for f in facts}
    assert len(loaded.cue_flags) == args.dream_tokens and any(loaded.cue_flags)
    sidecar = (tmp_path / "dream_s1234.txt").read_text()
    assert "[CUE]" in sidecar and cache.dream_sha in sidecar


@pytest.mark.parametrize("arm", ["replay", "ce-on-dream", "drain", "counterfactual", "sft-ref"])
def test_run_sleep_trains_every_arm_from_the_cached_dream(tmp_path, arm):
    from dream_sleep import run_sleep

    args = _sleep_args(ce_on_dream=(arm == "ce-on-dream"))
    model, opt, wake, cache, facts, transcript, encode, decode = _built_cache(tmp_path, args)
    before = [p.detach().clone() for p in model.parameters()]
    records: list[dict] = []

    carried = run_sleep(arm, model, opt, args, 1, wake, transcript, [(1, f) for f in facts], 3, cache,
                        encode, decode, _FakeTokenizer(), USER, ASST, records.append,
                        lambda step: records.append({"phase": "periodic", "step": step}))

    assert _moved(model, before)
    sleep = next(r for r in records if r["phase"] == "sleep")
    assert sleep["token_gradients"] > 0
    assert (carried is None) == (ARM_CARRY[arm] == "none")
    if arm != "sft-ref":
        dream = next(r for r in records if r["phase"] == "dream")
        assert "bound_cov" in dream and "misbound" in dream


def test_run_sleep_streams_the_probe_battery_on_the_registered_cadence(tmp_path):
    from dream_sleep import run_sleep

    args = _sleep_args(distill_steps=6, probe_every=2)
    model, opt, wake, cache, facts, transcript, encode, decode = _built_cache(tmp_path, args)
    probes: list[int] = []

    run_sleep("counterfactual", model, opt, args, 1, wake, transcript, [(1, f) for f in facts], 3, cache,
              encode, decode, _FakeTokenizer(), USER, ASST, lambda r: None, probes.append)

    assert probes == [2, 4]  # not at the last step -- the end-of-sleep battery covers that
