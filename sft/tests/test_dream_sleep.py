"""CPU tests for dream_sleep's pure pieces: the frozen-teacher bypass, the
sampling ban, the state erase applied per layer, the paraphrase probes, and
the dream's fact-rehearsal accounting."""

import torch
import torch.nn as nn

from consolidation_null import Fact
from dream_sleep import (
    ARM_CARRY,
    distill_live,
    distill_replay,
    distill_sft,
    distill_stateful,
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
                          drain=False, decode_token=str, needles=[],
                          cues=[[3, 4], [5, 6]], cue_every=6)

    emitted = dream.tokens.tolist()[0]
    windows = [emitted[i:i + 2] for i in range(len(emitted) - 1)]
    assert [3, 4] in windows and [5, 6] in windows


def test_cue_schedule_leaves_the_answer_slot_free_to_come_from_the_state():
    """The cue supplies the question; the tokens after it must still be
    sampled, or the dream would be teacher-forced text rather than recall."""
    model, _, wake, seed = _tiny_setup()

    a = teacher_dream(model, wake, seed, 20, 1.0, (), False, str, [], cues=[[3, 4]], cue_every=5)
    b = teacher_dream(model, wake, seed, 20, 1.0, (), False, str, [], cues=[[3, 4]], cue_every=5)

    assert a.tokens.tolist() != b.tokens.tolist()


def test_cue_greedy_decodes_the_answer_span_deterministically():
    """The digits after a cue are the recall being rehearsed: sampled at
    temperature they almost never come out right, so they decode greedily
    while the rest of the dream stays sampled."""
    model, _, wake, seed = _tiny_setup()

    dream = teacher_dream(model, wake, seed, n_tokens=18, temperature=1.0, banned=(),
                          drain=False, decode_token=str, needles=[],
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


def test_distill_stateful_steps_both_arms_and_leaves_the_wake_state_untouched():
    for drain in (True, False):
        model, opt, wake, seed = _tiny_setup()
        dream = teacher_dream(model, wake, seed, 4, 0.0, (), drain, str, [])
        before = [p.detach().clone() for p in model.parameters()]
        wake_before = wake.ssm_states[0].clone()

        distill_stateful(model, opt, dream, wake, steps=6, kl_temp=1.0, accum=1, drain=drain,
                         on_step=lambda step, loss: None)

        assert _moved(model, before)
        torch.testing.assert_close(wake.ssm_states[0], wake_before)


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


def test_arm_carry_matches_the_registered_sequences():
    assert ARM_CARRY == {
        "replay": "none",
        "drain": "drained",
        "counterfactual": "intact",
        "drain-live": "drained",
        "sft-ref": "none",
        "no-sleep": "intact",
    }
