"""CPU tests for dream_sleep's pure pieces: the frozen-teacher bypass, dream
termination, the state erase applied per layer, the paraphrase probes, and
the dream's fact-rehearsal accounting."""

import pytest
import torch
import torch.nn as nn

from consolidation_null import Fact
from dream_sleep import (
    ARM_CARRY,
    CachedDream,
    distill_dream_set,
    distill_counterfactual,
    distill_live,
    distill_replay,
    distill_sft,
    erase_ssm,
    erased_start,
    fact_read_positions,
    gate_agreement,
    load_dream_cache,
    pilot_path,
    sidecar_path,
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


def test_sample_next_is_greedy_at_zero_temperature():
    logits = torch.tensor([[0.1, 5.0, 0.2]])
    assert sample_next(logits, temperature=0.0).item() == 1


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


@pytest.mark.parametrize("op", ["raw", "deflated"])
def test_the_erase_gradient_reaches_the_query_projection(op):
    """Sec 3.4: the gradient runs through the ablation direction's query path.
    The state here is a constant, so the projection's only path to the loss is
    the direction it produces."""
    torch.manual_seed(0)
    proj = nn.Linear(8, 8, bias=False)
    ssm = torch.randn(1, 2, 4, 8)
    c = proj(torch.randn(1, 8))

    erased, _ = erase_ssm(ssm, c, op=op)
    erased.float().pow(2).sum().backward()

    assert proj.weight.grad is not None
    assert proj.weight.grad.abs().max().item() > 0


@pytest.mark.parametrize("op", ["raw", "deflated"])
def test_the_erase_never_differentiates_the_protected_subspace(op):
    """v is detached at its source, so no arm -- however deep its BPTT -- can
    rotate the protected subspace onto the fact it is meant to guard."""
    ssm = torch.randn(1, 2, 4, 8, requires_grad=True)
    v = state_top_dirs(ssm, 1)

    assert not v.requires_grad and v.grad_fn is None
    assert not deflate(torch.randn(1, 8), v).requires_grad


def test_the_raw_erase_zeroes_the_read_along_the_query_itself():
    torch.manual_seed(3)
    state = FakeState([torch.randn(1, 2, 4, 8) for _ in range(3)])
    queries = [torch.randn(1, 8) for _ in range(3)]

    skipped = erase_state(state, queries, op="raw")

    assert skipped == 0
    for ssm, c in zip(state.ssm_states, queries, strict=True):
        assert read(ssm, c).abs().max().item() < 1e-4


def test_the_raw_erase_cuts_a_pure_cone_query_the_deflated_one_skips():
    torch.manual_seed(2)
    cone = torch.nn.functional.normalize(torch.randn(8), dim=0)
    ssm = (torch.randn(2, 4, 1) * cone).unsqueeze(0)
    state = FakeState([ssm.clone()])

    skipped = erase_state(state, [cone.unsqueeze(0)], op="raw")

    assert skipped == 0
    assert read(state.ssm_states[0], cone.unsqueeze(0)).abs().max().item() < 1e-4


def test_every_result_record_carries_the_erase_operator():
    import io
    import json

    from dream_sleep import make_emit

    buf = io.StringIO()
    make_emit(buf, erase_op="raw")({"phase": "sleep", "arm": "drain"})

    record = json.loads(buf.getvalue())
    assert record["erase_op"] == "raw" and record["phase"] == "sleep"


def test_the_erase_operator_defaults_to_the_registered_deflated_form():
    from dream_sleep import build_parser

    assert build_parser().parse_args([]).erase_op == "deflated"
    assert build_parser().parse_args(["--erase-op", "raw"]).erase_op == "raw"


def _adapter_model(rank: int = 2, alpha: float = 4.0) -> nn.Module:
    from lora import apply_lora

    model = nn.Sequential(nn.Linear(4, 4))
    model.marker_delta = nn.Module()
    model.marker_delta.delta = nn.Parameter(torch.randn(2, 4))
    return apply_lora(model, ["0"], rank=rank, alpha=alpha, dropout=0.0)


def _write_checkpoint(monkeypatch, tmp_path, model, rank: int, alpha: float):
    """A real train.py checkpoint directory (trainable.pt + lora_config.json),
    which is what --init-adapter takes."""
    import train

    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    return train.save_checkpoint(
        model, optimizer, step=1, epoch=0, slots=[None], next_ptr=0,
        total_tokens=1.0, lora_rank=rank, lora_alpha=alpha,
    )


def _trainable(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def test_a_train_checkpoint_round_trips_into_a_freshly_initialised_model(monkeypatch, tmp_path):
    from dream_sleep import load_init_adapter

    torch.manual_seed(0)
    trained = _adapter_model()
    with torch.no_grad():
        for p in trained.parameters():
            p += 0.5
    ckpt = _write_checkpoint(monkeypatch, tmp_path, trained, rank=2, alpha=4.0)

    torch.manual_seed(1)
    fresh = _adapter_model()
    expected = _trainable(trained)
    assert any(not torch.equal(v, expected[k]) for k, v in _trainable(fresh).items())

    sha = load_init_adapter(fresh, ckpt, rank=2, alpha=4.0)

    for name, tensor in _trainable(fresh).items():
        torch.testing.assert_close(tensor, expected[name])
    from dream_sleep import file_sha
    assert sha == file_sha(ckpt / "trainable.pt")


@pytest.mark.parametrize("rank,alpha", [(4, 4.0), (2, 8.0)])
def test_an_adapter_from_a_different_lora_config_refuses_to_load(monkeypatch, tmp_path, rank, alpha):
    from dream_sleep import load_init_adapter

    ckpt = _write_checkpoint(monkeypatch, tmp_path, _adapter_model(rank=rank, alpha=alpha), rank, alpha)

    with pytest.raises(ValueError, match="rank|alpha"):
        load_init_adapter(_adapter_model(rank=2, alpha=4.0), ckpt, rank=2, alpha=4.0)


def test_a_checkpoint_of_an_unrelated_model_refuses_to_load(monkeypatch, tmp_path):
    from dream_sleep import load_init_adapter

    other = nn.Sequential(nn.Linear(4, 4))
    other[0].weight.requires_grad_(True)
    ckpt = _write_checkpoint(monkeypatch, tmp_path, other, rank=2, alpha=4.0)

    with pytest.raises(RuntimeError):
        load_init_adapter(_adapter_model(), ckpt, rank=2, alpha=4.0)


def test_every_result_record_carries_the_warm_start_hash():
    import io
    import json

    from dream_sleep import make_emit

    buf = io.StringIO()
    emit = make_emit(buf, init_adapter_sha256="deadbeef", init_adapter="warm_start.pt")
    emit({"phase": "sleep"})

    record = json.loads(buf.getvalue())
    assert record["init_adapter_sha256"] == "deadbeef" and record["init_adapter"] == "warm_start.pt"


def test_no_warm_start_is_a_null_hash_on_every_record():
    import io
    import json

    from dream_sleep import make_emit

    buf = io.StringIO()
    make_emit(buf, init_adapter_sha256=None, init_adapter=None)({"phase": "sleep"})

    record = json.loads(buf.getvalue())
    assert record["init_adapter_sha256"] is None


def test_the_warm_start_checkpoint_is_optional_and_absent_by_default():
    from dream_sleep import build_parser

    assert build_parser().parse_args([]).init_adapter is None
    assert build_parser().parse_args(["--init-adapter", "ckpt/step-400"]).init_adapter == "ckpt/step-400"


def test_the_warm_start_loads_before_the_cache_build_and_the_battery():
    """Order of operations, not decoration: the dream cache and the
    self-calibrated battery are both artifacts *of* the warm-started model
    (sec 3.1's battery-recalibration mechanism), so the load has to precede
    every use of the model."""
    import inspect

    import dream_sleep

    source = inspect.getsource(dream_sleep.main)
    load = source.index("load_init_adapter(")
    assert load < source.index("load_or_build_battery(")
    assert load < source.index("build_dream_set if args.dreams else build_cache")
    assert load < source.index("load_dream_cache(")


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
            state = FakeState([torch.zeros(ids.shape[0], 1, 1, dim) for _ in self.layers])
        out = []
        for t in range(ids.shape[1]):
            h = self.embedding(ids[:, t])
            for i in range(len(self.layers)):
                if self.c_capture is not None:
                    self.c_capture.append(h.detach())
                if self.erase_hook is not None:
                    state.ssm_states[i] = self.erase_hook(i, state.ssm_states[i], h)
                state.ssm_states[i] = state.ssm_states[i] + h.view(-1, 1, 1, dim)
            read = torch.einsum("bhpn,bn->bn", state.ssm_states[0], h)
            out.append(self.head(h + read))
        return torch.stack(out, dim=1), state


def _detachable(state: FakeState) -> FakeState:
    return FakeState([s.detach() for s in state.ssm_states])


FakeState.detach = _detachable


def _tiny_setup(rows=1):
    """`rows` is the state's h*p, i.e. the rank available to state_top_dirs.
    The default 1 makes the state rank-1, which is fine for everything that
    does not deflate but degenerate for anything that does -- see
    test_the_deep_fused_spine_carries_gradient_and_the_detached_one_does_not."""
    torch.manual_seed(0)
    model = TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    wake = FakeState([torch.randn(1, rows, 1, 4) for _ in model.layers])
    seed = torch.tensor([[1]])
    return model, opt, wake, seed


def _sentences(token_id: int) -> str:
    """Decoded text where every token ends a sentence, so a fired cue timer
    splices at the next token rather than waiting for a boundary."""
    return f"{token_id}."


def _moved(model, before) -> bool:
    return any(not torch.equal(p, b) for p, b in zip(model.parameters(), before, strict=True))


def test_teacher_dream_caches_a_position_per_token():
    model, _, wake, seed = _tiny_setup()

    dream = teacher_dream(model, wake, seed, n_tokens=8, temperature=1.0,
                          drain=False, decode_token=lambda i: f"<{i}>", needles=["<1>"])

    assert dream.tokens.shape == (1, 8)
    assert dream.logits.shape[0] == 8
    assert all(len(q) == len(model.layers) for q in dream.queries) and len(dream.queries) == 8
    assert len(dream.token_texts) == 8


def test_cue_schedule_forces_every_cue_into_the_dream_in_rotation():
    model, _, wake, seed = _tiny_setup()

    dream = teacher_dream(model, wake, seed, n_tokens=24, temperature=1.0,
                          drain=False, decode_token=_sentences, needles=[],
                          cues=[[3, 4], [5, 6]], cue_every=6)

    emitted = dream.tokens.tolist()[0]
    windows = [emitted[i:i + 2] for i in range(len(emitted) - 1)]
    assert [3, 4] in windows and [5, 6] in windows


def test_cue_schedule_leaves_the_answer_slot_free_to_come_from_the_state():
    """The cue supplies the question; the tokens after it must still be
    sampled, or the dream would be teacher-forced text rather than recall."""
    model, _, wake, seed = _tiny_setup()

    a = teacher_dream(model, wake, seed, 20, 1.0, False, _sentences, [], cues=[[3, 4]], cue_every=5)
    b = teacher_dream(model, wake, seed, 20, 1.0, False, _sentences, [], cues=[[3, 4]], cue_every=5)

    assert a.tokens.tolist() != b.tokens.tolist()


def test_cue_greedy_decodes_the_answer_span_deterministically():
    """The digits after a cue are the recall being rehearsed: sampled at
    temperature they almost never come out right, so they decode greedily
    while the rest of the dream stays sampled."""
    model, _, wake, seed = _tiny_setup()

    dream = teacher_dream(model, wake, seed, n_tokens=18, temperature=1.0,
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
    kwargs = dict(seed_ids=seed, n_tokens=8, temperature=0.0, drain=False,
                  decode_token=str, needles=[])

    torch.manual_seed(0)
    plain = teacher_dream(model, wake, **kwargs)
    torch.manual_seed(0)
    empty = teacher_dream(model, wake, cues=[], cue_every=4, **kwargs)

    assert plain.tokens.tolist() == empty.tokens.tolist()


def _forced(model, token_id: int):
    """Bias the head so greedy decoding always emits `token_id`."""
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[token_id] = 10.0
    return model


def test_teacher_dream_ends_at_the_conversation_end_token():
    model, _, wake, seed = _tiny_setup()
    _forced(model, 5)

    dream = teacher_dream(model, wake, seed, n_tokens=16, temperature=0.0, drain=False,
                          decode_token=str, needles=[], stop_id=5)

    assert dream.stop_reason == "eoc"
    assert dream.tokens.shape[1] < 16
    assert dream.tokens.shape[1] == dream.logits.shape[0] == len(dream.token_texts)


def test_teacher_dream_runs_to_the_token_budget_when_the_stop_token_never_fires():
    model, _, wake, seed = _tiny_setup()
    _forced(model, 4)

    dream = teacher_dream(model, wake, seed, n_tokens=8, temperature=0.0, drain=False,
                          decode_token=str, needles=[], stop_id=5)

    assert dream.stop_reason == "max-tokens"
    assert dream.tokens.shape == (1, 8)


def test_teacher_dream_stops_on_the_turn_backstop_when_the_stop_token_never_fires():
    """Unbanned eos is ordinary dream-internal turn structure, so a dream that
    never emits <|eoc|> is bounded by turns as well as by tokens."""
    model, _, wake, seed = _tiny_setup()
    _forced(model, 4)

    dream = teacher_dream(model, wake, seed, n_tokens=64, temperature=0.0, drain=False,
                          decode_token=str, needles=[], stop_id=5, turn_id=4, max_turns=3)

    assert dream.stop_reason == "turn-backstop"
    assert dream.tokens.shape == (1, 3)


def test_teacher_dream_drains_the_state_it_generates_from():
    model, _, wake, seed = _tiny_setup()
    kwargs = dict(seed_ids=seed, n_tokens=6, temperature=0.0, decode_token=str, needles=[])

    intact = teacher_dream(model, wake, drain=False, **kwargs)
    drained = teacher_dream(model, wake, drain=True, **kwargs)

    assert not torch.allclose(intact.final_state.ssm_states[0], drained.final_state.ssm_states[0])


def test_distill_replay_and_sft_take_their_optimizer_steps():
    model, opt, wake, seed = _tiny_setup()
    dream = teacher_dream(model, wake, seed, 6, 0.0, False, str, [])
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

    state, dream = distill_live(model, opt, wake, seed, n_tokens=5, temperature=1.0,
                                kl_temp=1.0, accum=1, decode_token=str, needles=[],
                                on_step=lambda s, l: None)

    assert _moved(model, before)
    assert dream.tokens.shape == (1, 5) and state is dream.final_state
    torch.testing.assert_close(wake.ssm_states[0], wake_before)  # the arm works on a copy


def _fixed_dream(model, wake, seed, n_tokens=4):
    return teacher_dream(model, wake, seed, n_tokens, 0.0, False, str, [])


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
        "b4-raw": "intact",
        "b4-deflated": "intact",
        "b4-qcm": "intact",
        # the sigma arm erases the raw subspace partially, so it too hands the
        # next wake an intact-shaped state
        "b4-sigma": "intact",
        "replay": "none",
        "ce-on-dream": "none",
        "drain": "drained",
        "counterfactual": "intact",
        "counterfactual-commit": "committed",
        "drain-live": "drained",
        "sft-ref": "none",
        "no-sleep": "intact",
        "b2-fused-detached": "intact",
        "b2-fused-deep": "intact",
        "b3-fused": "intact",
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
                fresh_state_replay=False, deep=False, ce_on_dream=False, n_facts=2, filler_tokens=4,
                erase_op="deflated", spine_block=3, cf_batch=3)
    return SimpleNamespace(**{**args, **overrides})


def _built_cache(tmp_path, args):
    from dream_sleep import build_cache

    model, opt, wake, seed = _tiny_setup()
    encode, decode = _fake_io()
    facts = [Fact("osprey", "bird", "1 2 3 4 5"), Fact("heron", "bird", "5 9 7 9 7")]
    transcript = torch.tensor([[1, 2, 3, 4, 5, 6]])
    cache = build_cache(model, args, tmp_path / "dream_cache_s1234.pt", transcript, facts, 3,
                        encode, decode, _FakeTokenizer(), USER, ASST, None)
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


def _warm_started_cache(tmp_path, args, adapter_sha):
    """A cache built by a model whose head carries a non-zero adapter, so
    "generated at the base" and "generated warm-started" are distinguishable in
    the dream tokens themselves."""
    from dream_sleep import build_cache

    model, _, _, _ = _tiny_setup()
    model.head = LoRALinear(model.head, rank=2, alpha=4.0, dropout=0.0)
    with torch.no_grad():
        # Asymmetric: a uniform shift moves every logit equally and no argmax.
        model.head.lora_B.copy_(-torch.arange(model.head.lora_B.numel(),
                                              dtype=model.head.lora_B.dtype).view_as(model.head.lora_B))
    scales: list[float] = []
    original = model.forward

    def spy(ids, state=None):
        if model.c_capture is not None:  # a generation step, not the wake pass
            scales.append(model.head.scale)
        return original(ids, state=state)

    model.forward = spy
    cache = build_cache(model, args, tmp_path / f"dream_{adapter_sha or 'base'}.pt",
                        torch.tensor([[1, 2, 3, 4, 5, 6]]),
                        [Fact("osprey", "bird", "1 2 3 4 5")], 3,
                        *_fake_io(), _FakeTokenizer(), USER, ASST, None, adapter_sha=adapter_sha)
    return cache, scales


def test_a_warm_started_cache_is_generated_with_the_loaded_adapter_active(tmp_path):
    """Sec 4(1): fresh dream caches are generated *from the warm-started
    weights*. Zeroing the adapter for generation would make --init-adapter a
    no-op on the one artifact every arm then distils."""
    args = _sleep_args()

    base, base_scales = _warm_started_cache(tmp_path, args, None)
    warm, warm_scales = _warm_started_cache(tmp_path, args, "deadbeef")

    assert base_scales and all(s == 0.0 for s in base_scales)
    assert warm_scales and all(s == 2.0 for s in warm_scales)  # alpha/rank, adapters live
    assert warm.dream_ids != base.dream_ids


def test_the_cache_records_which_weights_generated_it(tmp_path):
    from dream_sleep import load_dream_cache

    args = _sleep_args()
    base, _ = _warm_started_cache(tmp_path, args, None)
    warm, _ = _warm_started_cache(tmp_path, args, "deadbeef")

    assert base.generator == "base" and warm.generator == "deadbeef"
    assert load_dream_cache(tmp_path / "dream_deadbeef.pt").generator == "deadbeef"
    assert "deadbeef" in (tmp_path / "dream_deadbeef.txt").read_text()


@pytest.mark.parametrize("arm", ["replay", "ce-on-dream", "drain", "counterfactual", "sft-ref"])
def test_run_sleep_trains_every_arm_from_the_cached_dream(tmp_path, arm):
    from dream_sleep import run_sleep

    args = _sleep_args(ce_on_dream=(arm == "ce-on-dream"))
    model, opt, wake, cache, facts, transcript, encode, decode = _built_cache(tmp_path, args)
    before = [p.detach().clone() for p in model.parameters()]
    records: list[dict] = []

    carried = run_sleep(arm, model, opt, args, 1, wake, transcript, [(1, f) for f in facts], 3, cache,
                        encode, decode, _FakeTokenizer(), USER, ASST, records.append, None,
                        lambda step: records.append({"phase": "periodic", "step": step}))

    assert _moved(model, before)
    sleep = next(r for r in records if r["phase"] == "sleep")
    assert sleep["token_gradients"] > 0
    assert (carried is None) == (ARM_CARRY[arm] == "none")
    if arm != "sft-ref":
        dream = next(r for r in records if r["phase"] == "dream")
        assert "bound_cov" in dream and "misbound" in dream


def test_run_sleep_cuts_along_the_operator_the_cell_was_launched_with(tmp_path):
    from dream_sleep import run_sleep

    trained = {}
    for op in ("raw", "deflated"):
        args = _sleep_args(erase_op=op)
        model, opt, wake, cache, facts, transcript, encode, decode = _built_cache(tmp_path, args)
        run_sleep("counterfactual", model, opt, args, 1, wake, transcript, [(1, f) for f in facts], 3, cache,
                  encode, decode, _FakeTokenizer(), USER, ASST, lambda r: None, lambda step: None, None)
        trained[op] = [p.detach().clone() for p in model.parameters()]

    assert any(not torch.equal(a, b) for a, b in zip(trained["raw"], trained["deflated"], strict=True))


def test_run_sleep_streams_the_probe_battery_on_the_registered_cadence(tmp_path):
    from dream_sleep import run_sleep

    args = _sleep_args(distill_steps=6, probe_every=2)
    model, opt, wake, cache, facts, transcript, encode, decode = _built_cache(tmp_path, args)
    probes: list[int] = []

    run_sleep("counterfactual", model, opt, args, 1, wake, transcript, [(1, f) for f in facts], 3, cache,
              encode, decode, _FakeTokenizer(), USER, ASST, lambda r: None, probes.append, None)

    assert probes == [2, 4]  # not at the last step -- the end-of-sleep battery covers that


def test_a_later_wave_generates_its_own_dream_from_the_carried_state(tmp_path):
    """A wave-2 sleep that reuses the cached wave-1 dream never rehearses the
    new facts. The dream a later wave distils comes from that wave's own
    carried state, so its tokens differ from the cached one's."""
    from dream_sleep import generate_wave_dream

    args = _sleep_args()
    model, _, _, cache, _, _, encode, decode = _built_cache(tmp_path, args)
    carried = FakeState([torch.randn(1, 1, 1, 4) for _ in model.layers])
    facts = [Fact("marimba", "instrument", "5 4 6 6 9")]
    states: list[object] = []
    original = model.forward

    def spy(ids, state=None):
        # The fake backbone advances the state in place, so snapshot it.
        states.append(None if state is None else [s.clone() for s in state.ssm_states])
        return original(ids, state=state)

    model.forward = spy
    dream = generate_wave_dream(model, args, carried, facts, encode, decode, _FakeTokenizer(),
                                USER, ASST, teacher="base", stop_id=None)

    assert dream.tokens.shape == (1, args.dream_tokens)
    assert len(dream.queries) == args.dream_tokens
    # Generated from this wave's carried state, not the cached wave-1 one.
    assert torch.equal(states[0][0], carried.ssm_states[0])
    assert not torch.equal(states[0][0], cache.wake_state.ssm_states[0])


def test_the_base_teacher_generates_with_the_adapters_bypassed(tmp_path):
    """`--wave-teacher base` keeps the wave-1 contract (frozen base, adapters
    off); `current` distils the student the run has already trained."""
    from dream_sleep import generate_wave_dream

    args = _sleep_args()
    model, _, _, _, _, _, encode, decode = _built_cache(tmp_path, args)
    model.lora = LoRALinear(model.head, rank=2, alpha=6.0, dropout=0.0)
    carried = FakeState([torch.randn(1, 1, 1, 4) for _ in model.layers])
    seen: list[float] = []
    original = model.forward

    def spy(ids, state=None):
        seen.append(model.lora.scale)
        return original(ids, state=state)

    model.forward = spy
    generate_wave_dream(model, args, carried, [Fact("oboe", "instrument", "3 7 8 5 0")],
                        encode, decode, _FakeTokenizer(), USER, ASST, teacher="base", stop_id=None)
    assert seen and all(s == 0.0 for s in seen)
    assert model.lora.scale == 3.0  # restored

    seen.clear()
    generate_wave_dream(model, args, carried, [Fact("oboe", "instrument", "3 7 8 5 0")],
                        encode, decode, _FakeTokenizer(), USER, ASST, teacher="current", stop_id=None)
    assert seen and all(s == 3.0 for s in seen)  # alpha/rank


def test_multi_wave_requires_an_explicit_wave_teacher():
    """Who teaches wave 2 -- the frozen base or the already-trained student --
    is a protocol choice the harness must not make silently."""
    from dream_sleep import validate_wave_args

    with pytest.raises(SystemExit):
        validate_wave_args(_sleep_args(waves=2, wave_teacher=None))
    validate_wave_args(_sleep_args(waves=2, wave_teacher="base"))
    validate_wave_args(_sleep_args(waves=1, wave_teacher=None))


# --- the fused B arms (sec 3.5) ------------------------------------------


def _serial_spine(model, tokens, wake):
    """Ground truth for spine_states: the state carried into each token by a
    plain one-token-at-a-time run."""
    from dream_sleep import copy_state

    state = copy_state(wake)
    per_token = []
    with torch.no_grad():
        for t in range(tokens.shape[1]):
            per_token.append([s.clone() for s in state.ssm_states])
            _, state = model(tokens[:, t : t + 1], state=state)
    return per_token


@pytest.mark.parametrize("block", [1, 2, 3, 4, 8])
def test_spine_states_match_a_plain_per_token_run(block):
    """The block/batch decomposition is an optimization, not a different
    trajectory: token t's carried state must be what a serial run would hold."""
    from dream_sleep import spine_states

    model, _, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed, n_tokens=6)
    expected = _serial_spine(model, dream.tokens, wake)

    with torch.no_grad():
        spine = spine_states(model, dream.tokens, wake, block)

    assert all(t.shape[0] == 6 for t in spine["ssm_states"])
    for t, per_layer in enumerate(expected):
        for i, want in enumerate(per_layer):
            torch.testing.assert_close(spine["ssm_states"][i][t : t + 1], want)


def _fused_run(arm, steps=1, rows=1, **kwargs):
    from dream_sleep import distill_fused

    model, opt, wake, seed = _tiny_setup(rows)
    dream = _fixed_dream(model, wake, seed, n_tokens=6)
    losses: list[float] = []
    records: list[dict] = []
    tokens = distill_fused(model, opt, dream, wake, steps=steps, kl_temp=1.0,
                           on_step=lambda s, l: losses.append(l), erase_op=kwargs.pop("erase_op", "deflated"),
                           deep=(arm == "b2-fused-deep"), block=kwargs.pop("block", 2),
                           cf_batch=kwargs.pop("cf_batch", 4),
                           frozen_spine=None, check=records.append, **kwargs)
    return model, tokens, losses, records


def test_b3_fused_equals_b2_fused_detached_at_pass_one():
    """Sec 3.5: at pass 1 the student's weights are the generator snapshot, so
    B3's cached spine and B2-detached's freshly recomputed one coincide."""
    from dream_sleep import distill_fused, spine_states

    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed, n_tokens=6)
    with torch.no_grad():
        generator_spine = spine_states(model, dream.tokens, wake, 2)

    b3: list[float] = []
    records: list[dict] = []
    distill_fused(model, opt, dream, wake, steps=1, kl_temp=1.0, on_step=lambda s, l: b3.append(l),
                  block=2, cf_batch=4, frozen_spine=generator_spine, check=records.append)

    _, _, b2, _ = _fused_run("b2-fused-detached")
    assert b3 == pytest.approx(b2)
    assert records and records[0]["equivalent"] is True
    assert records[0]["b3_loss"] == pytest.approx(records[0]["b2_loss"])


def test_b3_fused_leaves_the_generator_spine_usable_after_the_equivalence_check():
    """The check builds a second full spine, so the generator's is parked on the
    host while it runs -- every later pass still distils on it, unchanged."""
    from dream_sleep import distill_fused, spine_states

    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed, n_tokens=6)
    with torch.no_grad():
        frozen = spine_states(model, dream.tokens, wake, 2)
    before = {a: [t.clone() for t in layers] for a, layers in frozen.items()}

    records: list[dict] = []
    distill_fused(model, opt, dream, wake, steps=2, kl_temp=1.0, on_step=lambda s, l: None,
                  block=2, cf_batch=4, frozen_spine=frozen, check=records.append)

    assert records and records[0]["equivalent"] is True
    for attr, layers in before.items():
        assert len(frozen[attr]) == len(layers)
        for got, want in zip(frozen[attr], layers, strict=True):
            assert got.device == want.device
            torch.testing.assert_close(got, want)


def test_b3_fused_reports_a_failed_equivalence_when_the_spine_is_not_the_generator_s():
    """The check has to be able to fail, or it is decoration."""
    from dream_sleep import distill_fused, spine_states

    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed, n_tokens=6)
    with torch.no_grad():
        wrong = spine_states(model, dream.tokens, wake, 2)
        wrong["ssm_states"][0] = wrong["ssm_states"][0] + 1.0

    records: list[dict] = []
    distill_fused(model, opt, dream, wake, steps=1, kl_temp=1.0, on_step=lambda s, l: None,
                  block=2, cf_batch=4, frozen_spine=wrong, check=records.append)

    assert records and records[0]["equivalent"] is False


@pytest.mark.parametrize("arm", ["b2-fused-detached", "b2-fused-deep"])
def test_the_fused_arms_run_end_to_end_with_finite_losses(arm):
    """Not compared to per-token B2 for equality: different step currency
    (sec 3.5). Both must simply train and stay finite."""
    import math

    model, tokens, losses, _ = _fused_run(arm, steps=3)

    assert tokens == 18 and len(losses) == 3
    assert all(math.isfinite(x) for x in losses)


def test_the_fused_and_per_token_b2_arms_both_train_on_the_same_dream():
    import math

    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed, n_tokens=6)
    before = [p.detach().clone() for p in model.parameters()]
    per_token, _ = distill_counterfactual(model, opt, dream, wake, steps=6, kl_temp=1.0, accum=1,
                                          in_place=False, deep=False, on_step=lambda s, l: None)
    assert per_token == 6 and _moved(model, before)

    fused_model, fused_tokens, losses, _ = _fused_run("b2-fused-detached")
    assert fused_tokens == 6 and math.isfinite(losses[0])


@pytest.mark.parametrize("op", ["raw", "deflated"])
def test_the_deep_fused_spine_carries_gradient_and_the_detached_one_does_not(op, monkeypatch):
    import dream_sleep

    seen: list[bool] = []
    original = dream_sleep.spine_states

    def spy(*a, **k):
        spine = original(*a, **k)
        seen.append(any(t.requires_grad for t in spine["ssm_states"]))
        return spine

    monkeypatch.setattr(dream_sleep, "spine_states", spy)
    # h*p = 4, not the default rank-1 state: with rank 1 the state's only top
    # singular direction IS its own row, so deflating the query against it
    # leaves a direction exactly orthogonal to the state, the erase is the
    # identity to fp precision, the counterfactual reproduces the teacher, and
    # the loss is 0 -- both arms then take a null step and coincide for want of
    # gradient signal rather than for want of a gradient path. (h*p = 16 is
    # degenerate the other way: the deflated direction falls under the skip-cone
    # threshold half the time.) The real 780m state is 3072x128.
    rows = 4
    detached, _, _, _ = _fused_run("b2-fused-detached", erase_op=op, rows=rows)
    assert seen == [False]

    seen.clear()
    deep, _, _, _ = _fused_run("b2-fused-deep", erase_op=op, rows=rows)
    assert seen == [True]

    assert any(not torch.equal(a, b) for a, b in
               zip(detached.parameters(), deep.parameters(), strict=True))


@pytest.mark.parametrize("op", ["raw", "deflated"])
@pytest.mark.parametrize("arm", ["b2-fused-detached", "b2-fused-deep"])
def test_the_fused_arms_never_differentiate_the_protected_subspace(op, arm, monkeypatch):
    """v is computed under no-grad at its source; the deep arm's live spine
    must not open a path to it (sec 3.4)."""
    import dream_sleep

    grads: list[bool] = []
    original = dream_sleep.state_top_dirs
    monkeypatch.setattr(dream_sleep, "state_top_dirs",
                        lambda s, k: (lambda v: (grads.append(v.requires_grad), v)[1])(original(s, k)))

    _fused_run(arm, erase_op=op)

    assert (op == "raw") or (grads and not any(grads))


def test_micro_batched_counterfactuals_match_the_unbatched_pass():
    from dream_sleep import fused_pass, spine_states

    model, _, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed, n_tokens=6)
    scored = list(range(6))
    with torch.no_grad():
        spine = spine_states(model, dream.tokens, wake, 2)
        whole, skipped_whole = fused_pass(model, dream, wake, spine, scored, 1.0, "deflated", 6, backward=False)
        micro, skipped_micro = fused_pass(model, dream, wake, spine, scored, 1.0, "deflated", 2, backward=False)

    assert micro == pytest.approx(whole, rel=1e-5)
    assert skipped_micro == skipped_whole


def test_the_raw_erase_zeroes_the_ablated_readout_through_the_fused_path(monkeypatch):
    """Sec 3.4's identity, asserted where the arm actually applies it: after a
    raw own-query cut the state's read along that query is identically zero."""
    import dream_sleep

    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = dream_sleep.erase_ssm

    def spy(ssm_state, c, *a, **k):
        erased, skipped = original(ssm_state, c, *a, **k)
        seen.append((erased, c))
        return erased, skipped

    monkeypatch.setattr(dream_sleep, "erase_ssm", spy)
    _fused_run("b2-fused-detached", erase_op="raw")

    assert seen
    for erased, c in seen:
        assert read(erased, c).abs().max().item() < 1e-4


def test_an_unknown_erase_op_is_refused():
    """argparse guards the CLI; the arms call erase_ssm programmatically."""
    with pytest.raises(ValueError, match="erase op"):
        erase_ssm(torch.randn(1, 2, 4, 8), torch.randn(1, 8), op="defalted")


def test_the_erase_operates_per_batch_element():
    """The fused arms ablate a whole batch of token-positions at once: every
    element gets its own deflation basis and its own skip decision."""
    torch.manual_seed(4)
    cone = torch.nn.functional.normalize(torch.randn(8), dim=0)
    pure = (torch.randn(2, 4, 1) * cone).unsqueeze(0)
    mixed = torch.randn(1, 2, 4, 8)
    state = torch.cat([pure, mixed], dim=0)
    queries = torch.cat([cone.unsqueeze(0), torch.randn(1, 8)], dim=0)

    erased, skipped = erase_ssm(state, queries)

    assert skipped == 1
    torch.testing.assert_close(erased[0], state[0])  # pure cone: skipped
    assert not torch.equal(erased[1], state[1])


@pytest.mark.parametrize("arm", ["b2-fused-detached", "b2-fused-deep", "b3-fused"])
def test_run_sleep_stamps_the_fused_arm_and_the_warm_start_on_every_record(tmp_path, arm):
    import io
    import json

    from dream_sleep import ARM_CARRY, make_emit, run_sleep

    args = _sleep_args(distill_steps=2)
    model, opt, wake, cache, facts, transcript, encode, decode = _built_cache(tmp_path, args)
    before = [p.detach().clone() for p in model.parameters()]
    buf = io.StringIO()
    emit = make_emit(buf, erase_op=args.erase_op, init_adapter_sha256="deadbeef", init_adapter="warm_start.pt")

    carried = run_sleep(arm, model, opt, args, 1, wake, transcript, [(1, f) for f in facts], 3, cache,
                        encode, decode, _FakeTokenizer(), USER, ASST, emit, lambda step: None, None)

    records = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert _moved(model, before)
    assert carried is wake and ARM_CARRY[arm] == "intact"
    assert records and all(r["init_adapter_sha256"] == "deadbeef" for r in records)
    assert all(r["arm"] == arm for r in records if "arm" in r)
    sleep = next(r for r in records if r["phase"] == "sleep")
    assert sleep["token_gradients"] > 0
    checks = [r for r in records if r["phase"] == "equivalence"]
    assert bool(checks) == (arm == "b3-fused")
    assert all(r["equivalent"] for r in checks)


# --- multi-sleep (sec 3.7) -----------------------------------------------


def _wave_sleep(tmp_path, arm, wave, teacher="current", wake=None, model=None, records=None,
                seen=None, transcript=None, verify=None, committed=None, **overrides):
    """One sleep of a multi-sleep run, driven at the given wave."""
    from dream_sleep import run_sleep

    args = _sleep_args(waves=4, wave_teacher=teacher, **overrides)
    built = _built_cache(tmp_path, args)
    model = model or built[0]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cache, facts, encode, decode = built[3], built[4], built[6], built[7]
    seen = seen if seen is not None else [(1, f) for f in facts] + [
        (w, Fact(f"fact{w}", "thing", f"{w} {w} {w} {w} {w}")) for w in range(2, wave + 1)]
    carried = run_sleep(arm, model, opt, args, wave, wake or built[2],
                        transcript if transcript is not None else built[5], seen, 3, cache,
                        encode, decode, _FakeTokenizer(), USER, ASST,
                        (records if records is not None else []).append, lambda step: None, None,
                        verify=verify, committed=committed)
    return carried, cache


def test_a_later_sleep_distils_a_dream_it_generated_itself(tmp_path):
    """Sec 3.7: per-arm per-wave dreams, generated from the state that sleep
    carried in, cued on that wave's facts only. Reusing the cached wave-1
    dream would rehearse none of the new material."""
    records: list[dict] = []
    model, _, _, _ = _tiny_setup()
    wake = FakeState([torch.randn(1, 1, 1, 4) for _ in range(2)])
    states: list[object] = []
    original = model.forward

    def spy(ids, state=None):
        if model.c_capture is not None and not states:
            states.append([s.clone() for s in state.ssm_states])
        return original(ids, state=state)

    model.forward = spy
    _, cache = _wave_sleep(tmp_path, "counterfactual", wave=2, wake=wake, model=model, records=records)

    built = next(r for r in records if r["phase"] == "cache" and r["wave"] == 2)
    assert built["cues_cover"] == ["fact2"]  # this wave's facts only
    assert built["dream_generator"] == "student"
    assert built["dream_tokens"] == 8
    # Generated from this sleep's carried state, not the cached wave-1 one.
    assert torch.equal(states[0][0], wake.ssm_states[0])
    assert not torch.equal(states[0][0], cache.wake_state.ssm_states[0])


def test_each_sleep_s_dream_comes_from_its_own_carried_state(tmp_path):
    """The generator is the student as of that sleep's start, generating from
    the state it carried in -- a different carry teaches a different dream."""
    from dream_sleep import generate_wave_dream

    args = _sleep_args(waves=4, wave_teacher="current")
    model, _, _, _, _, _, encode, decode = _built_cache(tmp_path, args)
    facts = [Fact("marimba", "instrument", "5 4 6 6 9")]
    dreams = [generate_wave_dream(model, args, FakeState([torch.randn(1, 1, 1, 4) * scale for _ in range(2)]),
                                  facts, encode, decode, _FakeTokenizer(), USER, ASST, teacher="current",
                                  stop_id=None)
              for scale in (1.0, 7.0)]

    assert not torch.equal(dreams[0].logits, dreams[1].logits)


def _generation_scales(tmp_path, teacher, wave):
    model, _, _, _ = _tiny_setup()
    model.head = LoRALinear(model.head, rank=2, alpha=4.0, dropout=0.0)
    with torch.no_grad():
        model.head.lora_B += 0.5
    scales: list[float] = []
    original = model.forward

    def spy(ids, state=None):
        if model.c_capture is not None:  # a generation step, not training
            scales.append(model.head.scale)
        return original(ids, state=state)

    model.forward = spy
    _wave_sleep(tmp_path, "counterfactual", wave=wave, teacher=teacher, model=model)
    return scales


def test_the_frozen_base_control_never_generates_from_the_trained_student(tmp_path):
    """Sec 3.2's drift-contribution control: `--wave-teacher base` pins the
    generator at the base at every sleep, while the registered protocol
    (`current`) lets it drift with the student."""
    for wave in (2, 3, 4):
        base, current = _generation_scales(tmp_path, "base", wave), _generation_scales(tmp_path, "current", wave)
        assert base and all(s == 0.0 for s in base)
        assert current and all(s == 2.0 for s in current)


def test_sequential_sft_ref_trains_only_on_the_current_wave_s_facts(tmp_path, monkeypatch):
    """The standard CL baseline: wave by wave, each sleep seeing only its own
    wake transcript -- which is where backward interference should appear."""
    import dream_sleep

    trained: list[list[int]] = []
    monkeypatch.setattr(dream_sleep, "distill_sft",
                        lambda model, opt, ids, *a, **k: trained.append(ids[0].tolist()) or 0)

    _wave_sleep(tmp_path, "sft-ref", wave=3, transcript=torch.tensor([[7, 8, 9]]))

    assert trained == [[7, 8, 9]]


def test_distractors_never_reuse_a_code_another_wave_already_holds(tmp_path):
    """Multi-sleep probes every wave's facts, so every wave needs foils -- and a
    foil that is some other wave's real code would score that fact as forgotten."""
    from dream_sleep import build_distractors

    wave1 = [Fact("osprey", "bird", "1 2 3 4 5"), Fact("heron", "bird", "5 9 7 9 7")]
    first = build_distractors(wave1, 1234)
    wave2 = [Fact("marimba", "instrument", first["osprey"])]  # collides on purpose

    second = build_distractors(wave2, 1234, taken=set(first.values()) | {f.code for f in wave1})

    assert build_distractors(wave1, 1234) == first  # wave 1 unchanged by the new argument
    assert set(second.values()).isdisjoint(set(first.values()) | {f.code for f in wave1 + wave2})


def _commit_setup():
    model, _, _, _ = _tiny_setup()
    carried = FakeState([torch.randn(1, 1, 1, 4) for _ in model.layers])
    facts = [Fact("osprey", "bird", "1 2 3 4 5"), Fact("heron", "bird", "5 9 7 9 7")]
    return model, carried, [(1, f) for f in facts]


def test_the_commit_step_erases_only_the_facts_that_verify():
    """B2' (08-06 sec 3): install by counterfactual training, then per fact a
    fresh-state margin check, and only where it passes does one real erase land
    on the carried state. The erase is a verified memory-policy step, never a
    training signal."""
    from dream_sleep import commit_erase

    model, carried, seen = _commit_setup()
    before = [s.clone() for s in carried.ssm_states]
    records: list[dict] = []

    fired = commit_erase(model, carried, seen, set(), _fake_io()[0], USER, ASST,
                         lambda f: (2.0, f.entity == "osprey", 0.0, 0.0), "raw", records.append, 1)

    assert fired == ["osprey"]
    assert not torch.equal(carried.ssm_states[0], before[0])
    assert [(r["fact"], r["committed"]) for r in records] == [("osprey", True), ("heron", False)]


def test_a_fact_is_committed_once_and_never_re_erased():
    """The commit is a policy step per fact, not a per-sleep tax on every fact
    that ever installed -- re-cutting along the same query every sleep would
    charge the bystanders again for nothing."""
    from dream_sleep import commit_erase

    model, carried, seen = _commit_setup()
    committed: set[str] = set()
    verify = lambda f: (2.0, True, 0.0, 0.0)  # noqa: E731

    first = commit_erase(model, carried, seen, committed, _fake_io()[0], USER, ASST,
                         verify, "raw", lambda r: None, 1)
    settled = [s.clone() for s in carried.ssm_states]
    second = commit_erase(model, carried, seen, committed, _fake_io()[0], USER, ASST,
                          verify, "raw", lambda r: None, 2)

    assert first == ["osprey", "heron"] and second == []
    assert torch.equal(carried.ssm_states[0], settled[0])


def test_the_commit_arm_moves_the_carried_state_and_plain_b2_does_not(tmp_path):
    """The arm-level contrast B2' exists to measure: same training, different
    state handed to the next wake."""
    for arm, moves in (("counterfactual", False), ("counterfactual-commit", True)):
        wake = FakeState([torch.randn(1, 1, 1, 4) for _ in range(2)])
        before = wake.ssm_states[0].clone()
        carried, _ = _wave_sleep(tmp_path, arm, wave=1, wake=wake,
                                 verify=lambda f: (2.0, True, 0.0, 0.0), committed=set())
        assert (not torch.equal(carried.ssm_states[0], before)) == moves


def test_the_commit_arm_is_registered_as_carrying_a_committed_state():
    assert ARM_CARRY["counterfactual-commit"] == "committed"


def test_the_r_matrix_row_averages_each_wave_s_facts_separately():
    """R[i][j] is wave i's facts after sleep j (sec 3.7), so one probe sweep
    over all facts so far is one column, split by the wave that taught them."""
    from dream_sleep import r_matrix_row

    row = r_matrix_row({
        "osprey": {"fact_wave": 1, "margin": 2.0, "install": True},
        "heron": {"fact_wave": 1, "margin": 0.0, "install": False},
        "marimba": {"fact_wave": 2, "margin": 3.0, "install": True},
    })

    assert row == {1: {"mean_margin": 1.0, "installs": 1, "facts": 2},
                   2: {"mean_margin": 3.0, "installs": 1, "facts": 1}}


def test_a_fact_with_no_foil_is_not_averaged_into_its_wave():
    from dream_sleep import r_matrix_row

    assert r_matrix_row({"osprey": {"fact_wave": 1, "margin": None, "install": False}}) == {}


def _triangle():
    return {(1, 1): {"mean_margin": 4.0, "installs": 2, "facts": 2},
            (1, 2): {"mean_margin": 1.0, "installs": 1, "facts": 2},
            (2, 2): {"mean_margin": 3.0, "installs": 2, "facts": 2}}


def test_backward_transfer_is_the_mean_move_from_each_wave_s_own_sleep_to_the_last():
    from dream_sleep import cl_summary

    summary = cl_summary(_triangle(), waves=2)

    assert summary["bwt"] == pytest.approx(-3.0)  # wave 1: 1.0 - 4.0
    assert summary["installs_final"] == 3  # 1 of wave 1 plus 2 of wave 2
    assert summary["installs_peak"] == 4  # wave 1 peaked at 2 before losing one


def test_backward_transfer_is_undefined_for_a_single_sleep():
    from dream_sleep import cl_summary

    assert cl_summary({(1, 1): {"mean_margin": 4.0, "installs": 2, "facts": 2}}, waves=1)["bwt"] is None


def test_the_r_matrix_is_lower_triangular_over_the_run_s_sleeps():
    """A wave's facts do not exist before their own wake, so the upper triangle
    is empty by construction and must not be counted as a loss."""
    from dream_sleep import cl_summary, r_matrix_rows

    r = {(i, j): {"mean_margin": float(j - i), "installs": 1, "facts": 1}
         for j in range(1, 5) for i in range(1, j + 1)}

    assert [len(row) for row in r_matrix_rows(r, waves=4)] == [4, 4, 4, 4]
    assert r_matrix_rows(r, waves=4)[0][0] == pytest.approx(0.0)
    assert r_matrix_rows(r, waves=4)[3][0] is None  # wave 4 did not exist at sleep 1
    assert cl_summary(r, waves=4)["bwt"] == pytest.approx((3 + 2 + 1) / 3)


@pytest.mark.parametrize("wave", [1, 2, 3])
def test_b3_fused_checks_its_pass_one_equivalence_at_every_sleep(tmp_path, wave):
    """Sec 3.5: B3's spine is refreshed each sleep from that sleep's generator
    snapshot, so the B3 == B2-fused-detached identity is machine-checked per
    sleep, not once per run."""
    records: list[dict] = []
    _wave_sleep(tmp_path, "b3-fused", wave=wave, records=records, distill_steps=2)

    checks = [r for r in records if r["phase"] == "equivalence"]
    assert len(checks) == 1 and checks[0]["equivalent"]


# --- context-leakage probe (DISCUSSION-20260808 sec 2.10.11) -----------------


def _leak_items():
    from erase_probe import Bystander

    return [Bystander("clara", "Where does Clara live?", "Clara lives in Lisbon.",
                      f"{USER} Where does Clara live?{ASST} Clara lives in", " Lisbon")]


def test_the_leakage_probe_scores_distractor_content_against_its_own_floor():
    from dream_sleep import probe_leakage

    items = _leak_items()
    records: list[dict] = []
    floor = probe_leakage(items, lambda p, a: ("nowhere.", -6.0), records.append,
                          "replay", 1, "leak_floor", (".", "\n"))
    assert floor == {"clara": -6.0}

    records.clear()
    probe_leakage(items, lambda p, a: (" Lisbon.", -0.5), records.append,
                  "replay", 1, "leakage", (".", "\n"), baseline=floor, step=800)

    assert len(records) == 1
    record = records[0]
    assert record["phase"] == "leakage" and record["item"] == "clara" and record["kind"] == "offformat"
    assert record["match"] is True
    assert record["logprob_delta"] == pytest.approx(5.5)


def test_the_leakage_probe_reports_a_miss_as_a_miss():
    from dream_sleep import probe_leakage

    records: list[dict] = []
    probe_leakage(_leak_items(), lambda p, a: ("Oslo.", -4.0), records.append,
                  "replay", 1, "leakage", (".", "\n"))

    assert records[0]["match"] is False and records[0]["logprob_delta"] is None


# ---- B4's dream-set training loop (DISCUSSION-20260808 sec 2.10.3-2.10.5) --


def _cached(model, wake, seed, n_tokens=4):
    """A CachedDream carrying a per-layer rank-1 eraser, as the cache builder
    writes it."""
    dream = _fixed_dream(model, wake, seed, n_tokens)
    bases = [torch.nn.functional.normalize(torch.randn(1, 4), dim=-1) for _ in model.layers]
    return CachedDream(
        dream_ids=[int(i) for i in dream.tokens[0]],
        token_texts=dream.token_texts,
        teacher_logits=dream.logits,
        cue_flags=[False] * n_tokens,
        prefix_len=1,
        stop_reason="eoc",
        divergence=[0.0] * n_tokens,
        gate_positions=[1],
        queries=[[q for q in dream.queries[1]]],
        spectra=[[1.0] for _ in model.layers],
        ranks={"ratio-gap": [1] * len(model.layers), "median": [1] * len(model.layers)},
        bases={v: bases for v in ("raw", "deflated", "qcm")},
    )


def test_arm_a_still_trains_from_a_blank_state():
    """Sec 2.9.6/2.10.12: A is total denial and `init_state=None` is what makes
    it so -- adding B4's start state must not have touched it."""
    model, opt, wake, seed = _tiny_setup()
    dream = _fixed_dream(model, wake, seed)
    seen = []
    model.forward = _recording(model, seen)

    distill_replay(model, opt, dream, steps=1, chunk_len=4, kl_temp=1.0, on_step=lambda s, l: None)

    assert seen[0] is None


def _recording(model, seen):
    original = model.__class__.forward

    def forward(ids, state=None):
        seen.append(None if state is None else state.ssm_states[0].clone())
        return original(model, ids, state)

    return forward


def test_b4_starts_every_dream_from_a_fresh_erased_copy_of_the_wake_state():
    """Sec 2.10.4's carry matrix: never erase-once-then-carry. Dream k+1's
    start state must be the wake state's own erasure, not dream k's leftovers."""
    model, opt, wake, seed = _tiny_setup(rows=4)
    dreams = [_cached(model, wake, seed), _cached(model, wake, seed)]
    seen = []
    model.forward = _recording(model, seen)

    distill_dream_set(model, opt, dreams, wake, variant="raw", epochs=1, kl_temp=1.0,
                      on_step=lambda s, l: None, on_boundary=lambda i, e, s: None)

    starts = [s for s in seen if s is not None]
    for start, dream in zip(starts, dreams, strict=True):
        torch.testing.assert_close(start, erased_start(wake, dream.bases["raw"]).ssm_states[0])


def test_the_dream_set_probes_at_every_boundary_and_carries_the_weights():
    model, opt, wake, seed = _tiny_setup(rows=4)
    dreams = [_cached(model, wake, seed) for _ in range(3)]
    before = [p.detach().clone() for p in model.parameters()]
    boundaries: list[int] = []

    distill_dream_set(model, opt, dreams, wake, variant=None, epochs=2, kl_temp=1.0,
                      on_step=lambda s, l: None,
                      on_boundary=lambda i, e, s: boundaries.append((e, i)))

    assert boundaries == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert _moved(model, before)  # one optimizer trajectory across the whole set


def test_fact_read_positions_are_the_bound_rehearsals_only():
    facts = [Fact("osprey", "bird", "5 9 7 9 7")]
    texts = ["The ", "code ", "for ", "the ", "osprey ", "is ", "5 9 7 9 7", "."]

    positions = fact_read_positions(texts, facts)

    assert positions["osprey"] == [6]
    assert fact_read_positions(["The ", "code ", "is ", "5 9 7 9 7", "."], facts)["osprey"] == []


def test_gate_agreement_scores_the_gate_against_the_binding_scan():
    """Diagnostic only (sec 2.10.6's named non-goal) -- but a zero per-fact
    contribution is loud."""
    agreement = gate_agreement([6, 7], {"osprey": [6], "heron": []})

    assert agreement["precision"] == 0.5 and agreement["recall"] == 1.0
    assert agreement["per_fact"] == {"osprey": 1, "heron": 0}


def test_a_dream_records_the_steer_prefix_it_was_seeded_with():
    """Sec 4: the prefix is what every arm's keep mask has to exclude, so the
    dream carries its length rather than each arm re-deriving it."""
    model, _, wake, _ = _tiny_setup()

    dream = teacher_dream(model, wake, torch.tensor([[1, 2, 3]]), n_tokens=6, temperature=0.0,
                          drain=False, decode_token=str, needles=[])

    assert dream.prefix_len == 3


def _builder_args(**over):
    from types import SimpleNamespace

    return SimpleNamespace(**{"dreams": 2, "seed": 1234, "dream_prompt": "", "dream_tokens": 6,
                              "dream_temp": 1.0, "gate_threshold": 0.0, "rank_rule": "ratio-gap",
                              "bind_min_dreams": 2, "pilot_capture": False,
                              "cue_every": 0, "cue_greedy": 0, **over})


def _build_set(tmp_path, decoded: str, battery=None, **over):
    """Drive the real cache builder on the fake backbone. `decoded` is what
    every token decodes to, which is how the binding scan is steered."""
    from types import SimpleNamespace

    from dream_sleep import build_dream_set

    model = TinyModel()
    path = tmp_path / "dream_set_s1234.pt"
    build_dream_set(model, _builder_args(**over), path, torch.tensor([[1, 2, 3]]),
                    [Fact("osprey", "bird", "5 9 7 9 7")], chunk_len=4,
                    encode=lambda text: torch.tensor([[1]]), decode=lambda ids: decoded,
                    tokenizer=SimpleNamespace(eos_token_id=None), user_open=USER, asst_open=ASST,
                    stop_id=None, battery=battery)
    return path


def test_a_failed_aggregate_gate_leaves_no_cache_on_disk(tmp_path):
    """Sec 4's gate has to REFUSE the cache, not annotate it: an arm cell
    prefers a set cache whenever one exists for the seed, so a refused build
    that left its file behind would poison every later cell silently."""
    with pytest.raises(SystemExit):
        _build_set(tmp_path, "nothing to see here. ")

    assert list(tmp_path.iterdir()) == []


def test_a_bound_set_is_written_with_its_sidecar(tmp_path):
    path = _build_set(tmp_path, "The code for the osprey is 5 9 7 9 7. ")

    cache = load_dream_cache(path)
    assert len(cache.dreams) == 2 and cache.set_sha
    assert all(d.gate_positions and d.bases["raw"] for d in cache.dreams)
    assert sidecar_path(path).exists()


def test_dream_epochs_interleave_full_passes_over_the_set():
    """Sec 2.10.2's variant cell is ~3 epochs over the SET: dream 1..N, then
    1..N again. Massed back-to-back repeats of each dream would rebuild the
    hundreds-of-passes-on-one-dream regime that cell exists to price against."""
    model, opt, wake, seed = _tiny_setup(rows=4)
    dreams = [_cached(model, wake, seed) for _ in range(3)]
    order: list[tuple[int, int]] = []

    distill_dream_set(model, opt, dreams, wake, variant=None, epochs=2, kl_temp=1.0,
                      on_step=lambda s, l: None,
                      on_boundary=lambda i, epoch, s: order.append((epoch, i)))

    assert order == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_every_epoch_restarts_each_dream_from_its_own_eraser():
    """The state reset is per boundary, not per set: epoch 2's dream k starts
    from the same erased copy epoch 1's did, never from epoch 1's leftovers."""
    model, opt, wake, seed = _tiny_setup(rows=4)
    dreams = [_cached(model, wake, seed) for _ in range(2)]
    seen = []
    model.forward = _recording(model, seen)

    distill_dream_set(model, opt, dreams, wake, variant="raw", epochs=2, kl_temp=1.0,
                      on_step=lambda s, l: None, on_boundary=lambda i, e, s: None)

    starts = [s for s in seen if s is not None]
    expected = [erased_start(wake, d.bases["raw"]).ssm_states[0] for d in dreams] * 2
    for start, want in zip(starts, expected, strict=True):
        torch.testing.assert_close(start, want)


# ---- the gate pilot's capture-everything path (sec 2.10.7) -----------------


def test_a_prod_cache_build_writes_no_pilot_capture(tmp_path):
    """Harness-only instrumentation: a prod build is byte-for-byte what it was."""
    path = _build_set(tmp_path, "The code for the osprey is 5 9 7 9 7. ")

    assert not pilot_path(path).exists()


def test_the_pilot_capture_holds_every_position_not_just_the_gated_ones(tmp_path):
    """Sec 2.10.7: every scheme is evaluated OFFLINE from one run, so the
    capture cannot be pre-filtered by the gate the prod cache happened to use."""
    path = _build_set(tmp_path, "The code for the osprey is 5 9 7 9 7. ", pilot_capture=True)

    cache, pilot = load_dream_cache(path), torch.load(pilot_path(path), weights_only=False)
    for cached, captured in zip(cache.dreams, pilot.dreams, strict=True):
        assert len(captured.queries) == len(cached.token_texts) > len(cached.gate_positions)
        assert len(captured.divergence) == len(cached.token_texts)
        assert all(len(per_layer) == 2 for per_layer in captured.queries)


def test_the_pilot_capture_carries_the_state_and_facts_the_scorer_needs(tmp_path):
    path = _build_set(tmp_path, "The code for the osprey is 5 9 7 9 7. ", pilot_capture=True)

    pilot = torch.load(pilot_path(path), weights_only=False)

    assert pilot.wake_state.ssm_states and pilot.facts == [("osprey", "bird", "5 9 7 9 7")]


def test_the_pilot_capture_records_battery_read_queries_on_gpu_not_offline(tmp_path):
    """The collateral pool sec 2.10.6 scores against needs a forward pass, so it
    is captured here; the offline scorer only reads it."""
    battery = [{"prompt": "q1", "answer": "a"}, {"prompt": "q2", "answer": "b"}]
    path = _build_set(tmp_path, "The code for the osprey is 5 9 7 9 7. ",
                      pilot_capture=True, battery=battery)

    pilot = torch.load(pilot_path(path), weights_only=False)

    assert sorted(pilot.battery_queries) == ["q1", "q2"]
    assert all(len(per_layer) == 2 for positions in pilot.battery_queries.values()
               for per_layer in positions)


def test_dream_is_degenerate_flags_mojibake_and_tolerates_ordinary_unicode():
    from dream_sleep import dream_is_degenerate

    assert dream_is_degenerate("�" * 16)
    assert dream_is_degenerate("fine text " + "�" + " more")   # any replacement char
    assert dream_is_degenerate("ñ" * 100)                            # non-ASCII flood
    assert not dream_is_degenerate("The code for the osprey is 5 9 7 9 7.")
    assert not dream_is_degenerate('He said “hello” — that’s fine.')


def test_a_degenerate_dream_is_regenerated_under_a_bumped_seed(tmp_path, monkeypatch, capsys):
    """Sec 2.1's mojibake clause, enforced per dream: a degenerate dream never
    enters the set; the slot is regenerated and counted, loudly."""
    import dream_sleep as ds

    real = ds.teacher_dream
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        dream = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            dream.token_texts = ["�"] * len(dream.token_texts)
        return dream

    monkeypatch.setattr(ds, "teacher_dream", flaky)
    path = _build_set(tmp_path, "The code for the osprey is 5 9 7 9 7. ")

    cache = load_dream_cache(path)
    assert len(cache.dreams) == 2
    assert not any("�" in "".join(d.token_texts) for d in cache.dreams)
    assert calls["n"] == 3  # two slots, one regeneration
    assert "degenerate" in capsys.readouterr().out


def test_a_slot_that_stays_degenerate_is_a_stop_and_think(tmp_path, monkeypatch):
    import dream_sleep as ds

    real = ds.teacher_dream

    def always_bad(*args, **kwargs):
        dream = real(*args, **kwargs)
        dream.token_texts = ["�"] * len(dream.token_texts)
        return dream

    monkeypatch.setattr(ds, "teacher_dream", always_bad)
    with pytest.raises(SystemExit, match="degenerate"):
        _build_set(tmp_path, "The code for the osprey is 5 9 7 9 7. ")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to cross devices")
def test_battery_read_queries_follow_the_model_device_not_the_callers():
    """build_dream_set hands over a wake state that state_to already moved to
    CPU for the cache; the battery forward must run wherever the model is
    (sec 1.8 -- the all-CPU fake backbone can never catch this)."""
    from dream_sleep import battery_read_queries

    model, _, wake, _ = _tiny_setup()
    model = model.to("cuda")
    out = battery_read_queries(model, [{"prompt": "hi"}], wake,
                               encode=lambda text: torch.tensor([[1, 2]], device="cuda"),
                               n_layers=len(model.layers))
    assert list(out) == ["hi"] and len(out["hi"]) == 2


def test_an_empty_gate_yields_an_empty_eraser_not_a_failure(tmp_path):
    """The gate selects B4's queries; it is NOT a dream-validity requirement
    (altrup, 2026-08-10). A dream the state never influenced has nothing to
    deny: its eraser is empty and its B4 start state is the intact wake state."""
    path = _build_set(tmp_path, "The code for the osprey is 5 9 7 9 7. ",
                      gate_threshold=1e9)

    cache = load_dream_cache(path)
    assert len(cache.dreams) == 2
    for dream in cache.dreams:
        assert dream.gate_positions == [] and dream.queries == []
        assert all(b.shape[0] == 0 for b in dream.bases["raw"])
    wake = FakeState([torch.randn(1, 2, 1, 4) for _ in range(len(cache.dreams[0].bases["raw"]))])
    torch.testing.assert_close(
        erased_start(wake, cache.dreams[0].bases["raw"]).ssm_states[0], wake.ssm_states[0])


def test_a_single_query_dream_with_an_empty_qcm_basis_still_builds(tmp_path, capsys):
    """qcm drops v1; with one gated query that leaves nothing. An empty
    variant basis is a loud NOTE, never a refused cache."""
    from dream_sleep import dream_bases

    wake = FakeState([torch.randn(1, 2, 1, 4)])
    queries = [[torch.tensor([[1.0, 0.0, 0.0, 0.0]])]]
    _, _, bases = dream_bases(queries, [0], wake, "ratio-gap")

    assert bases["qcm"][0].shape[0] == 0
    assert "empty" in capsys.readouterr().out


def test_dream_seed_offset_makes_parallel_builds_disjoint(tmp_path):
    """A set's generation seeds are derived from --seed, so two processes on
    the same wake state produce IDENTICAL dreams -- parallel builds would
    duplicate instead of extending the set. The offset shifts the derivation
    so N processes cover disjoint dream indices."""
    from dream_sleep import dream_generation_seed

    base = [dream_generation_seed(1234, i, attempt=0, offset=0) for i in range(20)]
    shifted = [dream_generation_seed(1234, i, attempt=0, offset=20) for i in range(20)]

    assert len(set(base)) == 20
    assert not set(base) & set(shifted)
    # a retry must not collide with another process's slot either
    retry = dream_generation_seed(1234, 0, attempt=1, offset=0)
    assert retry not in set(base) | set(shifted)
