"""Fast (CPU, seconds) tests for sft/train.py's generic, model-agnostic
training machinery: the chunk slicer, mid-example state replay on resume,
and the token-based checkpoint trigger. Exercised against a tiny fake
stateful model + fake hooks module instead of a real Mamba2 model, since
none of this logic is specific to any one model -- that's the whole point of
it living here rather than in a model's train_hooks.py.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

import train


VOCAB = 8


class FakeStatefulModel(torch.nn.Module):
    """forward(input_ids, state=None) -> (logits, state), mirroring the real
    chunked models' signature. `state` is a per-row (shape (B,)) running
    count of tokens consumed so far in each slot's current example -- enough
    to verify state is threaded/detached/batched correctly, and (critically)
    that a slot's count doesn't drift from continuing to be fed dummy
    zero-padding after it's done, without needing a real recurrent
    architecture. Always adds a full chunk's width regardless of masking,
    same as a real model's forward would touch every physical batch
    position every call -- callers should only assert on rows they expect
    to have been fed real (unpadded) chunks throughout."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, 4)
        self.head = torch.nn.Linear(4, VOCAB)

    def forward(self, input_ids, state=None):
        x = self.embed(input_ids)
        logits = self.head(x)
        batch_size = input_ids.shape[0]
        prior = state if state is not None else torch.zeros(batch_size)
        new_state = prior + input_ids.shape[1]
        return logits, new_state


class FakeHooks:
    DEFAULT_CHUNK_LEN = 4

    @staticmethod
    def chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight):
        logits, state = model(input_ids, state=state)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="none")
        weight = mask_slice.reshape(-1).float() if mask_slice is not None else torch.ones_like(loss)
        return (loss * weight).sum(), weight.sum(), state


def _ids(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=g)


# ---------------------------------------------------------------------------
# dataset_fingerprint -- lets resume detect a --data swap
# ---------------------------------------------------------------------------

def test_dataset_fingerprint_differs_for_different_paths_or_sizes(tmp_path):
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    assert train.dataset_fingerprint(str(a), 10) != train.dataset_fingerprint(str(b), 10)
    assert train.dataset_fingerprint(str(a), 10) != train.dataset_fingerprint(str(a), 11)
    assert train.dataset_fingerprint(str(a), 10) == train.dataset_fingerprint(str(a), 10)


def test_save_checkpoint_stores_given_dataset_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    slots = [train._Slot(0, 0, _ids(5), None)]
    fp = train.dataset_fingerprint("some/train.pt", 42)

    path = train.save_checkpoint(
        model, optimizer, step=1, epoch=0, slots=slots, next_ptr=1,
        total_tokens=1.0, lora_rank=4, lora_alpha=8.0,
        dataset_fingerprint=fp,
    )

    state = torch.load(path / "state.pt", weights_only=True)
    assert state["dataset_fingerprint"] == fp


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint round trip with slot_states + token counters
# ---------------------------------------------------------------------------

def test_checkpoint_state_round_trips_slot_states_and_token_counters(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    slots = [train._Slot(0, 12, _ids(5), None)]
    slots[0].pos = 8

    path = train.save_checkpoint(
        model, optimizer, step=5, epoch=0, slots=slots, next_ptr=13,
        total_tokens=123.0, lora_rank=4, lora_alpha=8.0,
    )

    state = torch.load(path / "state.pt", weights_only=True)
    # last_ckpt_tokens always equals this checkpoint's own total_tokens --
    # it's the anchor a future resume counts --ckpt-every-tokens from, not
    # a record of whatever the caller's baseline was before this save (see
    # save_checkpoint's docstring for why that distinction matters).
    assert state == {
        "epoch": 0, "slot_states": [(12, 8)], "next_ptr": 13,
        "dataset_fingerprint": None,
        "total_tokens": 123.0, "last_ckpt_tokens": 123.0,
    }


# ---------------------------------------------------------------------------
# save_checkpoint / rotate_full_state -- optional full internal state
# ---------------------------------------------------------------------------

def test_save_checkpoint_writes_mem_state_when_batched_state_given(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    slots = [train._Slot(0, 3, _ids(5), None)]
    slots[0].pos = 2

    path = train.save_checkpoint(
        model, optimizer, step=1, epoch=0, slots=slots, next_ptr=1,
        total_tokens=10.0, lora_rank=4, lora_alpha=8.0,
        batched_state=torch.tensor([1.0, 2.0]),
    )

    assert (path / "mem_state.pt").exists()
    state = torch.load(path / "state.pt", weights_only=True)
    assert state["state_batch_size"] == 1


def test_save_checkpoint_omits_mem_state_when_batched_state_is_none(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    slots = [train._Slot(0, 3, _ids(5), None)]

    path = train.save_checkpoint(
        model, optimizer, step=1, epoch=0, slots=slots, next_ptr=1,
        total_tokens=10.0, lora_rank=4, lora_alpha=8.0,
    )

    assert not (path / "mem_state.pt").exists()
    state = torch.load(path / "state.pt", weights_only=True)
    assert "state_batch_size" not in state


def test_save_checkpoint_crashing_mid_save_leaves_previous_checkpoint_newest(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    slots = [train._Slot(0, 3, _ids(5), None)]
    good = train.save_checkpoint(
        model, optimizer, step=1, epoch=0, slots=slots, next_ptr=1,
        total_tokens=10.0, lora_rank=4, lora_alpha=8.0,
    )

    real_save = torch.save

    def fail_writing_state(obj, f, *args, **kwargs):
        if Path(f).name == "state.pt":
            raise RuntimeError("simulated crash mid-save")
        return real_save(obj, f, *args, **kwargs)

    monkeypatch.setattr(train.torch, "save", fail_writing_state)

    with pytest.raises(RuntimeError):
        train.save_checkpoint(
            model, optimizer, step=2, epoch=0, slots=slots, next_ptr=1,
            total_tokens=20.0, lora_rank=4, lora_alpha=8.0,
        )

    assert [s for s, _ in train.iter_checkpoints()] == [1]
    assert train.latest_checkpoint() == good


def test_rotate_full_state_keeps_mem_state_only_in_newest_n(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    slots = [train._Slot(0, 0, _ids(5), None)]
    paths = [
        train.save_checkpoint(
            model, optimizer, step=step, epoch=0, slots=slots, next_ptr=1,
            total_tokens=float(step), lora_rank=4, lora_alpha=8.0,
            batched_state=torch.tensor([1.0]),
        )
        for step in (1, 2, 3)
    ]

    train.rotate_full_state(keep=2)

    assert not (paths[0] / "mem_state.pt").exists()
    assert (paths[1] / "mem_state.pt").exists()
    assert (paths[2] / "mem_state.pt").exists()
    # the rest of the checkpoint (trainable.pt, etc.) is untouched by pruning.
    assert (paths[0] / "trainable.pt").exists()


def test_rotate_full_state_zero_removes_all(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    slots = [train._Slot(0, 0, _ids(5), None)]
    path = train.save_checkpoint(
        model, optimizer, step=1, epoch=0, slots=slots, next_ptr=1,
        total_tokens=1.0, lora_rank=4, lora_alpha=8.0,
        batched_state=torch.tensor([1.0]),
    )

    train.rotate_full_state(keep=0)

    assert not (path / "mem_state.pt").exists()


# ---------------------------------------------------------------------------
# main training loop: token-based checkpoint cadence, mid-example resume
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    defaults = dict(
        # accum_tokens=4 with chunk_len=4 derives to accum_steps=1 (see
        # run_training's accum_tokens -> accum_steps derivation), matching
        # the old accum_steps=1 default these tests were written against.
        epochs=1, eos_weight=1.0, recall_weight=1.0, head_weight=1.0, head_tokens=1024,
        accum_tokens=4, chunk_len=4,
        ckpt_every_tokens=8, keep_ckpts=5, keep_full_state=5, lora_rank=4, lora_alpha=8.0,
        max_len=float("inf"), batch_size=1, data="fake_dataset.pt",
        # warmup_steps=0 disables warmup so these tests keep the constant
        # --lr they were written against.
        warmup_steps=0, lr=1e-3,
    )
    defaults.update(overrides)
    from types import SimpleNamespace
    return SimpleNamespace(**defaults)


def test_run_training_checkpoints_mid_example_and_resume_continues_same_example(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    torch.manual_seed(0)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainable_params = list(model.parameters())

    # One long example (28 tokens -> 27 targets -> 7 chunks of 4) so a small
    # token threshold forces a checkpoint to land mid-example.
    train_ids = [_ids(28, seed=7)]
    train_masks = [None]
    args = _make_args(ckpt_every_tokens=8)  # 8 tokens = 2 chunks in

    train.run_training(
        FakeHooks, model, optimizer, trainable_params, train_ids, train_masks, [None] * len(train_ids), [None] * len(train_ids), "cpu", args,
        start_epoch=0, start_slot_states=None, start_next_ptr=0, start_step=0,
        start_total_tokens=0.0, start_last_ckpt_tokens=0.0,
    )

    ckpts = sorted(train.iter_checkpoints())
    assert len(ckpts) > 0
    _, first_ckpt_path = ckpts[0]
    state = torch.load(first_ckpt_path / "state.pt", weights_only=True)
    example_idx, pos = state["slot_states"][0]
    assert pos > 0, "first checkpoint should land mid-example given the small token threshold"
    assert example_idx == 0


def test_run_training_resume_from_mid_example_checkpoint_continues_without_crashing(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    torch.manual_seed(0)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainable_params = list(model.parameters())

    train_ids = [_ids(28, seed=7)]
    train_masks = [None]
    args = _make_args(ckpt_every_tokens=8)

    train.run_training(
        FakeHooks, model, optimizer, trainable_params, train_ids, train_masks, [None] * len(train_ids), [None] * len(train_ids), "cpu", args,
        start_epoch=0, start_slot_states=None, start_next_ptr=0, start_step=0,
        start_total_tokens=0.0, start_last_ckpt_tokens=0.0,
    )
    ckpt = train.latest_checkpoint()
    saved_state = torch.load(ckpt / "state.pt", weights_only=True)

    # Fresh model/optimizer, as a real resume would start with.
    model2 = FakeStatefulModel()
    train.load_checkpoint(model2, ckpt)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

    train.run_training(
        FakeHooks, model2, optimizer2, list(model2.parameters()), train_ids, train_masks, [None] * len(train_ids), [None] * len(train_ids), "cpu", args,
        start_epoch=saved_state["epoch"], start_slot_states=saved_state["slot_states"],
        start_next_ptr=saved_state["next_ptr"], start_step=0,
        start_total_tokens=saved_state["total_tokens"],
        start_last_ckpt_tokens=saved_state["last_ckpt_tokens"],
    )

    final_ckpt = train.latest_checkpoint()
    final_state = torch.load(final_ckpt / "state.pt", weights_only=True)
    assert final_state["total_tokens"] >= 27  # whole 28-token example (27 targets) eventually trained on


def test_run_training_resume_without_saved_state_restarts_mid_example_slot_from_zero(monkeypatch, tmp_path):
    """Regression: with no saved internal state (start_full_state=None), a
    slot that was mid-example at checkpoint time must restart that example
    from position 0 rather than continuing from its saved position --
    reconstructing the carried state by replaying the prefix was tried and
    dropped (see run_training's resume comment)."""
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_ids = [_ids(28, seed=7)]  # 27 targets, chunk_len=4
    train_masks = [None]
    # ckpt_every_tokens=4 checkpoints every chunk (7 total) -- keep_ckpts
    # must exceed that so rotate_checkpoints doesn't prune away the first
    # one before we can inspect it.
    args = _make_args(ckpt_every_tokens=4, keep_ckpts=99)

    train.run_training(
        FakeHooks, model, optimizer, list(model.parameters()), train_ids, train_masks, [None] * len(train_ids), [None] * len(train_ids), "cpu", args,
        start_epoch=0, start_slot_states=[(0, 12)], start_next_ptr=0, start_step=0,
        start_total_tokens=12.0, start_last_ckpt_tokens=12.0,
    )

    ckpts = sorted(train.iter_checkpoints())
    _, first_ckpt_path = ckpts[0]
    state = torch.load(first_ckpt_path / "state.pt", weights_only=True)
    example_idx, pos = state["slot_states"][0]
    assert example_idx == 0
    assert pos == 4, "should restart from 0 (then advance one chunk), not continue from the saved pos 12"


def test_run_training_resume_at_example_boundary_starts_next_example_fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_ids = [_ids(13, seed=1), _ids(9, seed=2)]
    train_masks = [None, None]
    # First example (13 tokens -> 12 targets -> 3 chunks of 4) finishes
    # exactly at 12 tokens; a checkpoint threshold of 12 lands right as the
    # slot rolls over to the second example, still at pos 0 -- the final
    # (only) forced save would instead land after BOTH examples finish, with
    # the slot back to None, which isn't the boundary this test is about.
    args = _make_args(ckpt_every_tokens=12)

    train.run_training(
        FakeHooks, model, optimizer, list(model.parameters()), train_ids, train_masks, [None] * len(train_ids), [None] * len(train_ids), "cpu", args,
        start_epoch=0, start_slot_states=None, start_next_ptr=0, start_step=0,
        start_total_tokens=0.0, start_last_ckpt_tokens=0.0,
    )

    ckpts = sorted(train.iter_checkpoints())
    assert len(ckpts) > 0
    _, first_ckpt_path = ckpts[0]
    state = torch.load(first_ckpt_path / "state.pt", weights_only=True)
    example_idx, pos = state["slot_states"][0]
    assert pos == 0
    assert example_idx == 1


# ---------------------------------------------------------------------------
# per-token loss weighting: --recall-weight / --head-weight
# ---------------------------------------------------------------------------

class _RecordingHooks(FakeHooks):
    """FakeHooks that records every weight tensor passed to chunk_loss."""

    recorded: list

    @classmethod
    def chunk_loss(cls, model, input_ids, target_ids, mask_slice, state, eos_weight):
        cls.recorded.append(mask_slice.clone())
        return FakeHooks.chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight)


def _run_with_weights(tmp_path, monkeypatch, recall, args):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    torch.manual_seed(0)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_ids = [_ids(13, seed=3)]

    class Hooks(_RecordingHooks):
        recorded = []

    train.run_training(
        Hooks, model, optimizer, list(model.parameters()), train_ids, [None], [recall], [None], "cpu", args,
        start_epoch=0, start_slot_states=None, start_next_ptr=0, start_step=0,
        start_total_tokens=0.0, start_last_ckpt_tokens=0.0,
    )
    # (n_chunks, chunk_len) weights for the single slot, target positions 1..12
    return torch.cat([w[0] for w in Hooks.recorded])


def test_recall_weight_boosts_exactly_the_marked_target_tokens(monkeypatch, tmp_path):
    recall = torch.zeros(13, dtype=torch.bool)
    recall[[5, 6]] = True
    weights = _run_with_weights(tmp_path, monkeypatch, recall, _make_args(recall_weight=3.0))
    expected = torch.ones(12)
    # weight index i covers target token at absolute position i + 1
    expected[[4, 5]] = 3.0
    assert torch.equal(weights, expected)


def test_head_weight_ramps_down_linearly_over_head_tokens(monkeypatch, tmp_path):
    weights = _run_with_weights(
        tmp_path, monkeypatch, None, _make_args(head_weight=5.0, head_tokens=8)
    )
    tpos = torch.arange(1, 13, dtype=torch.float32)
    expected = 1.0 + 4.0 * (1.0 - tpos / 8).clamp(min=0.0)
    assert torch.allclose(weights, expected)


# ---------------------------------------------------------------------------
# sleep_positions: mid-example backbone resets (episodic chains)
# ---------------------------------------------------------------------------

class _SleepRecordingHooks(_RecordingHooks):
    """Adds a sleep_slot hook that records each firing as (slot_idx, chunks
    seen so far) -- the chunk count pins down *when* the sleep fired relative
    to the chunk stream, which is the snapping behavior under test."""

    fired: list

    @classmethod
    def sleep_slot(cls, model, state, slot_idx):
        cls.fired.append((slot_idx, len(cls.recorded)))


def _run_with_sleeps(tmp_path, monkeypatch, sleeps, args, n_tokens=13):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    torch.manual_seed(0)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_ids = [_ids(n_tokens, seed=3)]

    class Hooks(_SleepRecordingHooks):
        recorded = []
        fired = []

    train.run_training(
        Hooks, model, optimizer, list(model.parameters()), train_ids, [None], [None], [sleeps], "cpu", args,
        start_epoch=0, start_slot_states=None, start_next_ptr=0, start_step=0,
        start_total_tokens=0.0, start_last_ckpt_tokens=0.0,
    )
    return Hooks


def test_sleep_fires_once_at_first_chunk_boundary_past_its_offset(monkeypatch, tmp_path):
    # chunk_len=4, sleep offset 6: pos hits 4 (< 6, no fire), then 8 -- the
    # sleep fires exactly once, before the third chunk is built.
    hooks = _run_with_sleeps(tmp_path, monkeypatch, torch.tensor([6]), _make_args())
    assert hooks.fired == [(0, 2)]


def test_head_weight_ramp_restarts_at_fired_sleep(monkeypatch, tmp_path):
    # Sleep at offset 6 fires at the chunk boundary pos=8, so the ramp's
    # origin moves to 8: target positions 9..12 are 1..4 tokens post-reset.
    hooks = _run_with_sleeps(
        tmp_path, monkeypatch, torch.tensor([6]), _make_args(head_weight=5.0, head_tokens=8)
    )
    weights = torch.cat([w[0] for w in hooks.recorded])
    tpos = torch.cat([torch.arange(1, 9), torch.arange(1, 5)]).float()
    expected = 1.0 + 4.0 * (1.0 - tpos / 8).clamp(min=0.0)
    assert torch.allclose(weights, expected)


def test_resume_does_not_refire_sleeps_already_reflected_in_saved_state(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    torch.manual_seed(0)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_ids = [_ids(13, seed=3)]
    full_state = tmp_path / "mem_state_fake.pt"
    torch.save(torch.zeros(1), full_state)

    class Hooks(_SleepRecordingHooks):
        recorded = []
        fired = []

    train.run_training(
        Hooks, model, optimizer, list(model.parameters()), train_ids, [None], [None],
        [torch.tensor([6])], "cpu", _make_args(),
        start_epoch=0, start_slot_states=[(0, 8)], start_next_ptr=1, start_step=0,
        start_total_tokens=8.0, start_last_ckpt_tokens=8.0, start_full_state=full_state,
    )
    assert Hooks.fired == []


# ---------------------------------------------------------------------------
# --max-steps: stop after N optimizer steps
# ---------------------------------------------------------------------------

def _run_to_max_steps(monkeypatch, tmp_path, max_steps, start_step=0):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    torch.manual_seed(0)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # 10 examples of 41 tokens: 100 chunks of 4, far more than any step budget here.
    train_ids = [_ids(41, seed=i) for i in range(10)]
    args = _make_args(max_steps=max_steps, ckpt_every_tokens=10**9)

    train.run_training(
        FakeHooks, model, optimizer, list(model.parameters()), train_ids, [None] * 10,
        [None] * 10, [None] * 10, "cpu", args,
        start_epoch=0, start_slot_states=None, start_next_ptr=0, start_step=start_step,
        start_total_tokens=0.0, start_last_ckpt_tokens=0.0,
    )
    return max(step for step, _ in train.iter_checkpoints())


def test_max_steps_stops_the_run_at_that_many_optimizer_steps(monkeypatch, tmp_path):
    assert _run_to_max_steps(monkeypatch, tmp_path, max_steps=3) == 3


def test_max_steps_counts_global_step_so_a_resumed_run_finishes_the_budget(monkeypatch, tmp_path):
    """The budget is a global_step ceiling, not "N more steps": resuming a
    run that already took 2 of 3 steps leaves exactly one to take."""
    assert _run_to_max_steps(monkeypatch, tmp_path, max_steps=3, start_step=2) == 3


# ---------------------------------------------------------------------------
# load_checkpoint -- marker_delta rows grown by a new special token
# ---------------------------------------------------------------------------

class _MarkerHolder(torch.nn.Module):
    def __init__(self, rows: int):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros(rows, 4))


class _MarkerModel(torch.nn.Module):
    def __init__(self, rows: int):
        super().__init__()
        self.marker_delta = _MarkerHolder(rows)


def test_load_checkpoint_pads_marker_delta_grown_by_a_new_special_token(tmp_path, capsys):
    old = _MarkerModel(rows=2)
    with torch.no_grad():
        old.marker_delta.delta.copy_(torch.arange(8.0).reshape(2, 4))
    torch.save(old.state_dict(), tmp_path / "trainable.pt")

    new = _MarkerModel(rows=3)
    train.load_checkpoint(new, tmp_path)

    assert torch.equal(new.marker_delta.delta[:2], old.marker_delta.delta)
    assert torch.equal(new.marker_delta.delta[2], torch.zeros(4))
    assert "marker_delta" in capsys.readouterr().out  # the pad is loud


def test_load_checkpoint_still_rejects_a_shrunk_marker_delta(tmp_path):
    old = _MarkerModel(rows=3)
    torch.save(old.state_dict(), tmp_path / "trainable.pt")

    with pytest.raises(RuntimeError):
        train.load_checkpoint(_MarkerModel(rows=2), tmp_path)
