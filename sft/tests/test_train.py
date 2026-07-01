"""Fast (CPU, seconds) tests for sft/train.py's generic, model-agnostic
training machinery: the chunk slicer, evaluate()/preflight(), mid-example
state replay on resume, and the token-based checkpoint trigger. Exercised
against a tiny fake stateful model + fake hooks module instead of a real
Mamba2 model, since none of this logic is specific to any one model -- that's
the whole point of it living here rather than in a model's train_hooks.py.
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
    chunked models' signature. `state` is just a running count of tokens
    consumed so far in the current example -- enough to verify state is
    threaded/detached correctly and that replay reconstructs it exactly,
    without needing a real recurrent architecture."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, 4)
        self.head = torch.nn.Linear(4, VOCAB)

    def forward(self, input_ids, state=None):
        x = self.embed(input_ids)
        logits = self.head(x)
        prior = state if state is not None else torch.tensor(0.0)
        new_state = prior + input_ids.numel()
        return logits, new_state


class FakeHooks:
    DEFAULT_CHUNK_LEN = 4

    @staticmethod
    def chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight):
        logits, state = model(input_ids, state=state)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="none")
        weight = mask_slice.float() if mask_slice is not None else torch.ones_like(loss)
        return (loss * weight).sum(), weight.sum(), state


def _ids(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator=g)


# ---------------------------------------------------------------------------
# _chunks
# ---------------------------------------------------------------------------

def test_chunks_splits_into_chunk_len_pieces():
    ids = _ids(13)  # seqlen-1 = 12 -> 3 chunks of 4
    chunks = list(train._chunks(ids, None, chunk_len=4))

    assert [c[0] for c in chunks] == [0, 4, 8]
    assert [c[1] for c in chunks] == [4, 8, 12]


def test_chunks_input_target_are_shifted_by_one():
    ids = _ids(6)
    start, end, input_ids, target_ids, _ = next(train._chunks(ids, None, chunk_len=4))

    assert torch.equal(input_ids.squeeze(0), ids[0:4])
    assert torch.equal(target_ids.squeeze(0), ids[1:5])


def test_chunks_slices_mask_to_match_targets():
    ids = _ids(6)
    mask = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool)
    _, _, _, _, mask_slice = next(train._chunks(ids, mask, chunk_len=4))

    assert torch.equal(mask_slice, mask[1:5])


def test_chunks_resumes_from_start_pos():
    ids = _ids(13)
    chunks = list(train._chunks(ids, None, chunk_len=4, start_pos=4))

    assert [c[0] for c in chunks] == [4, 8]


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def test_evaluate_returns_finite_loss_and_does_not_change_params():
    model = FakeStatefulModel()
    before = {n: p.clone() for n, p in model.named_parameters()}
    eval_ids = [_ids(9, seed=1), _ids(5, seed=2)]
    eval_masks = [None, None]

    loss = train.evaluate(FakeHooks, model, eval_ids, eval_masks, "cpu", max_len=100, chunk_len=4)

    assert torch.isfinite(torch.tensor(loss))
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n])


def test_evaluate_skips_examples_with_an_all_false_mask():
    model = FakeStatefulModel()
    eval_ids = [_ids(9, seed=1)]
    eval_masks = [torch.zeros(9, dtype=torch.bool)]

    loss = train.evaluate(FakeHooks, model, eval_ids, eval_masks, "cpu", max_len=100, chunk_len=4)

    assert loss != loss  # nan: nothing contributed


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def test_preflight_passes_when_gradients_reach_trainable_params():
    model = FakeStatefulModel()
    trainable_params = list(model.parameters())
    all_ids = [_ids(9, seed=1)]
    all_masks = [None]

    train.preflight(FakeHooks, model, trainable_params, all_ids, all_masks, "cpu", max_len=100, eos_weight=1.0, chunk_len=4)
    # preflight zeroes grads itself when done; reaching here without an
    # AssertionError is the pass condition.


def test_preflight_shows_live_progress_when_hooks_define_chunk_extra_log(monkeypatch):
    """Regression: preflight used to go through process_example, which had
    its own live per-chunk progress display for slow models -- that display
    must not be lost now that preflight no longer calls process_example."""
    calls = []
    monkeypatch.setattr(train, "_print_live", lambda lines, prev_n_lines: calls.append(lines) or len(lines))

    class HooksWithChunkExtraLog(FakeHooks):
        @staticmethod
        def chunk_extra_log(model):
            return "extra status"

    model = FakeStatefulModel()
    all_ids = [_ids(9, seed=1)]  # seqlen-1=8, chunk_len=4 -> 2 chunks
    all_masks = [None]

    train.preflight(HooksWithChunkExtraLog, model, list(model.parameters()), all_ids, all_masks, "cpu", max_len=100, eos_weight=1.0, chunk_len=4)

    assert len(calls) == 2
    assert "extra status" in calls[0][1]


def test_preflight_shows_no_live_progress_when_hooks_lack_chunk_extra_log(monkeypatch):
    calls = []
    monkeypatch.setattr(train, "_print_live", lambda lines, prev_n_lines: calls.append(lines) or len(lines))

    model = FakeStatefulModel()
    all_ids = [_ids(9, seed=1)]
    all_masks = [None]

    train.preflight(FakeHooks, model, list(model.parameters()), all_ids, all_masks, "cpu", max_len=100, eos_weight=1.0, chunk_len=4)

    assert calls == []


def test_preflight_raises_when_no_example_fits_max_len():
    model = FakeStatefulModel()
    all_ids = [_ids(9, seed=1)]
    all_masks = [None]

    with pytest.raises(AssertionError):
        train.preflight(FakeHooks, model, list(model.parameters()), all_ids, all_masks, "cpu", max_len=2, eos_weight=1.0, chunk_len=4)


# ---------------------------------------------------------------------------
# replay_state
# ---------------------------------------------------------------------------

def test_replay_state_returns_none_at_chunk_pos_zero():
    model = FakeStatefulModel()
    ids = _ids(13)

    state, prev_n_lines = train.replay_state(FakeHooks, model, ids, None, chunk_len=4, chunk_pos=0, device="cpu")

    assert state is None
    assert prev_n_lines == 0


def test_replay_state_reconstructs_state_identical_to_uninterrupted_run():
    model = FakeStatefulModel()
    ids = _ids(13)  # 3 chunks: [0:4] [4:8] [8:12]

    # Uninterrupted: thread state through all chunks up to (not including) the third.
    state = None
    for start, end, input_ids, target_ids, mask_slice in train._chunks(ids, None, chunk_len=4):
        if start >= 8:
            break
        _, _, state = FakeHooks.chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight=1.0)
        state = state.detach()
    expected = state

    replayed, _ = train.replay_state(FakeHooks, model, ids, None, chunk_len=4, chunk_pos=8, device="cpu")

    assert torch.equal(replayed, expected)


def test_replay_state_does_not_require_grad():
    model = FakeStatefulModel()
    ids = _ids(13)

    state, _ = train.replay_state(FakeHooks, model, ids, None, chunk_len=4, chunk_pos=8, device="cpu")

    assert not state.requires_grad


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint round trip with chunk_pos + token counters
# ---------------------------------------------------------------------------

def test_checkpoint_state_round_trips_chunk_pos_and_token_counters(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    path = train.save_checkpoint(
        model, optimizer, step=5, epoch=0, example_idx=12, chunk_pos=8,
        total_tokens=123.0, last_ckpt_tokens=100.0, lora_rank=4, lora_alpha=8.0,
    )

    state = torch.load(path / "state.pt", weights_only=True)
    assert state == {"epoch": 0, "example_idx": 12, "chunk_pos": 8, "total_tokens": 123.0, "last_ckpt_tokens": 100.0}


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
        total_tokens=10.0, last_ckpt_tokens=0.0, lora_rank=4, lora_alpha=8.0,
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
        total_tokens=10.0, last_ckpt_tokens=0.0, lora_rank=4, lora_alpha=8.0,
    )

    assert not (path / "mem_state.pt").exists()
    state = torch.load(path / "state.pt", weights_only=True)
    assert "state_batch_size" not in state


def test_rotate_full_state_keeps_mem_state_only_in_newest_n(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    slots = [train._Slot(0, 0, _ids(5), None)]
    paths = [
        train.save_checkpoint(
            model, optimizer, step=step, epoch=0, slots=slots, next_ptr=1,
            total_tokens=float(step), last_ckpt_tokens=0.0, lora_rank=4, lora_alpha=8.0,
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
        total_tokens=1.0, last_ckpt_tokens=0.0, lora_rank=4, lora_alpha=8.0,
        batched_state=torch.tensor([1.0]),
    )

    train.rotate_full_state(keep=0)

    assert not (path / "mem_state.pt").exists()


# ---------------------------------------------------------------------------
# main training loop: token-based checkpoint cadence, mid-example resume
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    defaults = dict(
        epochs=1, eos_weight=1.0, accum_steps=1, chunk_len=4,
        ckpt_every_tokens=8, keep_ckpts=5, keep_full_state=5, lora_rank=4, lora_alpha=8.0,
        max_len=float("inf"),
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
    eval_ids, eval_masks = [], []
    args = _make_args(ckpt_every_tokens=8)  # 8 tokens = 2 chunks in

    train.run_training(
        FakeHooks, model, optimizer, trainable_params, train_ids, train_masks, eval_ids, eval_masks, "cpu", args,
        start_epoch=0, start_example=0, start_step=0, start_chunk_pos=0, start_total_tokens=0.0, start_last_ckpt_tokens=0.0,
    )

    ckpts = sorted(train.iter_checkpoints())
    assert len(ckpts) > 0
    _, first_ckpt_path = ckpts[0]
    state = torch.load(first_ckpt_path / "state.pt", weights_only=True)
    assert state["chunk_pos"] > 0, "first checkpoint should land mid-example given the small token threshold"
    assert state["example_idx"] == 0


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
        FakeHooks, model, optimizer, trainable_params, train_ids, train_masks, [], [], "cpu", args,
        start_epoch=0, start_example=0, start_step=0, start_chunk_pos=0, start_total_tokens=0.0, start_last_ckpt_tokens=0.0,
    )
    ckpt = train.latest_checkpoint()
    saved_state = torch.load(ckpt / "state.pt", weights_only=True)

    # Fresh model/optimizer, as a real resume would start with.
    model2 = FakeStatefulModel()
    train.load_checkpoint(model2, ckpt)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

    train.run_training(
        FakeHooks, model2, optimizer2, list(model2.parameters()), train_ids, train_masks, [], [], "cpu", args,
        start_epoch=saved_state["epoch"], start_example=saved_state["example_idx"], start_step=0,
        start_chunk_pos=saved_state["chunk_pos"], start_total_tokens=saved_state["total_tokens"],
        start_last_ckpt_tokens=saved_state["last_ckpt_tokens"],
    )

    final_ckpt = train.latest_checkpoint()
    final_state = torch.load(final_ckpt / "state.pt", weights_only=True)
    assert final_state["total_tokens"] >= 27  # whole 28-token example (27 targets) eventually trained on


def test_run_training_resume_at_example_boundary_starts_next_example_fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_ids = [_ids(13, seed=1), _ids(9, seed=2)]
    train_masks = [None, None]
    args = _make_args(ckpt_every_tokens=1000)  # never triggers mid-run -- only the final forced save

    train.run_training(
        FakeHooks, model, optimizer, list(model.parameters()), train_ids, train_masks, [], [], "cpu", args,
        start_epoch=0, start_example=0, start_step=0, start_chunk_pos=0, start_total_tokens=0.0, start_last_ckpt_tokens=0.0,
    )

    ckpt = train.latest_checkpoint()
    state = torch.load(ckpt / "state.pt", weights_only=True)
    assert state["chunk_pos"] == 0
    assert state["example_idx"] == 1
