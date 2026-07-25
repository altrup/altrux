"""CPU tests for train.py's multi-dataset support: spec parsing, per-dataset
training config, share-proportional mixing, curriculum-order preservation,
and the ramped recall weight. Same fake model/hooks as test_train.py -- none
of this is model-specific.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import train
from tests.test_train import FakeHooks, FakeStatefulModel, _ids, _make_args


# ---------------------------------------------------------------------------
# recall-weight ramp (pure function)
# ---------------------------------------------------------------------------

def test_recall_weight_ramp_starts_at_start_and_ends_at_end():
    assert train.recall_weight_at(0, 1.0, 16.0, 32) == pytest.approx(1.0)
    assert train.recall_weight_at(32, 1.0, 16.0, 32) == pytest.approx(16.0)
    assert train.recall_weight_at(999, 1.0, 16.0, 32) == pytest.approx(16.0)


def test_recall_weight_ramp_is_linear_in_between():
    assert train.recall_weight_at(16, 1.0, 16.0, 32) == pytest.approx(8.5)


def test_recall_weight_ramp_disabled_holds_the_end_weight():
    assert train.recall_weight_at(0, 1.0, 16.0, 0) == pytest.approx(16.0)


def test_recall_weight_geometric_ramp_is_log_linear():
    mid = train.recall_weight_at(16, 1.0, 16.0, 32, shape="geometric")
    assert mid == pytest.approx(4.0)


def test_recall_weight_ramp_applies_to_marked_tokens_as_the_step_advances(monkeypatch, tmp_path):
    """The boost on recall-marked tokens must grow with the optimizer step,
    not sit at the final multiplier from chunk one."""
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    torch.manual_seed(0)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    recall = torch.ones(13, dtype=torch.bool)

    recorded: list[torch.Tensor] = []

    class Hooks(FakeHooks):
        @staticmethod
        def chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight):
            recorded.append(mask_slice.clone())
            return FakeHooks.chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight)

    # accum_tokens == chunk_len -> one optimizer step per chunk, so chunk k
    # trains under global_step k.
    args = _make_args(recall_weight=16.0, recall_ramp_start=1.0, recall_ramp_steps=2)
    train.run_training(
        Hooks, model, optimizer, list(model.parameters()), [_ids(13, seed=3)], [None], [recall], [None],
        "cpu", args, start_epoch=0, start_slot_states=None, start_next_ptr=0, start_step=0,
        start_total_tokens=0.0, start_last_ckpt_tokens=0.0,
    )

    first_of_chunk = [w[0, 0].item() for w in recorded]
    assert first_of_chunk[0] == pytest.approx(1.0)
    assert first_of_chunk[1] == pytest.approx(8.5)
    assert first_of_chunk[2] == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# --data spec parsing
# ---------------------------------------------------------------------------

def test_bare_path_takes_the_global_defaults():
    spec = train.parse_data_spec("data/train.pt", default_chunk_len=48, default_batch_size=24)
    assert spec.path == "data/train.pt"
    assert (spec.chunk_len, spec.batch_size) == (48, 24)
    assert spec.grad_checkpoint is False
    assert spec.shuffle is True
    assert spec.share is None


def test_spec_overrides_per_dataset_training_config():
    spec = train.parse_data_spec(
        "data/train_cram.pt,share=35,chunk-len=512,batch-size=6,grad-checkpoint=1,shuffle=0",
        default_chunk_len=48, default_batch_size=24,
    )
    assert spec.path == "data/train_cram.pt"
    assert spec.share == pytest.approx(35.0)
    assert (spec.chunk_len, spec.batch_size) == (512, 6)
    assert spec.grad_checkpoint is True
    assert spec.shuffle is False


def test_spec_rejects_unknown_keys():
    with pytest.raises(ValueError):
        train.parse_data_spec("a.pt,chunklen=512", default_chunk_len=48, default_batch_size=24)


def test_spec_rejects_a_non_numeric_value():
    with pytest.raises(ValueError):
        train.parse_data_spec("a.pt,chunk-len=big", default_chunk_len=48, default_batch_size=24)


def test_shares_must_be_given_for_all_datasets_or_none():
    specs = [
        train.parse_data_spec("a.pt,share=1", default_chunk_len=4, default_batch_size=1),
        train.parse_data_spec("b.pt", default_chunk_len=4, default_batch_size=1),
    ]
    with pytest.raises(ValueError):
        train.check_shares(specs)


# ---------------------------------------------------------------------------
# loading several .pt slices into one flat example list
# ---------------------------------------------------------------------------

def _write_dataset(path, lengths, *, recall=False, seed=0):
    ids = [_ids(n, seed=seed + i) for i, n in enumerate(lengths)]
    data = {"ids": ids, "masks": [torch.ones(n, dtype=torch.bool) for n in lengths]}
    if recall:
        data["recall_masks"] = [torch.zeros(n, dtype=torch.bool) for n in lengths]
    torch.save(data, path)
    return path


def test_load_datasets_concatenates_slices_and_records_index_ranges(tmp_path):
    a = _write_dataset(tmp_path / "a.pt", [10, 12])
    b = _write_dataset(tmp_path / "b.pt", [8], recall=True, seed=50)
    specs = [
        train.parse_data_spec(str(a), default_chunk_len=4, default_batch_size=1),
        train.parse_data_spec(str(b), default_chunk_len=4, default_batch_size=1),
    ]

    ids, masks, recall, sleeps = train.load_datasets(specs)

    assert [t.numel() for t in ids] == [10, 12, 8]
    assert (specs[0].lo, specs[0].hi) == (0, 2)
    assert (specs[1].lo, specs[1].hi) == (2, 3)
    # a slice without recall_masks still contributes correctly-sized Nones
    assert recall[0] is None and recall[1] is None
    assert recall[2] is not None
    assert len(masks) == len(sleeps) == 3


# ---------------------------------------------------------------------------
# grouping by training config
# ---------------------------------------------------------------------------

def test_datasets_sharing_a_config_share_a_group():
    specs = [
        train.DataSpec(path="chains.pt", chunk_len=48, batch_size=24),
        train.DataSpec(path="ballast.pt", chunk_len=48, batch_size=24),
        train.DataSpec(path="cram.pt", chunk_len=512, batch_size=6, grad_checkpoint=True),
    ]
    groups = train.group_specs(specs)
    assert [[s.path for s in g] for g in groups] == [["chains.pt", "ballast.pt"], ["cram.pt"]]


# ---------------------------------------------------------------------------
# example ordering: curriculum preservation + share-proportional interleaving
# ---------------------------------------------------------------------------

def _keep_all(_idx):
    return True


def test_shuffle_off_preserves_the_artifacts_own_order():
    ids = [_ids(4, seed=i) for i in range(8)]
    spec = train.DataSpec(path="cram.pt", chunk_len=4, batch_size=1, shuffle=False, lo=0, hi=8)
    assert train.build_order([spec], ids, epoch=0, keep=_keep_all) == list(range(8))


def test_single_shuffled_dataset_keeps_the_historical_per_epoch_permutation():
    ids = [_ids(4, seed=i) for i in range(8)]
    spec = train.DataSpec(path="chains.pt", chunk_len=4, batch_size=1, lo=0, hi=8)
    expected = torch.randperm(8, generator=torch.Generator().manual_seed(3)).tolist()
    assert train.build_order([spec], ids, epoch=3, keep=_keep_all) == expected


def test_members_of_a_group_interleave_by_token_share():
    # 12 examples of 10 tokens each: first 8 are dataset A, last 4 dataset B.
    ids = [_ids(10, seed=i) for i in range(12)]
    a = train.DataSpec(path="a.pt", chunk_len=4, batch_size=1, share=3.0, shuffle=False, lo=0, hi=8)
    b = train.DataSpec(path="b.pt", chunk_len=4, batch_size=1, share=1.0, shuffle=False, lo=8, hi=12)

    order = train.build_order([a, b], ids, epoch=0, keep=_keep_all)

    assert sorted(order) == list(range(12))
    # every prefix stays close to the 3:1 target -- 3 A's for each B
    from_b = sum(1 for idx in order[:8] if idx >= 8)
    assert from_b == 2
    # each member's own order is untouched
    assert [i for i in order if i < 8] == list(range(8))
    assert [i for i in order if i >= 8] == list(range(8, 12))


# ---------------------------------------------------------------------------
# end-to-end: two groups with different chunk-len/batch-size
# ---------------------------------------------------------------------------

def test_run_training_uses_each_groups_own_chunk_len_and_batch_size(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    torch.manual_seed(0)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    shapes: list[tuple[int, int]] = []

    class Hooks(FakeHooks):
        @staticmethod
        def chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight):
            shapes.append(tuple(input_ids.shape))
            return FakeHooks.chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight)

    train_ids = [_ids(17, seed=i) for i in range(4)]
    specs = [
        train.DataSpec(path="a.pt", chunk_len=2, batch_size=1, share=1.0, shuffle=False, lo=0, hi=2),
        train.DataSpec(path="b.pt", chunk_len=8, batch_size=3, share=1.0, shuffle=False, lo=2, hi=4),
    ]
    args = _make_args(chunk_len=None, batch_size=1, mix_segment_tokens=16)

    train.run_training(
        Hooks, model, optimizer, list(model.parameters()), train_ids, [None] * 4, [None] * 4, [None] * 4,
        "cpu", args, start_epoch=0, start_slot_states=None, start_next_ptr=0, start_step=0,
        start_total_tokens=0.0, start_last_ckpt_tokens=0.0, specs=specs,
    )

    assert (1, 2) in shapes, "group A must run at its own chunk_len=2 / batch=1"
    assert (3, 8) in shapes, "group B must run at its own chunk_len=8 / batch=3"
    # both groups get trained, and the run alternates rather than draining one first
    assert set(shapes) == {(1, 2), (3, 8)}
    switches = sum(1 for x, y in zip(shapes, shapes[1:]) if x != y)
    assert switches >= 2, "segments should interleave the two groups"

    state = torch.load(train.latest_checkpoint() / "state.pt", weights_only=True)
    assert state["group_ptrs"] == [2, 2], "both groups' example orders fully consumed"


def test_run_training_trains_every_token_of_slices_that_share_a_config(monkeypatch, tmp_path):
    """Slices with the same training config merge into one group and are
    consumed inside a single batch stream -- no segment handover at all."""
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path)
    torch.manual_seed(0)
    model = FakeStatefulModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_ids = [_ids(9, seed=i) for i in range(4)]
    specs = [
        train.DataSpec(path="a.pt", chunk_len=4, batch_size=1, share=1.0, shuffle=False, lo=0, hi=2),
        train.DataSpec(path="b.pt", chunk_len=4, batch_size=1, share=1.0, shuffle=False, lo=2, hi=4),
    ]
    args = _make_args(chunk_len=None, batch_size=1, mix_segment_tokens=8)

    train.run_training(
        FakeHooks, model, optimizer, list(model.parameters()), train_ids, [None] * 4, [None] * 4, [None] * 4,
        "cpu", args, start_epoch=0, start_slot_states=None, start_next_ptr=0, start_step=0,
        start_total_tokens=0.0, start_last_ckpt_tokens=0.0, specs=specs,
    )

    state = torch.load(train.latest_checkpoint() / "state.pt", weights_only=True)
    assert state["total_tokens"] == 4 * 8  # 4 examples x 8 target tokens each
    assert "group_ptrs" not in state, "a single config group needs no per-group resume state"
