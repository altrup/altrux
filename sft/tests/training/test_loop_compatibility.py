import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2]))

import train
from training import loop


def test_train_reexports_training_loop_primitives():
    for name in ("_Slot", "_collect_tensors", "_slot_state_finite", "_print_live", "_clear_live", "_show_batch_progress"):
        assert getattr(train, name) is getattr(loop, name)


def test_train_keeps_a_run_training_compatibility_wrapper():
    assert train.run_training is not loop.run_training


def test_run_segment_is_a_top_level_loop_implementation():
    assert callable(loop.run_segment)
