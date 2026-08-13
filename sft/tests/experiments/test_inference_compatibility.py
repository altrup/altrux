import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[2]))

import consolidation_null
from experiments import inference


def test_consolidation_null_reexports_inference_primitives():
    names = ("generate", "kl_loss", "replay_step", "run_chunks", "target_logprob")

    assert all(getattr(consolidation_null, name) is getattr(inference, name) for name in names)


def test_replay_schedule_stays_model_free():
    assert [inference.replay_step(step, 3, False) for step in range(4)] == [
        (0, True), (1, False), (2, False), (0, True),
    ]


def test_kl_loss_is_zero_for_identical_logits():
    logits = torch.randn(2, 3, 5)

    assert inference.kl_loss(logits, logits, 1.0).item() == 0.0
