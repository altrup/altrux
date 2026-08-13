import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).parents[2]))

import dream_fidelity
from diagnostics import dream_fidelity as implementation


def test_dream_fidelity_reexports_cpu_safe_probe_surface():
    for name in ("generate", "load_trainable", "overlap_with_prime", "main"):
        assert getattr(dream_fidelity, name) is getattr(implementation, name)


def test_overlap_excludes_common_token_types():
    generated = torch.tensor([1, 2, 3, 3])
    prime = torch.tensor([2, 3, 4])

    assert implementation.overlap_with_prime(generated, prime, {3}) == 1 / 3
