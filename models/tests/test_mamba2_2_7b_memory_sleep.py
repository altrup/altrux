"""Fast (seconds, no download, no GPU) tests for Model.sleep_slot -- the
"sleep" boundary of the three-tier memory design (see the model README):
the backbone's per-layer SSM/conv state is wiped to a fresh sequence's
zeros while the neural memory persists, so anything recalled across a
sleep provably flows through the episodic store.

Model itself is hardcoded to the real 2.7B backbone dims, so these call
Model.sleep_slot unbound against a hand-built small MemoryState -- the
method only touches the state, which is exactly the property being pinned:
it must not depend on (or disturb) anything else.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.mamba2_2_7b_memory.model import MemoryState, Model, _NeuralMemory

BATCH, MEM_DIM, MEM_HIDDEN, N_LAYERS = 3, 16, 32, 4


def _make_state() -> MemoryState:
    torch.manual_seed(0)
    conv_states = [torch.randn(BATCH, 8, 5) for _ in range(N_LAYERS)]
    ssm_states = [torch.randn(BATCH, 2, 4, 6) for _ in range(N_LAYERS)]
    nm = _NeuralMemory(BATCH, MEM_DIM, MEM_HIDDEN, torch.device("cpu"), torch.float32)
    for s in nm.momentum:
        s.normal_()
    return MemoryState(conv_states, ssm_states, nm, torch.randn(BATCH, MEM_DIM), torch.randn(BATCH))


def test_sleep_wipes_backbone_state_of_that_slot_only():
    state = _make_state()
    Model.sleep_slot(None, state, 1)
    for conv, ssm in zip(state.conv_states, state.ssm_states):
        assert (conv[1] == 0).all() and (ssm[1] == 0).all()
        for b in (0, 2):
            assert not (conv[b] == 0).all() and not (ssm[b] == 0).all()
    assert (state.last_o_t[1] == 0).all() and state.last_surprise[1] == 0
    assert not (state.last_o_t[0] == 0).all()


def test_sleep_preserves_neural_memory_exactly():
    state = _make_state()
    nm = state.neural_memory
    before = [t.clone() for t in (nm.w1, nm.b1, nm.w2, nm.b2, nm.w1_init, nm.w2_init, *nm.momentum)]
    Model.sleep_slot(None, state, 1)
    after = [nm.w1, nm.b1, nm.w2, nm.b2, nm.w1_init, nm.w2_init, *nm.momentum]
    for b, a in zip(before, after):
        assert torch.equal(b, a)
