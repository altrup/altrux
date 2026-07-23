"""Mamba2-780M backbone + the same Titans memory subsystem as
mamba2_2_7b_memory -- the stage-2 SCREENING platform (see
notes/DISCUSSION-20260723-780m-integration-screen.md): mechanism-level A/Bs
run here fast, winners get confirmed at 2.7B. Never compare magnitudes
across scales.

The whole implementation lives in models/mamba2_2_7b_memory/model.py, whose
Model derives its memory geometry from the backbone; this module only picks
the 780M layer indices and the integration mode. The MEMORY_INTEGRATION env
var (read at model construction) selects the A/B arm:

  state (default) -- the 2.7B design, layer indices scaled to 48 layers:
      gated-delta merge of the read into ssm_state at INJECTED_LAYERS,
      front-end at READ_LAYER (2/3 depth).
  mix -- token-mix integration (_TokenMixInjection): the gated read is
      ADDED to the residual stream entering MIX_READ_LAYER (1/3 depth),
      same token, no ssm_state injections anywhere.

See README.md for the layer/bottleneck sizing rationale and the known
A/B confound (the read point necessarily moves with the arm).
"""

import os

import torch

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from ..mamba2_2_7b_memory.model import (  # noqa: F401  (re-exported for probe/tooling parity)
    MemoryState,
    _NeuralMemory,
    _TokenMixInjection,
    fused_kernel_usable,
)
from ..mamba2_2_7b_memory.model import Model as _MemoryModel

MODEL_ID = "state-spaces/mamba2-780m"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
TARGET_LORA_MODULES: list[str] = ["in_proj", "out_proj"]

USER_OPEN = "[USER]"
ASST_OPEN = "[ASSISTANT]"
SPECIAL_TOKENS = [USER_OPEN, ASST_OPEN]

# mamba2-780m backbone: 48 layers, d_model 1536.
N_LAYER = 48
# Scaled from the 2.7B constants (READ_LAYER 42/64, INJECTED_LAYERS
# 22..62 step 2) by the 48/64 layer ratio -- same fractional depths.
READ_LAYER = 32
INJECTED_LAYERS: tuple[int, ...] = tuple(range(16, N_LAYER, 2))
# The mix arm reads (and lands) at the layer-21/22-boundary analog: after
# block 15, i.e. 1/3 into the stack -- see README.md.
MIX_READ_LAYER = 16


def integration_mode() -> str:
    mode = os.getenv("MEMORY_INTEGRATION", "state")
    if mode not in ("state", "mix"):
        raise ValueError(f"MEMORY_INTEGRATION must be 'state' or 'mix', got {mode!r}")
    return mode


class Model(_MemoryModel):
    """The shared memory Model bound to this backbone's layer indices, with
    the integration arm chosen by MEMORY_INTEGRATION at construction."""

    def __init__(self, mamba_model: MambaLMHeadModel):
        mode = integration_mode()
        super().__init__(
            mamba_model,
            read_layer=MIX_READ_LAYER if mode == "mix" else READ_LAYER,
            injected_layers=INJECTED_LAYERS,
            integration=mode,
        )
        print(f"mamba2_780m_memory integration: {mode}")


def load_base(device: str) -> MambaLMHeadModel:
    """Load the raw HuggingFace model (bf16, ~1.6 GB); LoRA is attached by
    the caller, same contract as mamba2_2_7b_memory.load_base."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.common import build_tokenizer, extend_embeddings

    model = MambaLMHeadModel.from_pretrained(MODEL_ID, device=device, dtype=torch.bfloat16)
    tokenizer = build_tokenizer(sys.modules[__name__])
    extend_embeddings(model, len(tokenizer))
    return model


def load_inference(device: str) -> Model:
    return Model(load_base(device)).to(device)
