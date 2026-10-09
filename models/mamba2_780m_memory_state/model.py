"""Mamba2-780M + the Titans memory subsystem, STATE-INJECTION arm -- the
2.7B design with layer indices scaled to 48 layers: gated-delta merge of the
read into ssm_state at INJECTED_LAYERS, front-end at READ_LAYER (2/3 depth).

One folder per integration arm (this vs models/mamba2_780m_memory_mix), so
MODEL_NAME selects the arm and each arm's checkpoints live under its own
checkpoints/ -- a checkpoint's folder tells you its architecture. This is
the stage-2 SCREENING platform's BX0 baseline (see
notes/discussion/DISCUSSION-20260723-780m-integration-screen.md); never compare delta
magnitudes across scales. The whole implementation lives in
models/mamba2_2_7b_memory/model.py, whose Model derives its memory geometry
from the backbone; this module only binds the 780M layer indices.
"""

import os

import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from ..mamba2_2_7b_memory.model import (  # noqa: F401  (re-exported for probe/tooling parity)
    MemoryState,
    _NeuralMemory,
    fused_kernel_usable,
)
from ..mamba2_2_7b_memory.model import Model as _MemoryModel

MODEL_ID = "state-spaces/mamba2-780m"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
TARGET_LORA_MODULES: list[str] = ["in_proj", "out_proj"]

USER_OPEN = "[USER]"
ASST_OPEN = "[ASSISTANT]"
SPECIAL_TOKENS = [USER_OPEN, ASST_OPEN]

# mamba2-780m backbone: 48 layers, d_model 1536. Indices scaled from the
# 2.7B constants (READ_LAYER 42/64, INJECTED_LAYERS 22..62 step 2) by the
# 48/64 layer ratio -- same fractional depths.
N_LAYER = 48
READ_LAYER = 32
INJECTED_LAYERS: tuple[int, ...] = tuple(range(16, N_LAYER, 2))


def read_layer() -> int:
    """Arm default 32, overridable via MEMORY_READ_LAYER for the conditional
    state@16 cross cell (see README.md). The mix arm deliberately has no
    such knob -- its read point IS the mix point, fixed by that arm's
    thesis."""
    override = os.getenv("MEMORY_READ_LAYER")
    layer = int(override) if override else READ_LAYER
    if not 0 < layer < N_LAYER:
        raise ValueError(f"MEMORY_READ_LAYER must be in (0, {N_LAYER}), got {layer}")
    return layer


class Model(_MemoryModel):
    def __init__(self, mamba_model: MambaLMHeadModel):
        super().__init__(
            mamba_model,
            read_layer=read_layer(),
            injected_layers=INJECTED_LAYERS,
        )
        print(f"mamba2_780m_memory_state read_layer: {self.read_layer}")


def load_base(device: str) -> MambaLMHeadModel:
    """Load the raw HuggingFace model (bf16, ~1.6 GB); LoRA is attached by
    the caller, same contract as mamba2_2_7b_memory.load_base."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.common import build_tokenizer, extend_embeddings

    model = MambaLMHeadModel.from_pretrained(MODEL_ID, device=device, dtype=torch.bfloat16)
    tokenizer = build_tokenizer(sys.modules[__name__])
    extend_embeddings(model, len(tokenizer), tokenizer)
    return model


def load_inference(device: str) -> Model:
    return Model(load_base(device)).to(device)
