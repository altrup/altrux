"""Mamba2-780M + the Titans memory subsystem, TOKEN-MIX arm -- the gated
read is ADDED to the residual stream entering layer 16 (the layer-21/22-
boundary analog, 1/3 depth), same token, no ssm_state injections anywhere
(_TokenMixInjection in the shared implementation).

One folder per integration arm (this vs models/mamba2_780m_memory_state), so
MODEL_NAME selects the arm and each arm's checkpoints live under its own
checkpoints/ -- a checkpoint's folder tells you its architecture. The
read/mix boundary is FIXED at 16, no override: this arm's thesis is early
entry (the whole upper stack computes over the read), a 2/3 variant would
never ship, and a fixed boundary means a checkpoint's geometry can never
contradict its folder name. This is BX1 of the screening A/B (see
notes/discussion/DISCUSSION-20260723-780m-integration-screen.md).
"""

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

# mamba2-780m backbone: 48 layers, d_model 1536. The read/mix boundary sits
# after block 15 -- 1/3 into the stack, the scaled layer-21/22 analog.
N_LAYER = 48
READ_LAYER = 16


class Model(_MemoryModel):
    def __init__(self, mamba_model: MambaLMHeadModel):
        super().__init__(
            mamba_model,
            read_layer=READ_LAYER,
            integration="mix",
        )


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
