"""Experiment: shared

Plain Mamba2-2.7B — the substrate registered by DISCUSSION-20260808
§2.10.14. NOT the memory-augmented `mamba2_2_7b_memory`.

The manual-mixer wrapper is mamba2_780m's, shared by import rather than
copied: the wrapper is the code that keeps needing hardware fixes, and a
copy would take them silently out of sync. Only the backbone id and the
loaders (which close over this module's globals) live here.

Scale notes vs the 780m arithmetic in the shared module's comments: at this
backbone's shape (64 layers, 80 heads of headdim 64, d_state 128, d_ssm
5120) a gradient-checkpoint boundary state is ~43.3M values, ~87 MB per
batch slot in bf16 — 2.2× the 780m figure. GRAD_CHECKPOINT_BLOCK stays 64
(the memory arms' default; see models/mamba2_2_7b_memory/model.py for the
arithmetic).
"""

import sys
from pathlib import Path

import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from ..mamba2_780m.model import (  # noqa: F401 -- re-exported interface
    ASST_OPEN,
    EOC,
    GRAD_CHECKPOINT_BLOCK,
    SPECIAL_TOKENS,
    TARGET_LORA_MODULES,
    TOKENIZER_ID,
    USER_OPEN,
    MixerState,
    Model,
)

MODEL_ID = "state-spaces/mamba2-2.7b"


def load_base(device: str) -> MambaLMHeadModel:
    """Load the raw HuggingFace model in bf16 (the shared module's load_base
    records why bf16). Used by sft/training/loop.py."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.common import build_tokenizer, extend_embeddings

    model = MambaLMHeadModel.from_pretrained(MODEL_ID, device=device, dtype=torch.bfloat16)
    tokenizer = build_tokenizer(sys.modules[__name__])
    extend_embeddings(model, len(tokenizer), tokenizer)
    return model


def load_inference(device: str) -> Model:
    """Load and wrap the model for inference. Used by the backend registry."""
    return Model(load_base(device)).to(device)
