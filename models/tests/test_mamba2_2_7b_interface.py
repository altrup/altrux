"""Interface test for the plain mamba2_2_7b substrate.

No weights are downloaded and no model is instantiated: this pins the module
contract every caller reads at import time (sft/train.py, sft/dream_sleep.py,
backend/app/model/registry.py) -- which is exactly the part that a
copy-adaptation from mamba2_780m can get wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import models.mamba2_2_7b as pkg
import models.mamba2_2_7b.model as M
import models.mamba2_2_7b.train_hooks as hooks
import models.mamba2_780m.model as M780


def test_points_at_the_27b_backbone_not_the_780m_one():
    assert M.MODEL_ID == "state-spaces/mamba2-2.7b"
    assert M.MODEL_ID != M780.MODEL_ID
    assert M.TOKENIZER_ID == M780.TOKENIZER_ID
    assert M.TARGET_LORA_MODULES == M780.TARGET_LORA_MODULES


def test_special_tokens_include_eoc_in_marker_order():
    assert M.SPECIAL_TOKENS == [M.USER_OPEN, M.ASST_OPEN, M.EOC]
    assert all(t == t.strip() for t in M.SPECIAL_TOKENS), "markers are bare -- callers add the separator"
    assert M.SPECIAL_TOKENS == M780.SPECIAL_TOKENS


def test_package_reexports_the_model_interface():
    for name in ("MODEL_ID", "TOKENIZER_ID", "TARGET_LORA_MODULES", "USER_OPEN", "ASST_OPEN", "EOC", "SPECIAL_TOKENS", "Model", "load_base", "load_inference"):
        assert getattr(pkg, name, None) is getattr(M, name), name


def test_lora_base_is_not_quantized():
    assert getattr(M, "QUANTIZE_LORA_BASE", False) is False


def test_train_hooks_export_the_generic_loops_interface():
    assert callable(hooks.setup_training)
    assert callable(hooks.chunk_loss)
    assert callable(hooks.set_grad_checkpoint)
    assert hooks.DEFAULT_CHUNK_LEN > 0
    assert hooks._model_mod is M, "hooks must drive this folder's model, not another folder's"


def test_is_not_the_memory_variant():
    """§2.10.14's substrate is plain -- no memory subsystem sharing the same
    backbone id."""
    assert not hasattr(M, "READ_LAYER") and not hasattr(M, "INJECTED_LAYERS")
    assert not hasattr(hooks, "DEFAULT_MEMORY_WINDOW")
