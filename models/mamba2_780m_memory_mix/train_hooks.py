"""Training hooks for mamba2_780m_memory_mix -- same training contract as
mamba2_2_7b_memory (stateful chunked forward, loss on every real token; see
that module's docstring). Local pieces: setup_training (binds this package's
load_base) and the logging hooks (the mix arm's log fields differ from the
gated-delta design's -- no retain/active-layers/ssm_norm, plus mix_w_norm).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sft"))
from lora import apply_lora

from ..mamba2_2_7b_memory.train_hooks import (  # noqa: F401  (re-exported hooks)
    EOS_ID,
    chunk_loss,
    init_state,
    on_step,
    reset_slot,
    sleep_slot,
)
from ..mamba2_2_7b_memory.train_hooks import extra_log as _extra_log_2_7b
from . import model as _model_mod

# See mamba2_780m_memory_state/train_hooks.py -- same sizing reasoning.
DEFAULT_CHUNK_LEN = 16
DEFAULT_MEMORY_WINDOW = 1


def setup_training(device, lora_rank: int, lora_alpha: float, lora_dropout: float):
    base = _model_mod.load_base(str(device))
    base = apply_lora(base, _model_mod.TARGET_LORA_MODULES, lora_rank, lora_alpha, lora_dropout)
    model = _model_mod.Model(base).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable_params):,}")
    return model, trainable_params


def extra_log(model) -> str | None:
    line = _extra_log_2_7b(model)
    if line is None:
        return None
    # The one signal BX1's "gate stuck at 0" decision rule needs: o_proj is
    # zero-init, so its weight norm growing is the mix pathway waking up.
    return line + f"  mix_w_norm {model.mix.o_proj.weight.detach().norm().item():.4g}"


def chunk_extra_log(model) -> list[str] | None:
    """Per-slot live lines for the mix arm's log fields (see
    Model._forward_manual/_forward_fused's mix log blocks)."""
    logs = model.last_token_log()
    if logs is None:
        return None
    return [
        f"beta {log['beta']:.4f}  surprise {log['surprise']:.4f}"
        f"  o_t_norm {log['o_t_norm']:.4f}  grad_norm {log['grad_norm']:.4g}"
        f"  resid_norm {log['resid_norm']:.4g}"
        for log in logs
    ]
