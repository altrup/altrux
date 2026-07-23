"""Training hooks for mamba2_780m_memory -- same training contract as
mamba2_2_7b_memory (stateful chunked forward, loss on every real token; see
that module's docstring), reusing its hook implementations wherever they
only act on the passed-in model. Only setup_training (binds this package's
load_base) and the logging hooks (integration-mode-aware fields) are local.
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

# Manual-path VRAM per token is ~2.8x smaller than the 2.7B variant
# (mem_hidden 6144 vs 10240, d_model 1536 vs 2560), so the local default can
# sit higher than its 7. Benchmark with `make smoke-test --chunk-len N`
# across 10+ consecutive chunks before raising further (same caveat as the
# 2.7B hooks: one isolated chunk's peak VRAM is not representative).
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
    if line is None or model.integration != "mix":
        return line
    # The one signal BX1's "gate stuck at 0" decision rule needs: o_proj is
    # zero-init, so its weight norm growing is the mix pathway waking up.
    return line + f"  mix_w_norm {model.mix.o_proj.weight.detach().norm().item():.4g}"


def chunk_extra_log(model) -> list[str] | None:
    """Per-slot live lines; tolerant of both integration modes' log fields
    (the mix arm has no per-layer active/cos-sim or ssm_norm entries)."""
    logs = model.last_token_log()
    if logs is None:
        return None
    lines = []
    for log in logs:
        parts = [f"beta {log['beta']:.4f}"]
        if "retain" in log:
            parts.append(f"retain {log['retain']:.4f}")
            parts.append(f"active {log['active_layers']:>2}/{log['n_layers']}  min_cos_sim {log['min_cos_sim']:.4f}")
        parts.append(f"surprise {log['surprise']:.4f}  o_t_norm {log['o_t_norm']:.4f}")
        parts.append(f"grad_norm {log['grad_norm']:.4g}")
        if "ssm_norm" in log:
            parts.append(f"ssm_norm {log['ssm_norm']:.4g}")
        parts.append(f"resid_norm {log['resid_norm']:.4g}")
        lines.append("  ".join(parts))
    return lines
