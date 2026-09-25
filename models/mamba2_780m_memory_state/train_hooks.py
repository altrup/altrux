"""Experiment: memory-model

Training hooks for mamba2_780m_memory_state -- same training contract as
mamba2_2_7b_memory (stateful chunked forward, loss on every real token; see
that module's docstring). Everything except setup_training (which binds this
package's load_base) and DEFAULT_CHUNK_LEN is reused from the 2.7B hooks:
this arm's log fields are identical to the 2.7B state-injection design's.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sft"))
from adapters.lora import apply_lora

from ..mamba2_2_7b_memory.train_hooks import (  # noqa: F401  (re-exported hooks)
    EOS_ID,
    chunk_extra_log,
    chunk_loss,
    extra_log,
    init_state,
    on_step,
    reset_slot,
    set_grad_checkpoint,
    sleep_slot,
)
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
