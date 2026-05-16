"""Device selection tests.

Three layers of verification:
  1. _build_max_memory: excluded GPUs get 0 bytes, included ones get a positive budget.
  2. ContinualLearningModel.__init__ passes that exact max_memory to from_pretrained.
  3. On a real CUDA system: base model parameters land on GPU, not CPU.
"""
import os
from contextlib import contextmanager
from unittest.mock import patch, MagicMock, call

import torch
import pytest

from cl_backend.model import _build_max_memory, COMPUTE_RESERVE_FRAC
from cl_backend.config import ServerConfig
from cl_backend.main import _load_model


# ── helpers ──────────────────────────────────────────────────────────────────

@contextmanager
def fake_cuda(n_gpus: int, vram_per_gpu: int = 16 * 1024**3):
    """Patch torch.cuda so it looks like n_gpus are available with vram_per_gpu each."""
    prop = MagicMock()
    prop.total_memory = vram_per_gpu
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=n_gpus), \
         patch("torch.cuda.get_device_properties", return_value=prop):
        yield prop


# ── _build_max_memory unit tests ─────────────────────────────────────────────

def test_build_max_memory_restricts_excluded_gpu():
    with fake_cuda(2):
        mem = _build_max_memory(device_ids=[0])
    assert mem[0] > 0, "included GPU[0] must have a positive memory budget"
    assert mem[1] == 0, "excluded GPU[1] must have a 0-byte budget"


def test_build_max_memory_budget_is_correct_fraction():
    vram = 16 * 1024**3
    with fake_cuda(2, vram_per_gpu=vram):
        mem = _build_max_memory(device_ids=[0])
    expected = int(vram * (1 - COMPUTE_RESERVE_FRAC))
    assert mem[0] == expected


def test_build_max_memory_multi_device():
    with fake_cuda(3):
        mem = _build_max_memory(device_ids=[0, 2])
    assert mem[0] > 0
    assert mem[1] == 0, "GPU[1] not in device_ids — must be excluded"
    assert mem[2] > 0


def test_build_max_memory_none_includes_all():
    with fake_cuda(2):
        mem = _build_max_memory(device_ids=None)
    assert mem[0] > 0
    assert mem[1] > 0


def test_build_max_memory_cpu_only():
    with patch("torch.cuda.is_available", return_value=False):
        assert _build_max_memory(device_ids=[0]) is None


# ── from_pretrained receives correct max_memory ───────────────────────────────

def test_from_pretrained_receives_restricted_max_memory():
    """ContinualLearningModel must forward the max_memory dict from _build_max_memory
    to from_pretrained unchanged — 0 for excluded GPUs, positive for included ones."""
    import cl_backend.model as _mod

    captured_kwargs: dict = {}

    fake_base = MagicMock()
    fake_base.model.layers = torch.nn.ModuleList()
    fake_base.config.hidden_size = 32
    fake_base.parameters.return_value = iter([torch.zeros(1)])
    fake_base.hf_device_map = {}

    def fake_from_pretrained(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return fake_base

    vram = 16 * 1024**3
    with fake_cuda(2, vram_per_gpu=vram), \
         patch("cl_backend.model.AutoModelForCausalLM.from_pretrained", side_effect=fake_from_pretrained), \
         patch("cl_backend.model.AutoTokenizer.from_pretrained", return_value=MagicMock()):
        try:
            _mod.ContinualLearningModel(model_name="mock", device_ids=[0])
        except Exception:
            pass  # may fail during critic init; we only care about from_pretrained args

    assert "max_memory" in captured_kwargs, "max_memory not passed to from_pretrained"
    mem = captured_kwargs["max_memory"]
    assert mem is not None, "max_memory was None — GPU budget not computed"
    assert mem[0] == int(vram * (1 - COMPUTE_RESERVE_FRAC)), \
        f"GPU[0] budget wrong: {mem[0]}"
    assert mem[1] == 0, \
        f"GPU[1] should be excluded (0 bytes) but got: {mem[1]}"


# ── _load_model sets CUDA_VISIBLE_DEVICES before construction ─────────────────

def _fake_model_cls(captured: dict):
    class _F:
        def __init__(self, **kwargs):
            captured["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return _F


def test_load_model_sets_cuda_visible_devices(tmp_path):
    captured: dict = {}
    cfg = ServerConfig(devices="0", checkpoint_dir=str(tmp_path), no_resume=True)
    with patch("cl_backend.main.ContinualLearningModel", _fake_model_cls(captured)):
        _load_model(cfg)
    assert captured["CUDA_VISIBLE_DEVICES"] == "0"


def test_load_model_no_restriction_leaves_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    captured: dict = {}
    cfg = ServerConfig(devices=None, checkpoint_dir=str(tmp_path), no_resume=True)
    with patch("cl_backend.main.ContinualLearningModel", _fake_model_cls(captured)):
        _load_model(cfg)
    assert captured["CUDA_VISIBLE_DEVICES"] is None


# ── real GPU placement (requires actual GPU hardware + real model) ────────────

@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires ≥2 CUDA devices",
)
def test_no_params_on_excluded_gpu(tmp_path, monkeypatch):
    """When CL_DEVICES=0, no parameter should land on cuda:1."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    import cl_backend.model as _mod
    model = _mod.ContinualLearningModel(device_ids=[0])
    bad = [p for p in model.base_model.parameters() if p.device == torch.device("cuda:1")]
    assert not bad, f"{len(bad)} parameter(s) found on cuda:1 (excluded GPU)"
