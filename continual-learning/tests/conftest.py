"""Shared fixtures — tiny mock model that mirrors the Granite interface exactly."""
import torch
import torch.nn as nn
import pytest
from transformers import BatchEncoding
from continual_learning.model import ContinualLearningModel
from continual_learning.trainer import Trainer, TrainingConfig

HIDDEN = 32
VOCAB = 64
NUM_LAYERS = 6  # split at 4, critic depth 4


class _TinyLayer(nn.Module):
    """Minimal hybrid-style layer: accepts hidden_states, returns a tuple."""
    def __init__(self, config=None, layer_idx=None):
        super().__init__()
        hidden = getattr(config, "hidden_size", HIDDEN) if config is not None else HIDDEN
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, hidden_states, **_kwargs):
        return (self.proj(hidden_states),)


class _TinyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([_TinyLayer() for _ in range(NUM_LAYERS)])
        self.norm = nn.LayerNorm(HIDDEN)


class _TinyOutput:
    def __init__(self, logits):
        self.logits = logits
        self.loss = None


class _TinyBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TinyInner()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        self.config = type("Cfg", (), {"hidden_size": HIDDEN})()

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)[0]
        h = self.model.norm(h)
        return _TinyOutput(self.lm_head(h))

    @property
    def device(self):
        return next(self.parameters()).device

    def generate(self, input_ids, max_new_tokens=4, **_kwargs):
        pad = torch.zeros(input_ids.shape[0], max_new_tokens, dtype=torch.long)
        return torch.cat([input_ids, pad], dim=1)


class _TinyTokenizer:
    eos_token_id = 1

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
        ids = torch.randint(2, VOCAB, (1, 8))
        return BatchEncoding({"input_ids": ids})

    def decode(self, token_ids, skip_special_tokens=True):
        return "mock response"


@pytest.fixture()
def tiny_model(monkeypatch):
    """ContinualLearningModel backed by a tiny in-process mock — no download needed."""
    base = _TinyBaseModel()
    tok = _TinyTokenizer()

    monkeypatch.setattr(
        "continual_learning.model.AutoModelForCausalLM.from_pretrained",
        lambda *a, **kw: base,
    )
    monkeypatch.setattr(
        "continual_learning.model.AutoTokenizer.from_pretrained",
        lambda *a, **kw: tok,
    )

    model = ContinualLearningModel(model_name="mock")
    model.eval()
    return model


@pytest.fixture()
def trainer(tiny_model):
    return Trainer(tiny_model, TrainingConfig())
