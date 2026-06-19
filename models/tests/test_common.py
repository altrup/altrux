import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.common import build_tokenizer, extend_embeddings


class FakeConfig:
    def __init__(self, tie_embeddings: bool):
        self.tie_embeddings = tie_embeddings


class FakeBackbone(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding


class FakeModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, tie_embeddings: bool):
        super().__init__()
        self.config = FakeConfig(tie_embeddings)
        embedding = nn.Embedding(vocab_size, d_model)
        self.backbone = FakeBackbone(embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = embedding.weight


def test_extend_embeddings_tied_grows_and_keeps_tie():
    model = FakeModel(vocab_size=10, d_model=4, tie_embeddings=True)
    old_weight = model.backbone.embedding.weight.detach().clone()

    extend_embeddings(model, new_vocab_size=12)

    assert model.backbone.embedding.weight.shape == (12, 4)
    assert model.lm_head.weight.shape == (12, 4)
    assert torch.allclose(model.backbone.embedding.weight[:10], old_weight)
    assert model.lm_head.weight is model.backbone.embedding.weight


def test_extend_embeddings_untied_grows_independently():
    model = FakeModel(vocab_size=10, d_model=4, tie_embeddings=False)
    old_embed = model.backbone.embedding.weight.detach().clone()
    old_lm_head = model.lm_head.weight.detach().clone()

    extend_embeddings(model, new_vocab_size=12)

    assert model.backbone.embedding.weight.shape == (12, 4)
    assert model.lm_head.weight.shape == (12, 4)
    assert torch.allclose(model.backbone.embedding.weight[:10], old_embed)
    assert torch.allclose(model.lm_head.weight[:10], old_lm_head)
    assert model.lm_head.weight is not model.backbone.embedding.weight


def test_extend_embeddings_noop_when_already_large_enough():
    model = FakeModel(vocab_size=10, d_model=4, tie_embeddings=True)
    embedding_before = model.backbone.embedding

    extend_embeddings(model, new_vocab_size=10)

    assert model.backbone.embedding is embedding_before


def test_build_tokenizer_registers_special_tokens_as_atomic():
    model_mod = SimpleNamespace(
        TOKENIZER_ID="EleutherAI/gpt-neox-20b",
        SPECIAL_TOKENS=["[USER]", "[ASSISTANT]"],
    )

    tokenizer = build_tokenizer(model_mod)

    assert tokenizer.eos_token_id is not None
    user_id = tokenizer.convert_tokens_to_ids("[USER]")
    assert user_id != tokenizer.unk_token_id

    ids = tokenizer.encode("[USER] hello\n", add_special_tokens=False)
    assert ids[0] == user_id
    tail_ids = tokenizer.encode(" hello\n", add_special_tokens=False)
    assert ids[1:] == tail_ids
