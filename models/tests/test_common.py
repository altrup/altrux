import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.common import MarkerDelta, build_tokenizer, extend_embeddings


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


@pytest.fixture(scope="module")
def gpt_neox_tokenizer():
    return build_tokenizer(
        SimpleNamespace(TOKENIZER_ID="EleutherAI/gpt-neox-20b", SPECIAL_TOKENS=["[USER]", "[ASSISTANT]"])
    )


def _bpe_mean(model, tokenizer, marker: str) -> torch.Tensor:
    spelled = tokenizer(marker, add_special_tokens=False, split_special_tokens=True)["input_ids"]
    assert len(spelled) > 1
    return model.backbone.embedding.weight[spelled].mean(dim=0)


def test_extend_embeddings_inits_marker_rows_from_bpe_mean_when_growing(gpt_neox_tokenizer):
    tok = gpt_neox_tokenizer
    model = FakeModel(vocab_size=len(tok) - 2, d_model=8, tie_embeddings=True)

    extend_embeddings(model, len(tok), tok)

    for marker in ("[USER]", "[ASSISTANT]"):
        row = model.backbone.embedding.weight[tok.convert_tokens_to_ids(marker)]
        assert torch.allclose(row, _bpe_mean(model, tok, marker))
    assert model.marker_token_ids == [tok.convert_tokens_to_ids(m) for m in ("[USER]", "[ASSISTANT]")]


def test_extend_embeddings_inits_marker_rows_inside_pretrained_padding(gpt_neox_tokenizer):
    # Mamba checkpoints pad the vocab (50277 -> 50288), so the marker ids
    # already have (junk) rows and no growth happens -- they must still get
    # the BPE-mean init.
    tok = gpt_neox_tokenizer
    model = FakeModel(vocab_size=50288, d_model=8, tie_embeddings=True)
    embedding_before = model.backbone.embedding

    extend_embeddings(model, len(tok), tok)

    assert model.backbone.embedding is embedding_before  # no growth
    for marker in ("[USER]", "[ASSISTANT]"):
        row = model.backbone.embedding.weight[tok.convert_tokens_to_ids(marker)]
        assert torch.allclose(row, _bpe_mean(model, tok, marker))


def test_marker_delta_zero_init_is_an_exact_noop():
    delta = MarkerDelta([5, 7], d_model=4)
    emb = nn.Embedding(10, 4)
    head = nn.Linear(4, 10, bias=False)
    head.weight = emb.weight
    ids = torch.tensor([[1, 5, 7, 2]])

    h = delta.embed(emb(ids), ids)
    assert torch.equal(h, emb(ids))
    logits = delta.head(head(h), h)
    assert torch.equal(logits, head(h))


def test_marker_delta_touches_only_marker_rows_and_columns():
    delta = MarkerDelta([5, 7], d_model=4)
    with torch.no_grad():
        delta.delta[0] += 1.0  # marker id 5 only
    emb = nn.Embedding(10, 4)
    head = nn.Linear(4, 10, bias=False)
    head.weight = emb.weight
    ids = torch.tensor([[1, 5, 7, 2]])

    base_h = emb(ids)
    h = delta.embed(base_h, ids)
    changed_positions = (h != base_h).any(dim=-1)
    assert changed_positions.tolist() == [[False, True, False, False]]
    assert torch.allclose(h[0, 1], base_h[0, 1] + 1.0)

    base_logits = head(base_h)
    logits = delta.head(base_logits, base_h)
    changed_columns = (logits != base_logits).reshape(-1, 10).any(dim=0)
    assert changed_columns.nonzero().flatten().tolist() == [5]
    # The marker column shifts by h . delta -- the tied-row update.
    assert torch.allclose(logits[..., 5], base_logits[..., 5] + base_h @ delta.delta[0])


def test_marker_delta_gets_gradient_while_table_stays_frozen():
    delta = MarkerDelta([5, 7], d_model=4)
    emb = nn.Embedding(10, 4)
    head = nn.Linear(4, 10, bias=False)
    head.weight = emb.weight
    emb.weight.requires_grad_(False)
    ids = torch.tensor([[1, 5, 7, 2]])

    h = delta.embed(emb(ids), ids)
    logits = delta.head(head(h), h)
    logits.sum().backward()

    assert delta.delta.grad is not None
    assert delta.delta.grad.abs().sum() > 0
    assert emb.weight.grad is None


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
