"""CPU tests for warm_start.py: the rendered corpus, its invariants, and the
train->save->load round trip on a fake backbone. The real ultrachat download
and the 400-step run are box work; nothing here touches either."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lora import DEFAULT_ALPHA, DEFAULT_DROPOUT, DEFAULT_RANK, apply_lora, load_adapter
from models.common import build_tokenizer
from warm_start import build_corpus, build_parser, corpus_invariants, sample_text, train

USER_OPEN, ASST_OPEN = "[USER]", "[ASSISTANT]"


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer(
        SimpleNamespace(TOKENIZER_ID="EleutherAI/gpt-neox-20b", SPECIAL_TOKENS=[USER_OPEN, ASST_OPEN])
    )


def _records(n: int) -> list[dict]:
    return [
        {"messages": [
            {"role": "user", "content": f"Question {i} about the weather?"},
            {"role": "assistant", "content": f"Answer {i}: it is sunny."},
            {"role": "user", "content": "And tomorrow?"},
            {"role": "assistant", "content": "Rain, most likely."},
        ]}
        for i in range(n)
    ]


def test_corpus_renders_markers_as_single_special_ids_and_stops_at_the_turn_budget(tokenizer):
    user_id = tokenizer.convert_tokens_to_ids(USER_OPEN)
    asst_id = tokenizer.convert_tokens_to_ids(ASST_OPEN)

    corpus, turns = build_corpus(_records(100), tokenizer, USER_OPEN, ASST_OPEN, max_len=4096, turns=10)

    assert turns >= 10
    assert len(corpus) == 3  # 4 turns each, stops at the first conversation past the budget
    for ids, mask in corpus:
        assert len(ids) == len(mask)
        assert ids[0] == user_id
        assert ids.count(asst_id) == 2
        # the marker's BPE spelling must never appear -- the embedding this
        # warm start trains only exists on the registered single id
        spelled = tokenizer(ASST_OPEN, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        assert len(spelled) > 1
        assert not any(ids[i:i + len(spelled)] == spelled for i in range(len(ids)))


def test_corpus_invariants_are_zero_on_a_well_formed_corpus(tokenizer):
    corpus, _ = build_corpus(_records(5), tokenizer, USER_OPEN, ASST_OPEN, max_len=4096, turns=8)

    assert corpus_invariants(corpus, tokenizer, USER_OPEN, ASST_OPEN) == {
        "user_turns_with_no_answer": 0,
        "trained_tokens_outside_assistant_turns": 0,
        "examples_with_no_trained_tokens": 0,
    }


def test_corpus_invariants_catch_an_unanswered_user_turn(tokenizer):
    corpus, _ = build_corpus(_records(1), tokenizer, USER_OPEN, ASST_OPEN, max_len=4096, turns=1)
    ids, mask = corpus[0]
    ids = ids + [tokenizer.convert_tokens_to_ids(USER_OPEN)]
    mask = mask + [False]

    counts = corpus_invariants([(ids, mask)], tokenizer, USER_OPEN, ASST_OPEN)

    assert counts["user_turns_with_no_answer"] == 1


def test_corpus_invariants_catch_loss_on_a_user_turn(tokenizer):
    corpus, _ = build_corpus(_records(1), tokenizer, USER_OPEN, ASST_OPEN, max_len=4096, turns=1)
    ids, mask = corpus[0]
    mask = [True] + mask[1:]

    counts = corpus_invariants([(ids, mask)], tokenizer, USER_OPEN, ASST_OPEN)

    assert counts["trained_tokens_outside_assistant_turns"] > 0


def test_sample_text_decodes_a_marker_join(tokenizer):
    corpus, _ = build_corpus(_records(1), tokenizer, USER_OPEN, ASST_OPEN, max_len=4096, turns=1)

    text = sample_text(corpus[0][0], tokenizer, ASST_OPEN, width=60)

    assert ASST_OPEN in text
    assert "it is sunny" in text


def test_lora_defaults_match_dream_sleep_s(tokenizer):
    """A checkpoint trained at a rank/alpha dream_sleep does not default to
    cannot be loaded by --init-adapter at all (load_adapter is fatal on a
    mismatch), so the two parsers are pinned to the same source."""
    import dream_sleep

    warm = build_parser().parse_args([])
    dream = dream_sleep.build_parser().parse_args([])

    assert (warm.lora_rank, warm.lora_alpha) == (dream.lora_rank, dream.lora_alpha)
    assert (warm.lora_rank, warm.lora_alpha) == (DEFAULT_RANK, DEFAULT_ALPHA)
    assert warm.steps == 400 and warm.lr == 1e-4 and warm.turns == 4000


class FakeBackbone(nn.Module):
    """Stands in for models.<name>.Model: same (logits, state) contract, same
    trainable parameter names (lora_*, marker_delta)."""

    def __init__(self, vocab: int = 32, rank: int = DEFAULT_RANK):
        super().__init__()
        self.embedding = nn.Embedding(vocab, 8)
        self.head = nn.Sequential(nn.Linear(8, vocab))
        self.marker_delta = nn.Module()
        self.marker_delta.delta = nn.Parameter(torch.zeros(2, 8))
        apply_lora(self.head, ["0"], rank=rank, alpha=DEFAULT_ALPHA, dropout=DEFAULT_DROPOUT)

    def forward(self, input_ids, state=None):
        h = self.embedding(input_ids) + self.marker_delta.delta.sum(0)
        return self.head(h), state


class FakeHooks:
    @staticmethod
    def setup_training(device, rank, alpha, dropout):
        model = FakeBackbone(rank=rank)
        trainable = [p for n, p in model.named_parameters()
                     if "lora_A" in n or "lora_B" in n or "marker_delta" in n]
        for p in model.parameters():
            p.requires_grad_(False)
        for p in trainable:
            p.requires_grad_(True)
        return model, trainable

    @staticmethod
    def chunk_loss(model, input_ids, target_ids, mask_slice, state, eos_weight):
        logits, state = model(input_ids, state=state)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="none")
        weights = mask_slice.reshape(-1).float()
        return (loss * weights).sum(), weights.sum(), state


def _fake_corpus() -> list[tuple[list[int], list[bool]]]:
    torch.manual_seed(0)
    return [([1] + list(range(2, 20)), [False] * 9 + [True] * 10) for _ in range(3)]


def test_two_step_run_trains_the_adapter_and_the_checkpoint_loads_back(tmp_path):
    model, trainable = FakeHooks.setup_training("cpu", DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT)
    before = [p.detach().clone() for p in trainable]
    ckpt = tmp_path / "warm_start.pt"

    train(model, FakeHooks, trainable, _fake_corpus(), steps=2, lr=1e-3,
          chunk_len=8, seed=0, device=torch.device("cpu"), log_every=1)
    from lora import save_adapter
    save_adapter(model, ckpt, DEFAULT_RANK, DEFAULT_ALPHA)

    assert any(not torch.equal(p, b) for p, b in zip(trainable, before, strict=True))

    fresh, _ = FakeHooks.setup_training("cpu", DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT)
    load_adapter(fresh, ckpt, DEFAULT_RANK, DEFAULT_ALPHA)

    from lora import adapter_state_dict
    want = adapter_state_dict(model)
    for name, got in adapter_state_dict(fresh).items():
        torch.testing.assert_close(got, want[name], msg=name)


def test_training_only_counts_chunks_that_carry_a_gradient():
    """A conversation opens with user-turn tokens only; those chunks have no
    trained tokens and must not burn steps from the pinned budget."""
    model, trainable = FakeHooks.setup_training("cpu", DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT)
    seen = []

    hooks = FakeHooks()
    real = FakeHooks.chunk_loss

    def counting(model_, inp, tgt, mask, state, eos_weight):
        seen.append(mask.sum().item())
        return real(model_, inp, tgt, mask, state, eos_weight)

    hooks.chunk_loss = counting
    stepped = train(model, hooks, trainable, _fake_corpus(), steps=3, lr=1e-3,
                    chunk_len=4, seed=0, device=torch.device("cpu"), log_every=10)

    assert stepped == 3
    assert 0.0 in seen  # a no-gradient chunk really did occur
    assert len(seen) > 3
