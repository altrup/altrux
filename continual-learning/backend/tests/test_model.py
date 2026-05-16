import torch
import pytest


def test_split_idx_is_two_thirds(tiny_model):
    n = len(tiny_model.base_model.model.layers)
    assert tiny_model.split_idx == round(n * 2 / 3)


def test_critic_depth_is_two_thirds(tiny_model):
    n = len(tiny_model.base_model.model.layers)
    assert len(tiny_model.critic_layers) == round(n * 2 / 3)


def test_critic_layers_are_independent_from_base(tiny_model):
    base_ids = {id(p) for p in tiny_model.base_model.parameters()}
    for p in tiny_model.critic_layers.parameters():
        assert id(p) not in base_ids


def test_forward_returns_reward_per_batch_item(tiny_model):
    input_ids = torch.randint(0, 50, (2, 8))
    _, reward = tiny_model(input_ids)
    assert reward.shape == (2,)


def test_reward_in_unit_interval(tiny_model):
    input_ids = torch.randint(0, 50, (1, 8))
    _, reward = tiny_model(input_ids)
    assert -1.0 <= reward.item() <= 1.0


def test_hook_populates_split_hidden(tiny_model):
    input_ids = torch.randint(0, 50, (1, 8))
    tiny_model(input_ids)
    h = tiny_model._split_hidden
    assert h is not None
    assert h.shape == (1, 8, tiny_model.base_model.config.hidden_size)


def test_generate_returns_string_and_float(tiny_model):
    text, reward = tiny_model.generate("hello")
    assert isinstance(text, str)
    assert isinstance(reward, float)
    assert -1.0 <= reward <= 1.0
