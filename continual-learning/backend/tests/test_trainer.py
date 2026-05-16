import math
import torch
import pytest


def _snapshot(module):
    return {n: p.detach().clone() for n, p in module.named_parameters()}


def _params_unchanged(before: dict, module) -> list[str]:
    changed = []
    for name, p in module.named_parameters():
        if not torch.equal(before[name], p.detach()):
            changed.append(name)
    return changed


def _params_changed(before: dict, module) -> list[str]:
    changed = []
    for name, p in module.named_parameters():
        if not torch.equal(before[name], p.detach()):
            changed.append(name)
    return changed


def test_critic_step_does_not_touch_base_model(trainer):
    before = _snapshot(trainer.model.base_model)
    trainer.critic_step("hello", " world", 0.5)
    changed = _params_unchanged(before, trainer.model.base_model)
    assert changed == [], f"Base model params changed during critic_step: {changed}"


def test_critic_step_updates_critic_layers(trainer):
    before = _snapshot(trainer.model.critic_layers)
    trainer.critic_step("hello", " world", 0.5)
    changed = _params_changed(before, trainer.model.critic_layers)
    assert changed, "No critic_layers params were updated during critic_step"


def test_critic_step_loss_is_finite(trainer):
    loss = trainer.critic_step("hello", " world", 0.5)
    assert math.isfinite(loss)


def test_critic_step_records_phase_1(trainer):
    trainer.critic_step("hello", " world", 0.3)
    assert trainer.history[-1]["phase"] == 1
    assert trainer.history[-1]["user_reward"] == pytest.approx(0.3)


def test_policy_step_does_not_touch_critic(trainer):
    before_layers = _snapshot(trainer.model.critic_layers)
    before_head = _snapshot(trainer.model.critic_head)
    trainer.policy_step("hello", " world")
    assert _params_unchanged(before_layers, trainer.model.critic_layers) == []
    assert _params_unchanged(before_head, trainer.model.critic_head) == []


def test_policy_step_updates_base_model(trainer):
    before = _snapshot(trainer.model.base_model)
    trainer.policy_step("hello", " world")
    changed = _params_changed(before, trainer.model.base_model)
    assert changed, "No base model params were updated during policy_step"


def test_policy_step_loss_is_finite(trainer):
    reward, loss = trainer.policy_step("hello", " world")
    assert math.isfinite(reward)
    assert math.isfinite(loss)


def test_policy_step_loss_is_negative_reward(trainer):
    reward, loss = trainer.policy_step("hello", " world")
    assert loss == pytest.approx(-reward, abs=1e-5)


def test_policy_step_records_phase_2(trainer):
    trainer.policy_step("hello", " world")
    assert trainer.history[-1]["phase"] == 2


def test_history_grows_with_each_step(trainer):
    assert len(trainer.history) == 0
    trainer.critic_step("a", "b", 1.0)
    assert len(trainer.history) == 1
    trainer.policy_step("a", "b")
    assert len(trainer.history) == 2


def test_mixed_phases_recorded_correctly(trainer):
    trainer.critic_step("a", "b", -0.5)
    trainer.policy_step("a", "b")
    assert trainer.history[0]["phase"] == 1
    assert trainer.history[1]["phase"] == 2
