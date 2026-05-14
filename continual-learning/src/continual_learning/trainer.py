from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F
from .model import ContinualLearningModel


@dataclass
class TrainingConfig:
    critic_lr: float = 1e-4
    base_model_lr: float = 1e-6
    max_grad_norm: float = 1.0
    max_seq_len: int = 512
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10  # save every N steps; 0 to disable auto-save
    keep_checkpoints: int = 1  # number of numbered auto-save snapshots to keep; 0 to keep all


class Trainer:
    def __init__(self, model: ContinualLearningModel, config: TrainingConfig = TrainingConfig()):
        self.model = model
        self.config = config
        self.history: list[dict] = []

        critic_params = (
            list(model.critic_layers.parameters()) + list(model.critic_head.parameters())
        )
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=config.critic_lr)

        base_params = [p for p in model.base_model.parameters() if p.requires_grad]
        self.base_optimizer: Optional[torch.optim.Optimizer] = (
            torch.optim.Adam(base_params, lr=config.base_model_lr) if base_params else None
        )

    def _encode(self, text: str) -> torch.Tensor:
        return self.model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len,
        ).to(self.model.device)["input_ids"]

    def critic_step(self, prompt: str, response: str, user_reward: float) -> float:
        """Phase 1: train the critic to predict user rewards. Base model is not updated."""
        self.model.train()

        _, predicted_reward = self.model(self._encode(prompt + response))

        target = torch.tensor(
            [user_reward], device=self.model.device, dtype=predicted_reward.dtype
        )
        loss = F.mse_loss(predicted_reward, target)

        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.critic_layers.parameters())
            + list(self.model.critic_head.parameters()),
            self.config.max_grad_norm,
        )
        self.critic_optimizer.step()
        # base_optimizer intentionally not touched

        loss_val = loss.item()
        self.history.append({
            "phase": 1,
            "user_reward": user_reward,
            "predicted_reward": predicted_reward.item(),
            "loss": loss_val,
        })

        self.model.eval()
        self._maybe_save()
        return loss_val

    def policy_step(self, prompt: str, response: str) -> tuple[float, float]:
        """Phase 2: critic is frozen; its reward signal updates the base model.

        The gradient of -reward flows through the critic's computation graph back
        through _split_hidden into the base model trunk, pushing it toward outputs
        the critic scores highly.

        Returns (reward_value, loss_value).
        """
        if self.base_optimizer is None:
            raise RuntimeError("No trainable base model parameters found.")

        self.model.train()

        _, predicted_reward = self.model(self._encode(prompt + response))

        # Maximise the critic's reward signal; critic params are not stepped so
        # they act as a frozen scoring function.
        loss = -predicted_reward.mean()

        self.base_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()  # clear critic grads, will not be stepped
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.base_model.parameters()),
            self.config.max_grad_norm,
        )
        self.base_optimizer.step()
        # critic_optimizer intentionally not stepped

        reward_val = predicted_reward.mean().item()
        loss_val = loss.item()
        self.history.append({
            "phase": 2,
            "predicted_reward": reward_val,
            "loss": loss_val,
        })

        self.model.eval()
        self._maybe_save()
        return reward_val, loss_val

    def _maybe_save(self) -> None:
        n = len(self.history)
        if self.config.save_every and n % self.config.save_every == 0:
            self.model.save(self.config.checkpoint_dir, step=n,
                            keep_checkpoints=self.config.keep_checkpoints,
                            is_manual=False)
            print(f"[checkpoint] auto-saved at step {n} → {self.config.checkpoint_dir}/")

    def save_now(self) -> None:
        """Manually trigger a checkpoint outside the auto-save schedule."""
        n = len(self.history)
        self.model.save(self.config.checkpoint_dir, step=n, is_manual=True)
        print(f"[checkpoint] manual save at step {n} → {self.config.checkpoint_dir}/")
