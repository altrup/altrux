import torch
import torch.nn as nn

from transformers import Mamba2ForCausalLM
from transformers.models.mamba2.modeling_mamba2 import Mamba2Block, Mamba2RMSNorm


class ContinualLearningModel(nn.Module):
    def __init__(self, mamba_model: Mamba2ForCausalLM):
        super().__init__()
        backbone = mamba_model.backbone
        config = mamba_model.config
        n_layers = len(backbone.layers)
        trunk_end = (n_layers * 2) // 3
        critic_depth = n_layers // 3

        self.d_model: int = config.hidden_size
        self._config = config

        self.embedding = backbone.embeddings
        self.trunk_layers = nn.ModuleList(list(backbone.layers[:trunk_end]))
        self.main_layers = nn.ModuleList(list(backbone.layers[trunk_end:]))
        self.norm_f = backbone.norm_f
        self.lm_head = mamba_model.lm_head

        device = backbone.embeddings.weight.device
        dtype = backbone.embeddings.weight.dtype

        self.critic_layers = nn.ModuleList([
            Mamba2Block(config, layer_idx=i).to(device=device, dtype=dtype)
            for i in range(critic_depth)
        ])
        self.critic_norm_f = Mamba2RMSNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        ).to(device=device, dtype=dtype)
        self.critic_head = nn.Sequential(
            nn.Linear(config.hidden_size, 1, device=device, dtype=dtype),
            nn.Tanh(),
        )

    def forward(
        self, input_ids: torch.Tensor, run_critic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Returns (logits, per_token_rewards, trunk_hidden).
          logits:             (B, T, vocab_size)
          per_token_rewards:  (B, T) in [-1, 1]  — None when run_critic=False
          trunk_hidden:       (B, T, d_model) float32, pre-norm trunk output
        """
        h = self.embedding(input_ids)

        for layer in self.trunk_layers:
            h = layer(h)

        # Trunk output in float32; used for data collection and critic entry.
        # Each Mamba2Block already adds its own residual internally.
        trunk_hidden = h.float()

        # Main path: continue through remaining layers then lm_head
        h2 = h
        for layer in self.main_layers:
            h2 = layer(h2)
        logits = self.lm_head(self.norm_f(h2))

        # Critic path (optional)
        per_token_rewards = None
        if run_critic:
            h_c = trunk_hidden.detach().to(h.dtype)
            for layer in self.critic_layers:
                h_c = layer(h_c)
            per_token_rewards = self.critic_head(self.critic_norm_f(h_c)).squeeze(-1)

        return logits, per_token_rewards, trunk_hidden
