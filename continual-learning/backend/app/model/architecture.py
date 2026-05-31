import copy
import torch
import torch.nn as nn

from transformers import Mamba2ForCausalLM
from transformers.cache_utils import Cache
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
        critic_cfg = copy.deepcopy(config)
        critic_cfg.num_hidden_layers = critic_depth
        self._critic_config = critic_cfg

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
        self,
        input_ids: torch.Tensor,
        run_critic: bool = False,
        cache_params: Cache | None = None,
        critic_cache_params: Cache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Returns (logits, per_token_rewards, trunk_hidden).
          logits:             (B, T, vocab_size)
          per_token_rewards:  (B, T) in [-1, 1]  — None when run_critic=False
          trunk_hidden:       (B, T, d_model) model-dtype, pre-norm trunk output

        When cache_params / critic_cache_params are provided, caches are updated
        in-place and incremental (single-token) inputs are supported.
        """
        h = self.embedding(input_ids)

        for layer in self.trunk_layers:
            h = layer(h, cache_params=cache_params)

        # Keep trunk output in model dtype; caller casts the single position it needs.
        trunk_hidden = h

        # Main path
        h2 = h
        for layer in self.main_layers:
            h2 = layer(h2, cache_params=cache_params)
        logits = self.lm_head(self.norm_f(h2))

        # Critic path: always run when a critic cache is active so the state stays
        # in sync with the main model, even if rewards aren't being returned.
        per_token_rewards = None
        if run_critic or critic_cache_params is not None:
            h_c = trunk_hidden.detach()
            for layer in self.critic_layers:
                h_c = layer(h_c, cache_params=critic_cache_params)
            if run_critic:
                per_token_rewards = self.critic_head(self.critic_norm_f(h_c)).squeeze(-1)

        return logits, per_token_rewards, trunk_hidden
