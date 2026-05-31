import torch
import torch.nn as nn

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, create_block
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn


class ContinualLearningModel(nn.Module):
    def __init__(self, mamba_model: MambaLMHeadModel):
        super().__init__()
        backbone = mamba_model.backbone
        n_layers = len(backbone.layers)
        trunk_end = (n_layers * 2) // 3
        critic_depth = n_layers // 3

        self.d_model: int = backbone.layers[0].mixer.d_model
        self.fused_add_norm: bool = backbone.fused_add_norm
        self.residual_in_fp32: bool = backbone.residual_in_fp32

        self.embedding = backbone.embedding
        self.trunk_layers = nn.ModuleList(list(backbone.layers[:trunk_end]))
        self.main_layers = nn.ModuleList(list(backbone.layers[trunk_end:]))
        self.norm_f = backbone.norm_f
        self.lm_head = mamba_model.lm_head

        device = backbone.embedding.weight.device
        dtype = backbone.embedding.weight.dtype

        self.critic_layers = nn.ModuleList(
            _make_critic_blocks(critic_depth, self.d_model, device, dtype)
        )
        self.critic_norm_f = RMSNorm(self.d_model, eps=1e-5, device=device, dtype=dtype)
        self.critic_head = nn.Sequential(
            nn.Linear(self.d_model, 1, device=device, dtype=dtype),
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
        residual = None

        for layer in self.trunk_layers:
            h, residual = layer(h, residual)

        # Unnormalized trunk output in float32; used for data collection and critic entry
        trunk_hidden = (h.float() + residual) if residual is not None else h.float()

        # Main path: continue through remaining layers then lm_head
        h2, r2 = h, residual
        for layer in self.main_layers:
            h2, r2 = layer(h2, r2)
        logits = self.lm_head(self._apply_norm_f(h2, r2))

        # Critic path (optional)
        per_token_rewards = None
        if run_critic:
            h_c = trunk_hidden.detach()
            r_c = None
            for layer in self.critic_layers:
                h_c, r_c = layer(h_c, r_c)
            c_out = self._apply_critic_norm_f(h_c, r_c)
            per_token_rewards = self.critic_head(c_out).squeeze(-1)

        return logits, per_token_rewards, trunk_hidden

    def _apply_norm_f(self, h: torch.Tensor, residual: torch.Tensor | None) -> torch.Tensor:
        if not self.fused_add_norm:
            combined = (h + residual) if residual is not None else h
            return self.norm_f(combined.to(self.norm_f.weight.dtype))
        return layer_norm_fn(
            h,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
            is_rms_norm=isinstance(self.norm_f, RMSNorm),
        )

    def _apply_critic_norm_f(
        self, h: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        if not self.fused_add_norm:
            combined = (h + residual) if residual is not None else h
            return self.critic_norm_f(combined.to(self.critic_norm_f.weight.dtype))
        return layer_norm_fn(
            h,
            self.critic_norm_f.weight,
            self.critic_norm_f.bias,
            eps=self.critic_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
            is_rms_norm=True,
        )


def _make_critic_blocks(
    n_blocks: int, d_model: int, device: torch.device, dtype: torch.dtype
) -> list:
    return [
        create_block(
            d_model=d_model,
            d_intermediate=0,
            ssm_cfg={"layer": "Mamba2"},
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            layer_idx=i,
            device=device,
            dtype=dtype,
        )
        for i in range(n_blocks)
    ]
