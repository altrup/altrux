import torch
import torch.nn as nn

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

MODEL_ID = "state-spaces/mamba2-780m"
TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
TARGET_LORA_MODULES = ["in_proj", "out_proj"]

# Chat format tokens — must match the SFT training format exactly.
# The trailing space is significant; keep it.
USER_OPEN = "[USER] "
ASST_OPEN = "[ASSISTANT] "


class Model(nn.Module):
    """Thin wrapper around the base Mamba LM for inference.

    Generates tokens until EOS. The model is fine-tuned to emit a <revise>
    tag after its response; for now that tag is treated like any other token.
    TODO: detect the <revise> tag and act on it.
    """

    def __init__(self, mamba_model: MambaLMHeadModel):
        super().__init__()
        backbone = mamba_model.backbone

        self.d_model: int = backbone.layers[0].mixer.d_model
        self.fused_add_norm: bool = backbone.fused_add_norm
        self.residual_in_fp32: bool = backbone.residual_in_fp32

        self.embedding = backbone.embedding
        self.layers = nn.ModuleList(list(backbone.layers))
        self.norm_f = backbone.norm_f
        self.lm_head = mamba_model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        inference_params=None,
    ) -> torch.Tensor:
        """Returns logits (B, T, vocab_size).

        When inference_params is provided, SSM states are updated in-place and
        incremental (single-token) inputs are supported.
        """
        h = self.embedding(input_ids)
        residual = None

        for layer in self.layers:
            h, residual = layer(h, residual, inference_params=inference_params)

        return self.lm_head(self._apply_norm_f(h, residual))

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


def load_base(device: str) -> MambaLMHeadModel:
    """Load the raw HuggingFace model. Used by sft/train.py."""
    return MambaLMHeadModel.from_pretrained(MODEL_ID, device=device)


def load_inference(device: str) -> Model:
    """Load and wrap the model for inference. Used by the backend registry."""
    return Model(load_base(device))
