import copy
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

SPLIT_RATIO = 2 / 3
CRITIC_DEPTH_RATIO = 2 / 3

DEFAULT_MODEL = "ibm-granite/granite-4.0-h-micro"


class ContinualLearningModel(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        super().__init__()

        quantization_kwargs: dict = {}
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **quantization_kwargs,
        )

        layers = list(self.base_model.model.layers)
        n = len(layers)
        self.split_idx = round(n * SPLIT_RATIO)
        critic_depth = round(n * CRITIC_DEPTH_RATIO)

        # Hook at the split point captures the hidden state for the critic branch
        self._split_hidden: Optional[torch.Tensor] = None
        layers[self.split_idx - 1].register_forward_hook(self._capture_hook)

        # Critic branch: critic_depth layers, deep-copied and cycled from the trunk.
        # The model uses NoPE (no position embeddings) so layers can be called standalone.
        try:
            self.critic_layers = nn.ModuleList([
                copy.deepcopy(layers[i % self.split_idx]) for i in range(critic_depth)
            ])
        except Exception:
            # Quantized layers can't be deep-copied; re-instantiate from config instead
            layer_cls = type(layers[0])
            config = self.base_model.config
            self.critic_layers = nn.ModuleList([
                layer_cls(config, layer_idx=i % self.split_idx)
                for i in range(critic_depth)
            ])

        hidden_size = self.base_model.config.hidden_size
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

    @property
    def device(self) -> torch.device:
        return next(self.critic_head.parameters()).device

    def _capture_hook(
        self,
        module: nn.Module,
        inputs: tuple,
        output: tuple | torch.Tensor,
    ) -> None:
        self._split_hidden = output[0] if isinstance(output, (tuple, list)) else output

    def _run_layer(self, layer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        try:
            out = layer(hidden)
        except Exception:
            return hidden
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple:
        # Full base-model forward pass; hook fires at layer split_idx-1
        lm_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Critic branch on the captured hidden state
        hidden = self._split_hidden  # [batch, seq, hidden]
        for layer in self.critic_layers:
            hidden = self._run_layer(layer, hidden)

        # Mean-pool over sequence → scalar reward in [-1, 1]
        reward = self.critic_head(hidden.mean(dim=1)).squeeze(-1)

        return lm_output, reward

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 256) -> tuple[str, float]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)

        output_ids = self.base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Separate full-sequence forward pass to get an honest reward estimate
        _, estimated_reward = self.forward(output_ids)

        return response, estimated_reward.item()
