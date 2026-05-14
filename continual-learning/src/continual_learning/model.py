import copy
import json
from pathlib import Path
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

        # Match the base model's device and dtype (bfloat16 by default).
        base_param = next(self.base_model.parameters())
        self.critic_layers.to(device=base_param.device, dtype=base_param.dtype)
        self.critic_head.to(device=base_param.device, dtype=base_param.dtype)

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

    def save(self, checkpoint_dir: str | Path, step: Optional[int] = None,
             keep_checkpoints: int = 2) -> Path:
        """Save critic weights and base model to a checkpoint directory.

        Saves to <checkpoint_dir>/latest/ and, if step is provided, also to
        <checkpoint_dir>/step_<N>/ as a numbered snapshot.
        """
        root = Path(checkpoint_dir)

        def _write(dest: Path) -> None:
            dest.mkdir(parents=True, exist_ok=True)
            torch.save(self.critic_layers.state_dict(), dest / "critic_layers.pt")
            torch.save(self.critic_head.state_dict(), dest / "critic_head.pt")
            self.base_model.save_pretrained(dest / "base_model")
            self.tokenizer.save_pretrained(dest / "base_model")
            meta = {"step": step, "split_idx": self.split_idx}
            (dest / "meta.json").write_text(json.dumps(meta, indent=2))

        latest = root / "latest"
        _write(latest)

        if step is not None:
            _write(root / f"step_{step}")

            if keep_checkpoints:
                import shutil
                snapshots = sorted(
                    (d for d in root.iterdir() if d.is_dir() and d.name.startswith("step_")),
                    key=lambda d: int(d.name.split("_")[1]),
                )
                for old in snapshots[:-keep_checkpoints]:
                    shutil.rmtree(old)

        return latest

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        snapshot: Optional[str] = None,
        **model_kwargs,
    ) -> "ContinualLearningModel":
        """Load a checkpoint saved by save().

        Args:
            checkpoint_dir: root checkpoints folder.
            snapshot: sub-folder name to load (e.g. "step_50"). Defaults to "latest".
            **model_kwargs: forwarded to ContinualLearningModel.__init__
                            (e.g. load_in_4bit=True).
        """
        root = Path(checkpoint_dir) / (snapshot or "latest")
        meta = json.loads((root / "meta.json").read_text())

        model = cls(model_name=str(root / "base_model"), **model_kwargs)

        model.critic_layers.load_state_dict(
            torch.load(root / "critic_layers.pt", map_location=model.device)
        )
        model.critic_head.load_state_dict(
            torch.load(root / "critic_head.pt", map_location=model.device)
        )
        return model
