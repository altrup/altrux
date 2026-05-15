import json
import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow the allocator to map non-contiguous virtual pages for large allocations so
# that memory fragmentation doesn't cause OOM when plenty of free pages exist.
# Must be set before the first CUDA tensor allocation (i.e. before model loading).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # pre-2.6 compat

SPLIT_RATIO = 2 / 3
CRITIC_DEPTH_RATIO = 2 / 3

DEFAULT_MODEL = "ibm-granite/granite-4.0-h-micro"

# Fraction of each GPU's total VRAM reserved for Mamba SSM compute intermediates.
# Raise this if you see OOM during generation; lower it to fit more weights on GPU.
COMPUTE_RESERVE_FRAC = 0.60


def _build_max_memory(device_ids: list[int] | None) -> dict | None:
    """Build an accelerate max_memory dict that reserves COMPUTE_RESERVE_FRAC of each
    included GPU for forward-pass scratch space, routing overflow to CPU.

    device_ids: GPU indices to use; None means all visible CUDA GPUs.
    Returns None on CPU-only systems (device_map="auto" will use CPU anyway).
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None
    included = set(device_ids if device_ids is not None else range(torch.cuda.device_count()))
    mem: dict = {}
    for i in range(torch.cuda.device_count()):
        if i in included:
            total = torch.cuda.get_device_properties(i).total_memory
            mem[i] = int(total * (1 - COMPUTE_RESERVE_FRAC))
        else:
            mem[i] = 0
    mem["cpu"] = "512GiB"
    return mem


class ContinualLearningModel(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_ids: list[int] | None = None,
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
        # lm_head.weight is weight-tied to embed_tokens.weight in this model family
        # and is intentionally absent from the checkpoint. Transformers treats it as
        # a missing key and tries to reinitialize it on GPU, which fails on some ROCm
        # devices. We skip that step and call tie_weights() ourselves after loading.

        _max_memory = _build_max_memory(device_ids)

        from transformers.modeling_utils import PreTrainedModel
        _orig_init_missing = PreTrainedModel._initialize_missing_keys

        def _patched_init_missing(model_self_, *a, **kw):
            try:
                return _orig_init_missing(model_self_, *a, **kw)
            except Exception as e:
                msg = str(e).lower()
                if "hip" in msg or "device function" in msg or "hiperrorinvaliddevicefunction" in msg:
                    pass  # lm_head.weight ROCm GPU reinit failure; tie_weights() fixes it
                else:
                    raise

        PreTrainedModel._initialize_missing_keys = _patched_init_missing
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                max_memory=_max_memory,
                **quantization_kwargs,
            )
        finally:
            PreTrainedModel._initialize_missing_keys = _orig_init_missing

        if hasattr(self.base_model, "tie_weights"):
            self.base_model.tie_weights()

        # When device_map offloads decoder layers to CPU, accelerate attaches an
        # AlignDevicesHook per module with place_submodules=False. pre_forward then
        # only loads the module's own parameters (recurse=False), skipping child
        # module parameters like mamba.conv1d.weight. But torch_forward accesses
        # conv1d.weight directly (not via conv1d.forward()), bypassing conv1d's hook.
        # Setting place_submodules=True on any offload hook whose children also have
        # offload hooks makes pre_forward recurse, loading those child parameters too.
        try:
            from accelerate.hooks import AlignDevicesHook
            for mod in self.base_model.modules():
                hook = getattr(mod, "_hf_hook", None)
                if isinstance(hook, AlignDevicesHook) and hook.offload and not hook.place_submodules:
                    if any(
                        isinstance(getattr(child, "_hf_hook", None), AlignDevicesHook)
                        and getattr(child, "_hf_hook").offload
                        for child in mod.children()
                    ):
                        hook.place_submodules = True
        except ImportError:
            pass

        layers = list(self.base_model.model.layers)
        n = len(layers)
        self.split_idx = round(n * SPLIT_RATIO)
        critic_depth = round(n * CRITIC_DEPTH_RATIO)

        # Hook at the split point captures the hidden state for the critic branch
        self._split_hidden: Optional[torch.Tensor] = None
        layers[self.split_idx - 1].register_forward_hook(self._capture_hook)

        # Critic branch: critic_depth layers instantiated fresh from config.
        # Deep-copying GPU layers would require the same VRAM that the base model just consumed,
        # so we instantiate from config instead (random init, stays on CPU until dispatch_model
        # moves them). The critic is trained from scratch via human reward signals anyway.
        layer_cls = type(layers[0])
        config = self.base_model.config
        print(f"Building critic branch ({critic_depth} layers of {layer_cls.__name__}) …", flush=True)
        self.critic_layers = nn.ModuleList([
            layer_cls(config, layer_idx=i % self.split_idx)
            for i in range(critic_depth)
        ])
        print("Critic layers built.", flush=True)

        hidden_size = self.base_model.config.hidden_size
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )

        base_param = next(self.base_model.parameters())
        dtype = base_param.dtype

        self.critic_layers.to(dtype=dtype)
        self.critic_head.to(dtype=dtype)

        if base_param.device.type == "cpu":
            pass  # CPU-only path (tests / no GPU)
        else:
            from accelerate import dispatch_model, infer_auto_device_map

            # Flush the caching allocator so memory_allocated() reflects actual weights only
            torch.cuda.empty_cache()

            included = set(device_ids if device_ids is not None else range(torch.cuda.device_count()))
            critic_mem: dict = {}
            for _i in range(torch.cuda.device_count()):
                if _i in included:
                    _total = torch.cuda.get_device_properties(_i).total_memory
                    _weights_budget = int(_total * (1 - COMPUTE_RESERVE_FRAC))
                    _base_used = torch.cuda.memory_allocated(_i)
                    critic_mem[_i] = max(0, _weights_budget - _base_used)
                else:
                    critic_mem[_i] = 0
            critic_mem["cpu"] = "512GiB"

            layer_cls_name = type(self.critic_layers[0]).__name__

            class _Critic(nn.Module):
                def __init__(self, layers: nn.ModuleList, head: nn.Sequential):
                    super().__init__()
                    self.layers = layers
                    self.head = head

            _c = _Critic(self.critic_layers, self.critic_head)
            device_map = infer_auto_device_map(
                _c, max_memory=critic_mem, no_split_module_classes=[layer_cls_name]
            )
            print(f"Critic device map: {device_map}", flush=True)
            _c = dispatch_model(_c, device_map=device_map)
            self.critic_layers = _c.layers
            self.critic_head = _c.head

    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

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

        # Critic branch — critic lives on CPU; move the captured hidden state there.
        # .to() is differentiable, so Phase-2 gradients flow back to the base model.
        critic_device = next(self.critic_layers.parameters()).device
        hidden = self._split_hidden.to(critic_device)
        for layer in self.critic_layers:
            hidden = self._run_layer(layer, hidden)

        reward = self.critic_head(hidden.mean(dim=1)).squeeze(-1)
        return lm_output, reward.to(self.device)

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

    def generate_stream(self, prompt: str, max_new_tokens: int = 256):
        """Yield (partial_text, None) as tokens arrive, then (full_text, reward) when done."""
        import threading
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        def _worker():
            with torch.no_grad():
                self.base_model.generate(**gen_kwargs)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        generated = ""
        for chunk in streamer:
            generated += chunk
            yield generated, None

        thread.join()

        with torch.no_grad():
            full_ids = self.tokenizer(
                prompt + generated, return_tensors="pt"
            ).to(self.base_model.device)["input_ids"]
            _, estimated_reward = self.forward(full_ids)

        yield generated, estimated_reward.item()

    def save(
        self,
        checkpoint_dir: str | Path,
        step: Optional[int] = None,
        keep_checkpoints: int = 1,
        is_manual: bool = False,
    ) -> Path:
        """Save critic weights and base model to a checkpoint directory.

        Manual saves write only to latest_manual/ (single slot, always overwritten).
        Auto saves write to latest_auto/ and a numbered step_N/ snapshot; old
        snapshots beyond keep_checkpoints are deleted.
        """
        import shutil
        root = Path(checkpoint_dir)

        def _write(dest: Path) -> None:
            dest.mkdir(parents=True, exist_ok=True)
            torch.save(self.critic_layers.state_dict(), dest / "critic_layers.pt")
            torch.save(self.critic_head.state_dict(), dest / "critic_head.pt")
            self.base_model.save_pretrained(dest / "base_model")
            self.tokenizer.save_pretrained(dest / "base_model")
            (dest / "meta.json").write_text(
                json.dumps({"step": step, "split_idx": self.split_idx}, indent=2)
            )

        if is_manual:
            dest = root / "latest_manual"
            _write(dest)
            return dest

        # Auto save
        dest = root / "latest_auto"
        _write(dest)

        if step is not None:
            _write(root / f"step_{step}")

            if keep_checkpoints:
                snapshots = sorted(
                    (d for d in root.iterdir() if d.is_dir() and d.name.startswith("step_")),
                    key=lambda d: int(d.name.split("_")[1]),
                )
                for old in snapshots[:-keep_checkpoints]:
                    shutil.rmtree(old)

        return dest

    @classmethod
    def _most_recent_snapshot(cls, checkpoint_dir: str | Path) -> str:
        """Return the name of the most recent snapshot folder by step number."""
        root = Path(checkpoint_dir)
        best_step, best_name = -1, None
        for name in ("latest_manual", "latest_auto"):
            meta_file = root / name / "meta.json"
            if meta_file.exists():
                step = json.loads(meta_file.read_text()).get("step") or 0
                if step > best_step:
                    best_step, best_name = step, name
        if best_name is None:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        return best_name

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
            snapshot: sub-folder name to load (e.g. "latest_manual", "step_50").
                      Defaults to whichever of latest_manual/latest_auto is newer.
            **model_kwargs: forwarded to ContinualLearningModel.__init__
                            (e.g. load_in_4bit=True).
        """
        root = Path(checkpoint_dir)
        name = snapshot or cls._most_recent_snapshot(root)
        dest = root / name

        model = cls(model_name=str(dest / "base_model"), **model_kwargs)
        model.critic_layers.load_state_dict(
            torch.load(dest / "critic_layers.pt", map_location=model.device)
        )
        model.critic_head.load_state_dict(
            torch.load(dest / "critic_head.pt", map_location=model.device)
        )
        return model
