import asyncio
import importlib

import torch

from models.common import build_tokenizer


class ModelRegistry:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.lock = asyncio.Lock()

        self.user_open: str = ""
        self.asst_open: str = ""

        self.session_input_ids: list[int] = []
        # Per-layer SSM/conv (and, for mamba2_780m_memory, memory) state --
        # every model's Model.forward(input_ids, state) -> (logits, state)
        # convention, carried across generate_one_token calls. Not an
        # mamba_ssm InferenceParams object: none of this repo's models use
        # that anymore (their fused kernels are broken on this hardware --
        # see models/mamba2_780m/model.py), so they all manage their own
        # plain-PyTorch state object instead.
        self._cache = None

        # Structured message list kept in sync with token storage.
        # Only updated by append_message(); append_input() does not touch it.
        self.messages: list[dict] = []

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self) -> None:
        from ..config import get_checkpoint, get_device, get_model_name
        from .lora import apply_lora, load_checkpoint, read_lora_config
        # Limit PyTorch's OpenMP thread pool — GPU inference doesn't need many CPU threads
        # and excess threads spin-wait, compounding the ROCm HSA busy-wait problem.
        torch.set_num_threads(4)
        device = get_device()
        if device.startswith("cuda"):
            print(f"device: {device} — {torch.cuda.get_device_name(device)} (index {torch.cuda.current_device()})")
            print(f"  VRAM total:  {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
            print(f"  VRAM free:   {torch.cuda.mem_get_info(device)[0] / 1024**3:.1f} GB")
        else:
            print(f"device: {device} (no CUDA/ROCm device found)")

        model_name = get_model_name()
        print(f"loading model: {model_name}")
        model_mod = importlib.import_module(f"models.{model_name}")

        self.user_open = model_mod.USER_OPEN + " "
        self.asst_open = model_mod.ASST_OPEN + " "

        base = model_mod.load_base(device)
        ckpt = get_checkpoint()
        if ckpt is not None:
            rank, alpha = read_lora_config(ckpt)
            apply_lora(base, model_mod.TARGET_LORA_MODULES, rank, alpha)
        self.model = model_mod.Model(base)
        if ckpt is not None:
            # Loaded *after* wrapping in Model, not before: a model like
            # mamba2_780m_memory has trainable state (front_end, injections)
            # that only exists on the Model wrapper, not on the raw backbone
            # -- loading into `base` would silently miss those keys.
            print(f"loading checkpoint: {ckpt}")
            load_checkpoint(self.model, ckpt)
        # Move everything to device after checkpoint load: models like
        # mamba2_780m_memory add new nn.Modules (front_end, injections) that
        # initialize on CPU and are not covered by load_base's device placement.
        self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = build_tokenizer(model_mod)
        dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
        with torch.no_grad():
            self.model(dummy)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self, text: str | None = None) -> None:
        self.session_input_ids = []
        self._cache = None
        self.messages = []
        if text:
            assert self.tokenizer is not None
            self.session_input_ids = self.tokenizer.encode(text)

    def append_input(self, text: str) -> None:
        """Append raw text to the session and invalidate the cache."""
        assert self.tokenizer is not None
        self.session_input_ids.extend(self.tokenizer.encode(text))
        self._cache = None

    def append_message(self, role: str, content: str) -> None:
        """Append a chat turn formatted with the model's role openers.

        Matches the fine-tuning format (see sft/prepare_data.py): user turns are
        `user_open + content + "\\n"`; assistant turns are `asst_open + content`
        followed by EOS. Special tokens are not added — the openers carry the
        structure, exactly as during training.
        """
        assert self.tokenizer is not None

        if role == "user":
            ids = self.tokenizer.encode(self.user_open + content + "\n", add_special_tokens=False)
        elif role == "assistant":
            ids = self.tokenizer.encode(self.asst_open + content, add_special_tokens=False)
            ids = ids + [self.tokenizer.eos_token_id]
        else:
            raise ValueError(f"unknown role: {role!r}")

        self.session_input_ids.extend(ids)
        self._cache = None
        self.messages.append({"role": role, "content": content})

    def get_session_tokens(self) -> list[dict]:
        assert self.tokenizer is not None
        return [
            {"id": tid, "text": self.tokenizer.decode([tid])}
            for tid in self.session_input_ids
        ]

    def get_session_text(self) -> str:
        assert self.tokenizer is not None
        return self.tokenizer.decode(self.session_input_ids)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_one_token(self, temperature: float = 0.8, top_p: float = 0.95) -> dict:
        assert self.model is not None and self.tokenizer is not None
        device = next(self.model.parameters()).device

        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            raise RuntimeError("Tokenizer has no bos_token_id — cannot seed an empty session")

        if self._cache is None:
            # Cold start: process the full session, starting state from scratch.
            input_ids = torch.tensor(
                [self.session_input_ids or [bos_id]],
                dtype=torch.long,
                device=device,
            )
        else:
            # Incremental: only process the token appended by the previous call,
            # continuing from the state returned by that call.
            input_ids = torch.tensor(
                [[self.session_input_ids[-1]]],
                dtype=torch.long,
                device=device,
            )

        with torch.no_grad():
            logits, self._cache = self.model(input_ids, state=self._cache)

        next_token_logits = logits[0, -1, :]
        if temperature == 0.0:
            next_token_id = int(torch.argmax(next_token_logits).item())
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                sorted_probs[cumulative - sorted_probs > top_p] = 0.0
                probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
                probs /= probs.sum()
            next_token_id = int(torch.multinomial(probs, num_samples=1).item())
        self.session_input_ids.append(next_token_id)

        return {
            "generated_token": self.tokenizer.decode([next_token_id]),
            "token_id": next_token_id,
            "is_eos": next_token_id == self.tokenizer.eos_token_id,
        }


registry = ModelRegistry()
