import asyncio
import importlib

import torch
from transformers import AutoTokenizer
from mamba_ssm.utils.generation import InferenceParams

_MAX_SEQ = 8192


class ModelRegistry:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.lock = asyncio.Lock()

        self.user_open: str = ""
        self.asst_open: str = ""

        self.session_input_ids: list[int] = []
        self._cache: InferenceParams | None = None

        # Structured message list kept in sync with token storage.
        # Only updated by append_message(); append_input() does not touch it.
        self.messages: list[dict] = []

        # Revise data collection state — reset with each session.
        self.revise_entry_offset: int | None = None
        self.revise_suggestions: list[tuple[int, str]] = []

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self) -> None:
        from ..config import get_device, get_model_name, get_sft_checkpoint
        from .lora import apply_lora, load_lora, read_lora_config
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

        self.user_open = model_mod.USER_OPEN
        self.asst_open = model_mod.ASST_OPEN

        base = model_mod.load_base(device)
        ckpt = get_sft_checkpoint()
        if ckpt is not None:
            print(f"loading SFT checkpoint: {ckpt}")
            rank, alpha = read_lora_config(ckpt)
            apply_lora(base, model_mod.TARGET_LORA_MODULES, rank, alpha)
            load_lora(base, ckpt)
        self.model = model_mod.Model(base)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_mod.TOKENIZER_ID, local_files_only=True)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_mod.TOKENIZER_ID)
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
        self.revise_entry_offset = None
        self.revise_suggestions = []
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
            # Cold start: process the full session to hydrate the cache.
            self._cache = InferenceParams(max_seqlen=_MAX_SEQ, max_batch_size=1)
            input_ids = torch.tensor(
                [self.session_input_ids or [bos_id]],
                dtype=torch.long,
                device=device,
            )
        else:
            # Incremental: only process the token appended by the previous call.
            input_ids = torch.tensor(
                [[self.session_input_ids[-1]]],
                dtype=torch.long,
                device=device,
            )

        with torch.no_grad():
            logits = self.model(input_ids, inference_params=self._cache)

        self._cache.seqlen_offset += input_ids.shape[1]

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
