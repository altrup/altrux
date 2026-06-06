import asyncio

import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams

from .architecture import ContinualLearningModel

_MODEL_ID = "state-spaces/mamba2-780m"
_TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
_MAX_SEQ = 8192


class ModelRegistry:
    def __init__(self) -> None:
        self.model: ContinualLearningModel | None = None
        self.tokenizer = None
        self.lock = asyncio.Lock()

        self.session_input_ids: list[int] = []
        self._cache: InferenceParams | None = None

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self) -> None:
        from ..config import get_device, get_sft_checkpoint
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
        mamba_model = MambaLMHeadModel.from_pretrained(_MODEL_ID, device=device)
        ckpt = get_sft_checkpoint()
        if ckpt is not None:
            print(f"loading SFT checkpoint: {ckpt}")
            rank, alpha = read_lora_config(ckpt)
            apply_lora(mamba_model, ["in_proj", "out_proj"], rank, alpha)
            load_lora(mamba_model, ckpt)
        self.model = ContinualLearningModel(mamba_model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID, local_files_only=True)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)
        dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
        with torch.no_grad():
            self.model(dummy)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self, text: str | None = None) -> None:
        self.session_input_ids = []
        self._cache = None
        if text:
            assert self.tokenizer is not None
            self.session_input_ids = self.tokenizer.encode(text)

    def append_input(self, text: str) -> None:
        """Append raw text to the session and invalidate the cache."""
        assert self.tokenizer is not None
        self.session_input_ids.extend(self.tokenizer.encode(text))
        self._cache = None

    def append_message(self, role: str, content: str) -> None:
        """Append a chat turn formatted with the configured role openers.

        Matches the fine-tuning format (see sft/prepare_data.py): user turns are
        `USER_OPEN + content + "\\n"`; assistant turns are `ASST_OPEN + content`
        followed by EOS. Special tokens are not added — the openers carry the
        structure, exactly as during training.
        """
        assert self.tokenizer is not None
        from ..config import ASST_OPEN, USER_OPEN

        if role == "user":
            ids = self.tokenizer.encode(USER_OPEN + content + "\n", add_special_tokens=False)
        elif role == "assistant":
            ids = self.tokenizer.encode(ASST_OPEN + content, add_special_tokens=False)
            ids = ids + [self.tokenizer.eos_token_id]
        else:
            raise ValueError(f"unknown role: {role!r}")

        self.session_input_ids.extend(ids)
        self._cache = None

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
