import asyncio
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams

from .architecture import ContinualLearningModel

_MODEL_ID = "state-spaces/mamba2-780m"
_TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
_MAX_SEQ = 8192

_DAT_PATH = Path("data/collected/trunk_hiddens.dat")
_JSONL_PATH = Path("data/collected/records.jsonl")


class ModelRegistry:
    def __init__(self) -> None:
        self.model: ContinualLearningModel | None = None
        self.tokenizer = None
        self._mode: Literal["frozen", "unfrozen"] = "frozen"
        self.lock = asyncio.Lock()

        self.session_input_ids: list[int] = []
        self.pending_trunk_hidden: np.ndarray | None = None  # (d_model,) float32
        self.pending_token_id: int | None = None

        self._cache: InferenceParams | None = None
        self._critic_cache: InferenceParams | None = None

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        await asyncio.to_thread(self._load_blocking)
        self._reconcile_data_files()

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
        self.set_mode("frozen")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID, local_files_only=True)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)
        _DAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DAT_PATH.touch(exist_ok=True)
        _JSONL_PATH.touch(exist_ok=True)
        dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
        with torch.no_grad():
            self.model(dummy)

    def _reconcile_data_files(self) -> None:
        """Truncate both files to min(n_hiddens, n_records) to fix crash-torn state."""
        assert self.model is not None
        d_model = self.model.d_model
        row_bytes = d_model * 4

        n_hiddens = _DAT_PATH.stat().st_size // row_bytes
        with open(_JSONL_PATH) as f:
            n_records = sum(1 for _ in f)

        n = min(n_hiddens, n_records)

        if _DAT_PATH.stat().st_size > n * row_bytes:
            with open(_DAT_PATH, "r+b") as f:
                f.truncate(n * row_bytes)

        if n_records > n:
            lines = _JSONL_PATH.read_text().splitlines(keepends=True)
            _JSONL_PATH.write_text("".join(lines[:n]))

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def set_mode(self, mode: Literal["frozen", "unfrozen"]) -> None:
        assert self.model is not None
        self._mode = mode
        self.pending_trunk_hidden = None
        self.pending_token_id = None
        self._cache = None
        self._critic_cache = None

        if mode == "frozen":
            for p in self.model.parameters():
                p.requires_grad_(False)
        else:
            # Unfrozen: base model trainable, critic frozen
            for p in self.model.parameters():
                p.requires_grad_(False)
            for component in (
                self.model.embedding,
                self.model.trunk_layers,
                self.model.main_layers,
                self.model.norm_f,
                self.model.lm_head,
            ):
                for p in component.parameters():
                    p.requires_grad_(True)

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_session(self, text: str | None = None) -> None:
        self.session_input_ids = []
        self.pending_trunk_hidden = None
        self.pending_token_id = None
        self._cache = None
        self._critic_cache = None
        if text:
            assert self.tokenizer is not None
            self.session_input_ids = self.tokenizer.encode(text)

    def append_input(self, text: str) -> None:
        """Append user text to the session and invalidate caches."""
        assert self.tokenizer is not None
        self.session_input_ids.extend(self.tokenizer.encode(text))
        self._cache = None
        self._critic_cache = None

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
            # Cold start: process the full session to hydrate both caches.
            self._cache = InferenceParams(max_seqlen=_MAX_SEQ, max_batch_size=1)
            self._critic_cache = InferenceParams(max_seqlen=_MAX_SEQ, max_batch_size=1)
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
            logits, per_token_rewards, trunk_hidden = self.model(
                input_ids,
                inference_params=self._cache,
                critic_inference_params=self._critic_cache,
            )

        self._cache.seqlen_offset += input_ids.shape[1]
        self._critic_cache.seqlen_offset += input_ids.shape[1]

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

        # Save trunk hidden at the last GENERATED position (before appending, index == -1)
        self.pending_trunk_hidden = trunk_hidden[0, -1, :].float().cpu().numpy()
        self.pending_token_id = next_token_id

        critic_reward = float(per_token_rewards[0, -1].item())
        if not (critic_reward == critic_reward):  # NaN check
            critic_reward = 0.0

        return {
            "generated_token": self.tokenizer.decode([next_token_id]),
            "token_id": next_token_id,
            "critic_reward": critic_reward,
            "is_eos": next_token_id == self.tokenizer.eos_token_id,
        }

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def save_reward(self, reward: float) -> int:
        """Append pending (trunk_hidden, token_id, reward) to both data files.

        Returns total record count after saving.
        Only writes in frozen mode.
        """
        assert self.pending_trunk_hidden is not None and self.pending_token_id is not None

        if self._mode == "frozen":
            index = _DAT_PATH.stat().st_size // (self.model.d_model * 4)
            with open(_DAT_PATH, "ab") as f:
                f.write(self.pending_trunk_hidden.tobytes())
            record = {
                "index": index,
                "token_id": self.pending_token_id,
                "reward": reward,
            }
            with open(_JSONL_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")

        self.pending_trunk_hidden = None
        self.pending_token_id = None

        return self.total_records()

    def total_records(self) -> int:
        assert self.model is not None
        return _DAT_PATH.stat().st_size // (self.model.d_model * 4)


registry = ModelRegistry()
