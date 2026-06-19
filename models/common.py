import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def build_tokenizer(model_mod) -> PreTrainedTokenizerBase:
    """Load model_mod.TOKENIZER_ID and register model_mod.SPECIAL_TOKENS.

    Deterministic given the same TOKENIZER_ID + SPECIAL_TOKENS list: any two
    callers (sft and backend) building a tokenizer this way get matching token
    ids, even though they never share a tokenizer instance.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_mod.TOKENIZER_ID, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(model_mod.TOKENIZER_ID)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    tokenizer.add_special_tokens({"additional_special_tokens": model_mod.SPECIAL_TOKENS})
    return tokenizer


def extend_embeddings(model: nn.Module, new_vocab_size: int) -> None:
    """Grow model.backbone.embedding and model.lm_head to new_vocab_size in place.

    MambaLMHeadModel is a plain nn.Module, not a HF PreTrainedModel, so there is
    no built-in resize_token_embeddings — this reassigns the embedding/lm_head
    attributes with larger tensors, copying existing rows and initializing new
    ones from the existing embedding's std. Re-ties lm_head.weight to the new
    embedding.weight when model.config.tie_embeddings is set.
    """
    embedding = model.backbone.embedding
    old_vocab_size, d_model = embedding.weight.shape
    if new_vocab_size <= old_vocab_size:
        return

    std = embedding.weight.std().item()
    device = embedding.weight.device
    dtype = embedding.weight.dtype

    new_embedding = nn.Embedding(new_vocab_size, d_model, device=device, dtype=dtype)
    with torch.no_grad():
        new_embedding.weight[:old_vocab_size] = embedding.weight
        new_embedding.weight[old_vocab_size:].normal_(mean=0.0, std=std)
    model.backbone.embedding = new_embedding

    if getattr(model.config, "tie_embeddings", False):
        model.lm_head.weight = new_embedding.weight
        return

    old_lm_head = model.lm_head
    new_lm_head = nn.Linear(d_model, new_vocab_size, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        new_lm_head.weight[:old_vocab_size] = old_lm_head.weight
        new_lm_head.weight[old_vocab_size:].normal_(mean=0.0, std=std)
    model.lm_head = new_lm_head
