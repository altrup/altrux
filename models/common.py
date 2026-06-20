import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def quantize_lora_targets(model: nn.Module, target_modules: list[str]) -> None:
    """Replace each nn.Linear submodule of `model` whose name ends with one of
    `target_modules` with a 4-bit (NF4) `bitsandbytes.nn.Linear4bit`, in place.

    For QLoRA: only the modules that will later get a LoRA adapter attached
    (see `apply_lora` in sft/lora.py and backend/app/model/lora.py) are
    quantized and frozen here. Everything else in the model -- embeddings,
    norms, other backbone params, and any from-scratch subsystem trained with
    full gradients -- is left untouched at full precision.

    A model opts into this by setting `QUANTIZE_LORA_BASE = True` and calling
    this from its own `load_base()`; models that don't set it (the default)
    use plain full-precision LoRA, unaffected by this function.
    """
    import bitsandbytes as bnb

    named = dict(model.named_modules())
    quantized = 0
    for name, module in list(named.items()):
        if not isinstance(module, nn.Linear) or isinstance(module, bnb.nn.Linear4bit):
            continue
        if not any(name.endswith(t) for t in target_modules):
            continue
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model if not parent_name else named[parent_name]

        # compute_dtype defaults to None (lazily inferred from the first forward
        # call's input dtype) if left unset -- pin it explicitly to the original
        # module's float dtype so it's never None and never the packed-uint8
        # storage dtype that .weight.dtype reports post-quantization. Callers
        # (LoRALinear in sft/lora.py, backend/app/model/lora.py) read this
        # attribute to size their adapter params correctly.
        quant = bnb.nn.Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            quant_type="nf4",
            compute_dtype=module.weight.dtype,
        )
        quant.weight = bnb.nn.Params4bit(module.weight.data.clone(), requires_grad=False, quant_type="nf4")
        if module.bias is not None:
            quant.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
        # Params4bit only actually quantizes its data on a device move.
        setattr(parent, child_name, quant.to(module.weight.device))
        quantized += 1

    if quantized == 0:
        raise RuntimeError(f"quantize_lora_targets matched 0 modules for targets {target_modules}")
    print(f"quantized {quantized} modules to 4-bit (nf4): {target_modules}")


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
