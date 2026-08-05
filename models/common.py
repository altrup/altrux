import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def blockwise_checkpoint(forward_fn, input_ids: torch.Tensor, state, block_len: int):
    """Run `forward_fn(input_ids_block, state) -> (logits, state)` over
    `block_len`-token blocks under `torch.utils.checkpoint`, threading `state`
    across them, and return the same `(logits, state)` the un-blocked call
    would have produced.

    Only each block's boundary state is retained for backward; the block's own
    activation graph is thrown away and recomputed when backward reaches it.
    Works for any model whose forward is already stateful and resumable over
    an arbitrary token count -- which is every model here, since that's what
    chunked training needs anyway.

    `state` must expose `flatten() -> tuple[Tensor, ...]` and
    `unflatten(tensors) -> state` (see MemoryState/MixerState): the carried
    tensors have to cross the checkpoint boundary as positional tensor
    arguments, or autograd cannot route gradient back through them.

    `use_reentrant=False` is load-bearing rather than stylistic: the memory
    models run a nested `torch.autograd.grad(..., create_graph=True)` inside
    the checkpointed region, and only the non-reentrant implementation replays
    that second-order graph correctly on recompute.
    """
    outs = []
    for start in range(0, input_ids.shape[1], block_len):
        ids = input_ids[:, start : start + block_len]

        def run(*flat, _ids=ids, _proto=state):
            logits, new_state = forward_fn(_ids, _proto.unflatten(flat))
            return (logits, *new_state.flatten())

        packed = torch.utils.checkpoint.checkpoint(run, *state.flatten(), use_reentrant=False)
        outs.append(packed[0])
        state = state.unflatten(packed[1:])
    return torch.cat(outs, dim=1), state


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


def extend_embeddings(model: nn.Module, new_vocab_size: int, tokenizer: PreTrainedTokenizerBase | None = None) -> None:
    """Grow model.backbone.embedding and model.lm_head to new_vocab_size in place.

    MambaLMHeadModel is a plain nn.Module, not a HF PreTrainedModel, so there is
    no built-in resize_token_embeddings — this reassigns the embedding/lm_head
    attributes with larger tensors, copying existing rows and initializing new
    ones from the existing embedding's std. Re-ties lm_head.weight to the new
    embedding.weight when model.config.tie_embeddings is set.

    With `tokenizer` (the build_tokenizer output), each registered
    additional-special-token row is additionally re-initialized to the MEAN of
    the rows of that token string's BPE spelling — a meaningful starting point
    instead of the random row it otherwise has, whether freshly grown or
    sitting inside the checkpoint's untrained vocab padding (Mamba pads
    50277 → 50288, so the marker ids usually need no growth at all). The
    random init stays only as fallback when the BPE spelling is unusable.
    Also records the special ids on the model as `marker_token_ids`, which the
    Model wrappers read to build their MarkerDelta.
    """
    embedding = model.backbone.embedding
    old_vocab_size, d_model = embedding.weight.shape
    tied = getattr(model.config, "tie_embeddings", False)

    if new_vocab_size > old_vocab_size:
        std = embedding.weight.std().item()
        device = embedding.weight.device
        dtype = embedding.weight.dtype

        new_embedding = nn.Embedding(new_vocab_size, d_model, device=device, dtype=dtype)
        with torch.no_grad():
            new_embedding.weight[:old_vocab_size] = embedding.weight
            new_embedding.weight[old_vocab_size:].normal_(mean=0.0, std=std)
        model.backbone.embedding = new_embedding

        if tied:
            model.lm_head.weight = new_embedding.weight
        else:
            old_lm_head = model.lm_head
            new_lm_head = nn.Linear(d_model, new_vocab_size, bias=False, device=device, dtype=dtype)
            with torch.no_grad():
                new_lm_head.weight[:old_vocab_size] = old_lm_head.weight
                new_lm_head.weight[old_vocab_size:].normal_(mean=0.0, std=std)
            model.lm_head = new_lm_head

    if tokenizer is None:
        return

    # transformers v5 renamed additional_special_tokens -> extra_special_tokens.
    markers = getattr(tokenizer, "additional_special_tokens", None) or getattr(tokenizer, "extra_special_tokens", [])
    marker_ids: list[int] = []
    with torch.no_grad():
        for token in markers:
            tid = tokenizer.convert_tokens_to_ids(token)
            marker_ids.append(tid)
            try:
                spelled = tokenizer(token, add_special_tokens=False, split_special_tokens=True)["input_ids"]
            except Exception:
                spelled = []
            spelled = [i for i in spelled if i < old_vocab_size and i != tid]
            if not spelled:
                continue  # fallback: keep the normal_-initialized row
            model.backbone.embedding.weight[tid] = model.backbone.embedding.weight[spelled].mean(dim=0)
            if not tied:
                model.lm_head.weight[tid] = model.lm_head.weight[spelled].mean(dim=0)
    model.marker_token_ids = marker_ids


class MarkerDelta(nn.Module):
    """Trainable additive delta for the role-marker rows of a frozen (usually
    tied) embedding/lm_head table.

    The full table stays requires_grad=False — sft/train.py checkpoints every
    requires_grad parameter, so unfreezing the table for 2 rows would save all
    ~77M of them and let frozen rows drift through optimizer state. Instead
    this zero-initialized (n_markers, d_model) parameter is added to the
    embedding output at marker positions (`embed`) and, with the same weights,
    to the marker columns of the logits (`head`) — arithmetically identical to
    training exactly those rows of a tied table. All arithmetic is
    branch-free on tensor content, so the per-token training loop never syncs.
    """

    def __init__(self, marker_ids: list[int], d_model: int, device=None, dtype=None):
        super().__init__()
        self.register_buffer(
            "marker_ids", torch.tensor(marker_ids, dtype=torch.long, device=device), persistent=False
        )
        self.delta = nn.Parameter(torch.zeros(len(marker_ids), d_model, device=device, dtype=dtype))

    def embed(self, h: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """h = embedding(input_ids), shape (..., d_model); adds each marker's
        delta row at that marker's positions."""
        onehot = (input_ids.unsqueeze(-1) == self.marker_ids).to(h.dtype)
        return h + onehot @ self.delta

    def head(self, logits: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """logits = lm_head(h); adds h · delta to the marker columns (the tied
        rows' logit contribution)."""
        extra = (h @ self.delta.to(h.dtype).t()).to(logits.dtype)
        return logits.index_add(logits.dim() - 1, self.marker_ids, extra)


def set_memory_injection(model: object, enabled: bool) -> bool:
    """Turn the neural-memory path on/off, reporting whether the model has one.

    A freshly-initialised memory model is NOT a plain-backbone proxy: the
    state arm's `beta_anneal_offset` is 0.0 until training sets it, so the
    gated-delta merge writes an untrained value into `ssm_state` at every
    window close. Measuring the backbone alone means disabling it, and
    measuring both ways gives the memory's ablation delta.
    """
    if not hasattr(model, "injection_enabled"):
        return False
    model.injection_enabled = enabled
    return True
