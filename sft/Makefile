HF_CACHE := $(CURDIR)/../.cache/huggingface
PYTHONPATH := $(CURDIR)/..

.PHONY: sync data data-memory prepare train resume preflight smoke-test test

sync:
	uv sync --inexact
	UV_TORCH_BACKEND=auto uv pip install torch --reinstall
	MAX_JOBS=12 uv pip install mamba-ssm --no-build-isolation

test:
	PYTHONPATH=$(PYTHONPATH) uv run --no-sync pytest tests ../models/tests

data:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python prepare_data.py \
		--hf-dataset HuggingFaceH4/ultrachat_200k \
		--max-examples 20000

# Long-context training data for mamba2_780m_memory: THUDM/LongAlign-10k
# (real long multi-turn conversations) mixed with RMT-team/babilong (synthetic
# needle-in-haystack recall QA, so the memory gates get genuine long-range
# recall pressure, not just long text -- see models/mamba2_780m_memory/README.md).
#
# --max-len here is deliberately way above any real example (LongAlign-10k's
# longest is ~65k tokens) -- this --max-len only controls what gets written to
# disk, which is nearly free (a few hundred KB per extra 100k tokens), so there
# is no reason to truncate at prep time. RAM is a *training*-time problem and
# belongs in chunked/truncated-BPTT training (Model.forward's state param is
# built for this -- see models/mamba2_780m_memory/README.md), not here.
data-memory:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python prepare_babilong.py \
		--configs 4k 8k \
		--max-per-split 100 \
		--output data/babilong_raw.jsonl
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python prepare_data.py \
		--hf-dataset THUDM/LongAlign-10k \
		--hf-split train \
		--max-len 100000 \
		--output data/train_memory_longalign.pt
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python prepare_data.py \
		--input data/babilong_raw.jsonl \
		--max-len 100000 \
		--output data/train_memory_babilong.pt
	PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python merge_data.py \
		data/train_memory_longalign.pt data/train_memory_babilong.pt \
		--output data/train_memory.pt

prepare:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python prepare_data.py $(ARGS)

train:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u train.py $(ARGS) 2>&1 | tee logs/train-$(shell date +%Y%m%d-%H%M%S).log

resume:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u train.py --resume $(ARGS) 2>&1 | tee logs/train-$(shell date +%Y%m%d-%H%M%S).log

# Loads the real model and data and runs the preflight gradient check, then
# exits before the full training loop -- a quick sanity check (still pays for
# loading the real model/checkpoint, but skips everything after that) before
# committing to a real run. Same env/flags as `train`, so pass --data,
# --chunk-len, etc. through ARGS the same way.
preflight:
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u train.py --preflight-only $(ARGS)

# Same gradient-wiring check as `preflight`, but on a short synthetic
# sequence instead of the real dataset's first valid example -- useful for
# mamba2_780m_memory, whose real examples are thousands of tokens and whose
# --chunk-len is capped at 8 by this hardware's 8GB VRAM (see
# smoke_test.py). Still pays for loading the real model. --length defaults
# to 5x the resolved chunk_len (exercises several consecutive chunks, not
# one isolated chunk -- see smoke_test.py's docstring for why that matters);
# pass --length to override.
smoke-test:
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u smoke_test.py $(ARGS)

# train.py is generic across every model in models/ -- it dispatches to
# models/{MODEL_NAME}/train_hooks.py for the model-specific parts (how to
# load the model, how to run forward+backward for one example). For
# mamba2_780m_memory specifically:
#   MODEL_NAME=mamba2_780m_memory make train ARGS="--data data/train_memory.pt"
