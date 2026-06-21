HF_CACHE := $(CURDIR)/../.cache/huggingface
PYTHONPATH := $(CURDIR)/..

.PHONY: sync data data-memory prepare train resume test

sync:
	uv sync --inexact
	UV_TORCH_BACKEND=auto uv pip install torch
	MAX_JOBS=12 uv pip install causal-conv1d mamba-ssm --no-build-isolation

test:
	PYTHONPATH=$(PYTHONPATH) uv run --no-sync pytest ../models/tests

data:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python prepare_data.py \
		--hf-dataset HuggingFaceH4/ultrachat_200k \
		--max-examples 20000

# Long-context training data for mamba2_2_7b_memory: THUDM/LongAlign-10k
# (real long multi-turn conversations) mixed with RMT-team/babilong (synthetic
# needle-in-haystack recall QA, so the memory gates get genuine long-range
# recall pressure, not just long text -- see models/mamba2_2_7b_memory/README.md).
#
# --max-len here is deliberately way above any real example (LongAlign-10k's
# longest is ~65k tokens) -- this --max-len only controls what gets written to
# disk, which is nearly free (a few hundred KB per extra 100k tokens), so there
# is no reason to truncate at prep time. RAM is a *training*-time problem and
# belongs in chunked/truncated-BPTT training (Model.forward's state param is
# built for this -- see models/mamba2_2_7b_memory/README.md), not here.
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
