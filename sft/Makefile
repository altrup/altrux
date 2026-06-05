HF_CACHE := $(CURDIR)/../.cache/huggingface

.PHONY: sync data prepare train resume

sync:
	uv sync --inexact
	UV_TORCH_BACKEND=auto uv pip install torch
	MAX_JOBS=12 uv pip install causal-conv1d mamba-ssm --no-build-isolation

data:
	HF_HOME=$(HF_CACHE) \
	uv run --no-sync python prepare_data.py \
		--hf-dataset HuggingFaceH4/ultrachat_200k \
		--max-examples 20000

prepare:
	HF_HOME=$(HF_CACHE) \
	uv run --no-sync python prepare_data.py $(ARGS)

train:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	uv run --no-sync python -u train.py $(ARGS) 2>&1 | tee logs/train-$(shell date +%Y%m%d-%H%M%S).log

resume:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	uv run --no-sync python -u train.py --resume $(ARGS) 2>&1 | tee logs/train-$(shell date +%Y%m%d-%H%M%S).log
