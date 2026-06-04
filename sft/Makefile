.PHONY: sync data prepare train resume

sync:
	uv sync --inexact
	UV_TORCH_BACKEND=auto UV_CACHE_DIR=.cache/uv uv pip install torch
	MAX_JOBS=12 UV_CACHE_DIR=.cache/uv uv pip install causal-conv1d mamba-ssm --no-build-isolation

data:
	HF_HOME=.cache/huggingface \
	uv run --no-sync python prepare_data.py \
		--hf-dataset HuggingFaceH4/ultrachat_200k \
		--max-examples 20000

prepare:
	HF_HOME=.cache/huggingface \
	uv run --no-sync python prepare_data.py $(ARGS)

train:
	@mkdir -p logs
	HF_HOME=.cache/huggingface HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	uv run --no-sync python -u train.py 2>&1 | tee logs/train-$(shell date +%Y%m%d-%H%M%S).log

resume:
	@mkdir -p logs
	HF_HOME=.cache/huggingface HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	uv run --no-sync python -u train.py --resume 2>&1 | tee logs/train-$(shell date +%Y%m%d-%H%M%S).log
