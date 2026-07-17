HF_CACHE := $(CURDIR)/../.cache/huggingface
PYTHONPATH := $(CURDIR)/..

.PHONY: sync data data-memory prepare prepare-interference prepare-chains probe-recall train resume detect-anomaly smoke-test test

sync:
	uv sync --inexact
	UV_TORCH_BACKEND=auto uv pip install torch --reinstall
	uv pip install wheel
	MAX_JOBS=12 uv pip install mamba-ssm --no-build-isolation
	@if uv run --no-sync python -c "import torch, sys; sys.exit(0 if (torch.cuda.is_available() and torch.version.hip is None) else 1)"; then \
		echo "CUDA (non-ROCm) GPU detected -- installing causal-conv1d for the fused kernel path"; \
		MAX_JOBS=12 uv pip install causal-conv1d --no-build-isolation; \
	else \
		echo "No CUDA (non-ROCm) GPU detected -- skipping causal-conv1d (broken on this project's ROCm gfx1102 dev box, see root CLAUDE.md); models fall back to the manual per-token loop"; \
	fi

test:
	PYTHONPATH=$(PYTHONPATH) uv run --no-sync pytest tests ../models/tests

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

# Dedicated diagnostic run for localizing a non-finite-gradient crash:
# --detect-anomaly makes the run stop immediately, with a full traceback to
# the exact forward op responsible, the first time it happens -- instead of
# the normal train/resume behavior of discarding that chunk and continuing.
# Slower (extra bookkeeping every forward op) and deliberately NOT
# resilient, so this is for a short, throwaway run to pin down a real
# crash, not the long unattended one -- use `make train`/`make resume` for
# that, without this flag. Always --resume: the whole point is usually to
# reproduce a crash you already hit partway through a real run, which means
# starting from its latest checkpoint, not from scratch -- and --resume
# already no-ops into a fresh start on its own if no checkpoint exists yet
# (see train.py's main(): "no checkpoint found, starting fresh"), so this
# is never wrong to default to.
detect-anomaly:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u train.py --detect-anomaly --resume $(ARGS) 2>&1 | tee logs/train-$(shell date +%Y%m%d-%H%M%S).log

# Splice interference-recall structure (labeled random codes + delayed
# query/answer turns) into a fraction of an already-tokenized dataset --
# manufactures the beyond-SSM-capacity recall regime the neural memory
# exists for. Token-level splice; runs in minutes on CPU.
prepare-interference:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u prepare_interference.py $(ARGS)

# Episodic-chain training data (data/train.pt + data/train_memory.pt ->
# data/train_chains.pt): concatenated episodes with sleep_positions,
# fact blocks, and queries at three distances -- see
# docs/superpowers/specs/2026-07-17-episodic-chains-design.md.
prepare-chains:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u prepare_chains.py $(ARGS)

# Long-gap recall probe against the latest checkpoint: states a random code,
# runs thousands of filler tokens, queries it back, and compares the code's
# log-prob with the neural memory intact vs ablated (fresh random M at query
# time). Isolates the Titans memory's contribution from Mamba's own SSM
# state. Safe to run alongside training (second model instance, ~10 GB).
probe-recall:
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u probe_recall.py $(ARGS)

# Gradient-wiring check on a synthetic dataset, driven through train.py's
# real run_training loop with the same --batch-size/--accum-steps defaults
# as `make train` -- useful for mamba2_2_7b_memory, whose real examples are
# thousands of tokens and whose --chunk-len is capped at 8 by this
# hardware's 8GB VRAM (see smoke_test.py). Still pays for loading the real
# model. Checkpoints from the run go to a throwaway temp dir, not the real
# checkpoints/ tree. --length defaults to accum_steps * chunk_len + 1 tokens
# per example (exercises a full accumulation window, not one isolated
# chunk -- see smoke_test.py's docstring for why that matters); pass
# --length to override.
smoke-test:
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u smoke_test.py $(ARGS)

# train.py is generic across every model in models/ -- it dispatches to
# models/{MODEL_NAME}/train_hooks.py for the model-specific parts (how to
# load the model, how to run forward+backward for one example). For
# mamba2_2_7b_memory specifically:
#   MODEL_NAME=mamba2_2_7b_memory make train ARGS="--data data/train_memory.pt"
