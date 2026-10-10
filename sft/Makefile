HF_CACHE := $(CURDIR)/../.cache/huggingface
PYTHONPATH := $(CURDIR)/..
SHELL := /bin/bash

# uv's torch backend for `sync`. `auto` detects the local accelerator, which
# works on the ROCm dev box but NOT on every CUDA host (e.g. GH200/aarch64
# needs an explicit wheel) -- override per machine, e.g. TORCH_BACKEND=cu128.
TORCH_BACKEND ?= auto
# Parallel compile jobs for the from-source builds (mamba-ssm, causal-conv1d).
# Defaults to all cores; each nvcc/hipcc job is RAM-heavy, so cap it on a
# low-memory box (the ROCm dev box) if the compile OOMs: MAX_JOBS=12.
MAX_JOBS ?= $(shell nproc)

.PHONY: sync data data-memory warm-start prepare prepare-interference prepare-chains prepare-cram prepare-needles filter-items probe-recall consolidation-null capacity-ladder erase-probe dream-sleep adaptive-multisleep lama-ckl-upstream-check lama-ckl-upstream-smoke lama-ckl-upstream-run lama-ckl-upstream-summarize lama-ckl-split lama-ckl-run lama-ckl-smoke lama-ckl-report summarize-grid train resume detect-anomaly smoke-test test acceptance-check eligibility-check measure-knobs topic-choice gate-pilot adaptive-wake-smoke user-generator cue-sweep cue-sweep2 cue-sweep3 cue-sweep4 cue-sweep5 install-probe grid grid2 ladder

sync:
	uv sync --inexact
	UV_TORCH_BACKEND=$(TORCH_BACKEND) uv pip install torch --reinstall
	uv pip install wheel
	MAX_JOBS=$(MAX_JOBS) uv pip install mamba-ssm --no-build-isolation
	@if uv run --no-sync python -c "import torch, sys; sys.exit(0 if (torch.cuda.is_available() and torch.version.hip is None) else 1)"; then \
		echo "CUDA (non-ROCm) GPU detected -- installing causal-conv1d for the fused kernel path"; \
		MAX_JOBS=$(MAX_JOBS) uv pip install causal-conv1d --no-build-isolation; \
	else \
		echo "No CUDA (non-ROCm) GPU detected -- skipping causal-conv1d (broken on this project's ROCm gfx1102 dev box, see root CLAUDE.md); models fall back to the manual per-token loop"; \
	fi

fmt:
	uv run --no-sync ruff check --select I --fix . ../models ../scripts
	uv run --no-sync ruff format . ../models ../scripts

lint:
	uv run --no-sync ruff check --select I . ../models ../scripts
	uv run --no-sync ruff format --check . ../models ../scripts

hooks:
	git config core.hooksPath scripts/git-hooks

test:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) uv run --no-sync pytest tests ../models/tests

# --max-len far above any real conversation: ultrachat trains UNCAPPED (the
# old 1024 cap only ever lived in a stale train.pt artifact; see
# notes/discussion/DISCUSSION-20260724-next-run-plan.md 1.3) -- explicit here so the
# no-cap intent is visible rather than riding on preparation/conversations.py's default.
data:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -m preparation.conversations \
		--hf-dataset HuggingFaceH4/ultrachat_200k \
		--max-examples 20000 \
		--max-len 100000

# Long-context training data for mamba2_2_7b_memory: THUDM/LongAlign-10k
# (long-document carrier data: single-exchange episodes, one long document
# turn and its answer -- NOT multi-turn conversation) mixed with
# RMT-team/babilong (synthetic needle-in-haystack recall QA, so the memory
# gates get genuine long-range recall pressure, not just long text -- see
# models/mamba2_2_7b_memory/README.md).
#
# --max-len here is deliberately way above any real example (LongAlign-10k's
# longest is ~65k tokens) -- this --max-len only controls what gets written to
# disk, which is nearly free (a few hundred KB per extra 100k tokens), so there
# is no reason to truncate at prep time. RAM is a *training*-time problem and
# belongs in chunked/truncated-BPTT training (Model.forward's state param is
# built for this -- see models/mamba2_2_7b_memory/README.md), not here.
data-memory:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -m preparation.babilong \
		--configs 4k 8k \
		--max-per-split 100 \
		--output data/babilong_raw.jsonl
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -m preparation.conversations \
		--hf-dataset THUDM/LongAlign-10k \
		--hf-split train \
		--max-len 100000 \
		--output data/train_memory_longalign.pt
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -m preparation.conversations \
		--input data/babilong_raw.jsonl \
		--max-len 100000 \
		--output data/train_memory_babilong.pt
	PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -m preparation.merge \
		data/train_memory_longalign.pt data/train_memory_babilong.pt \
		--output data/train_memory.pt

prepare:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -m preparation.conversations $(ARGS)

train:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m training.cli $(ARGS) 2>&1 | tee logs/train-$(shell date +%Y%m%d-%H%M%S).log

resume:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m training.cli --resume $(ARGS) 2>&1 | tee logs/train-$(shell date +%Y%m%d-%H%M%S).log

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
# (see training/cli.py's main(): "no checkpoint found, starting fresh"), so this
# is never wrong to default to.
detect-anomaly:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m training.cli --detect-anomaly --resume $(ARGS) 2>&1 | tee logs/train-$(shell date +%Y%m%d-%H%M%S).log

# Splice interference-recall structure (labeled random codes + delayed
# query/answer turns) into a fraction of an already-tokenized dataset --
# manufactures the beyond-SSM-capacity recall regime the neural memory
# exists for. Token-level splice; runs in minutes on CPU.
prepare-interference:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m preparation.interference $(ARGS)

# Episodic-chain training data (data/train.pt + data/train_memory.pt ->
# data/train_chains.pt): concatenated episodes with sleep_positions,
# fact blocks, and queries at three distances -- see
# docs/superpowers/specs/2026-07-17-episodic-chains-design.md.
prepare-chains:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m preparation.chains $(ARGS)

# Ten-second cross-slice read before training: decoded windows + headline
# counts per artifact. Hand the output to a reviewer (cheap subagent on the
# box) asking "does each slice look like what it claims to be?"
sanity-sample:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m preparation.inspection $(ARGS)

# Wikipedia IMR cram blocks -> data/train_cram.pt (+ data/eval_cram.pt from
# held-out articles): alternating turns of fresh passages and one earlier
# passage's sentence re-shown with its NER entity blanked, answered by the
# same sentence completed. Gap curriculum encoded in emission order; recall
# credit on the entity span only. Streams Wikipedia and runs a NER pass, so
# this one takes a while -- see sft/README.md's data-generation section.
prepare-cram:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m preparation.cram $(ARGS)

# babilong-style needle blocks -> data/train_needles.pt (+ data/eval_needles.pt):
# one bAbI story per block, Wikipedia passages as the interference in the gap,
# same curriculum and recall weighting as the cram slice.
prepare-needles:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m preparation.needles $(ARGS)

# Three-test solvability filter over a cram/needle artifact: recall credit
# survives only on items the plain backbone answers from their source, fails
# without it, and fails with it in-stream. Two forward passes over the
# artifact, so this one wants the GPU -- and MODEL_NAME must name a plain
# backbone (mamba2_780m), which is what "M-ablated" means here.
filter-items:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m preparation.filtering $(ARGS) 2>&1 | tee logs/filter-$(shell date +%Y%m%d-%H%M%S).log

# Long-gap recall probe against the latest checkpoint: states a random code,
# runs thousands of filler tokens, queries it back, and compares the code's
# log-prob with the neural memory intact vs ablated (fresh random M at query
# time). Isolates the Titans memory's contribution from Mamba's own SSM
# state. Safe to run alongside training (second model instance, ~10 GB).
probe-recall:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m diagnostics.recall $(ARGS) 2>&1 | tee logs/probe-$(shell date +%Y%m%d-%H%M%S).log

probe-correction:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m diagnostics.correction $(ARGS) 2>&1 | tee logs/probe-correction-$(shell date +%Y%m%d-%H%M%S).log

read-diagnostic:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m diagnostics.reads $(ARGS) 2>&1 | tee logs/read-diag-$(shell date +%Y%m%d-%H%M%S).log

# Transcript-consolidation null: state N facts in a wake transcript, distil
# the in-context teacher into LoRA, then probe generation from a fresh state
# with no context. The field-default consolidation baseline that gates the
# M-replay arm -- see the module docstring. Box tool (rented CUDA hardware).
consolidation-null:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.consolidation.null $(ARGS) 2>&1 | tee logs/consolidation-null-$(shell date +%Y%m%d-%H%M%S).log

# Erase-efficacy probe: does the rank-1 state erase S(I - g c c^T) make the
# primed SSM forget the targeted fact, and only it? Pure inference; the
# physics check for the dream-distillation sleep protocol. Runs locally.
erase-probe:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.erasure.probe $(ARGS) 2>&1 | tee logs/erase-probe-$(shell date +%Y%m%d-%H%M%S).log

# Dream-distillation sleep, the CL A/B: wake on N facts, sleep with one of the
# registered arms (--arm replay|drain|counterfactual, --ce-on-dream, or
# --sft-ref / --no-sleep), then the margin/generality/locality battery. Build
# the seed's shared dream cache first (--build-dream-cache); every arm loads it
# and none regenerates. Trains a LoRA and holds a full-vocab dream logit cache
# -- box tool (rented CUDA hardware), not the local ROCm box.
dream-sleep:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.dreams.cli $(ARGS) 2>&1 | tee logs/dream-sleep-$(shell date +%Y%m%d-%H%M%S).log

adaptive-multisleep:
	@mkdir -p logs
	set -o pipefail; HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.adaptive.cli $(ARGS) 2>&1 | tee logs/adaptive-multisleep-$(shell date +%Y%m%d-%H%M%S).log

TAALM_DIR ?= $(CURDIR)/../.cache/TAALM
TAALM_COMMIT := b12f344a9dbae555c239635b1c192c555bed001b
LAMA_CKL_CACHE := $(CURDIR)/../.cache/lama_ckl

lama-ckl-upstream-check:
	test "$$(git -C "$(TAALM_DIR)" rev-parse HEAD)" = "$(TAALM_COMMIT)"
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.lama_ckl.upstream verify "$(TAALM_DIR)"

lama-ckl-upstream-smoke: lama-ckl-upstream-check
	@mkdir -p logs
	set -o pipefail; { \
		HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) uv run --no-sync python -u -m experiments.lama_ckl.upstream prepare-single-gpu-smoke "$(TAALM_DIR)"; \
		HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) uv run --no-sync python -u -m experiments.lama_ckl.upstream adapt-single-gpu "$(TAALM_DIR)" --smoke \
			| (cd "$(TAALM_DIR)" && HF_HOME="$(HF_CACHE)" PYTHONUNBUFFERED=1 bash); \
	} 2>&1 | awk '{ print "[" strftime("%H:%M:%S") "] " $$0; fflush(); }' | tee logs/lama-ckl-upstream-smoke-$(shell date +%Y%m%d-%H%M%S).log

lama-ckl-upstream-run: lama-ckl-upstream-check
	@mkdir -p logs
	set -o pipefail; HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.lama_ckl.upstream adapt-single-gpu "$(TAALM_DIR)" \
		| (cd "$(TAALM_DIR)" && HF_HOME="$(HF_CACHE)" PYTHONUNBUFFERED=1 bash) 2>&1 \
		| awk '{ print "[" strftime("%H:%M:%S") "] " $$0; fflush(); }' \
		| tee logs/lama-ckl-upstream-run-$(shell date +%Y%m%d-%H%M%S).log

lama-ckl-upstream-summarize:
	test -n "$(RESULT)"
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.lama_ckl.upstream summarize "$(RESULT)"
	mkdir -p "$(LAMA_CKL_CACHE)/upstream"
	cp "$(RESULT)" "$(LAMA_CKL_CACHE)/upstream/finetune_qlora.pkl"

lama-ckl-split:
	@mkdir -p logs
	set -o pipefail; HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.lama_ckl.split $(ARGS) 2>&1 | tee logs/lama-ckl-split-$(shell date +%Y%m%d-%H%M%S).log

lama-ckl-run:
	@mkdir -p logs
	set -o pipefail; HF_HOME=$(HF_CACHE) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.lama_ckl.runner $(ARGS) 2>&1 | tee logs/lama-ckl-run-$(shell date +%Y%m%d-%H%M%S).log

lama-ckl-smoke:
	$(MAKE) lama-ckl-run ARGS="$(ARGS) --smoke"

lama-ckl-report:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.lama_ckl.report $(ARGS)

# Warm-start SFT (DISCUSSION-20261010 sec 3.1): 1000 ultrachat conversations
# plus 1000 Dolly context rows (summarization, closed_qa,
# information_extraction) in one shuffled pool, packed 2-5 to an example behind
# <|endofconversation|> with --repeat-rate 0.3 verbatim repeats, then 800
# training/loop.py steps / 2 epochs at 1e-4 and dream_sleep's LoRA config.
# training.cli always writes checkpoints/epoch-*, so the finished tree moves to
# WARM_START_DIR; refusing a stale epoch-* tree keeps --keep-ckpts 1 from
# pruning the new run against an old one. Box tool (rented CUDA hardware).
WARM_START_CKPTS := $(CURDIR)/../models/mamba2_2_7b/checkpoints
WARM_START_DIR := $(WARM_START_CKPTS)/repeat030
warm-start:
	@mkdir -p logs
	@if compgen -G '$(WARM_START_CKPTS)/epoch-*' >/dev/null || [ -e '$(WARM_START_DIR)' ]; then \
		echo "warm-start: move $(WARM_START_CKPTS)/epoch-* and $(WARM_START_DIR) aside first"; exit 1; fi
	@set -o pipefail; { HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	   uv run --no-sync python -u -m preparation.dolly \
		--max-examples 1000 --output data/dolly_context.jsonl && \
	   HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	   uv run --no-sync python -u -m preparation.conversations \
		--hf-dataset HuggingFaceH4/ultrachat_200k --max-examples 1000 \
		--input data/dolly_context.jsonl --max-len 8192 --pack --repeat-rate 0.3 \
		--output data/warm_start.pt && \
	   HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	   uv run --no-sync python -u -m training.cli --data data/warm_start.pt \
		--max-steps 800 --epochs 2 --lr 1e-4 --chunk-len 512 \
		--lora-rank 16 --lora-alpha 32 --lora-dropout 0 --keep-ckpts 1 $(ARGS) && \
	   mkdir -p '$(WARM_START_DIR)' && mv '$(WARM_START_CKPTS)'/epoch-* '$(WARM_START_DIR)/' && \
	   ls -d '$(WARM_START_DIR)'/epoch-2/step-800; \
	 } 2>&1 | tee logs/warm-start-$(shell date +%Y%m%d-%H%M%S).log

# In-context capacity ladder: the largest fact count whose in-context control
# still clears 0.8, i.e. the N at which consolidation-null has a valid premise.
capacity-ladder:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.consolidation.capacity $(ARGS) 2>&1 | tee logs/capacity-ladder-$(shell date +%Y%m%d-%H%M%S).log

# Score the corrected single-sleep grid (logs/g2_*_s*.jsonl). Refuses to pool
# cells of a seed whose wave-1 dream or transcript hashes disagree.
summarize-grid:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -m reporting.grid $(ARGS)

dream-fidelity:
	@mkdir -p logs
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m diagnostics.dream_fidelity $(ARGS) 2>&1 | tee logs/dream-$(shell date +%Y%m%d-%H%M%S).log

# Gradient-wiring check on a synthetic dataset, driven through training/loop.py's
# real run_training loop with the same --batch-size/--accum-steps defaults
# as `make train` -- useful for mamba2_2_7b_memory, whose real examples are
# thousands of tokens and whose --chunk-len is capped at 8 by this
# hardware's 8GB VRAM (see diagnostics/training_smoke.py). Still pays for loading the real
# model. Checkpoints from the run go to a throwaway temp dir, not the real
# checkpoints/ tree. --length defaults to accum_steps * chunk_len + 1 tokens
# per example (exercises a full accumulation window, not one isolated
# chunk -- see diagnostics/training_smoke.py's docstring for why that matters); pass
# --length to override.
smoke-test:
	HF_HOME=$(HF_CACHE) HSA_ENABLE_INTERRUPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m diagnostics.training_smoke $(ARGS)

acceptance-check:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m diagnostics.acceptance $(ARGS)

eligibility-check:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m preparation.eligibility $(ARGS)

measure-knobs:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m diagnostics.knobs $(ARGS)

topic-choice:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m diagnostics.topic_choice $(ARGS)

gate-pilot:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.erasure.pilot $(ARGS)

adaptive-wake-smoke:
	HF_HOME=$(HF_CACHE) PYTHONPATH=$(PYTHONPATH) \
	uv run --no-sync python -u -m experiments.adaptive.runner $(ARGS)

user-generator:
	PYTHONPATH=$(PYTHONPATH) uv run --no-sync python -u -m providers.user_generator $(ARGS)

cue-sweep:
	bash scripts/cue_sweep.sh $(ARGS)

cue-sweep2:
	bash scripts/cue_sweep2.sh $(ARGS)

cue-sweep3:
	bash scripts/cue_sweep3.sh $(ARGS)

cue-sweep4:
	bash scripts/cue_sweep4.sh $(ARGS)

cue-sweep5:
	bash scripts/cue_sweep5.sh $(ARGS)

install-probe:
	bash scripts/install_probe.sh $(ARGS)

grid:
	bash scripts/run_grid.sh $(ARGS)

grid2:
	bash scripts/run_grid2.sh $(ARGS)

ladder:
	bash scripts/run_ladder.sh $(ARGS)

# --data is repeatable, and each slice can carry its own token share and its
# own chunk-len/batch-size/gradient-checkpointing. Quote each slice so the
# commas survive the shell:
#   make train ARGS="--data 'data/train_chains.pt,share=35' \
#                    --data 'data/train_cram.pt,share=35,chunk-len=512,grad-checkpoint=1,shuffle=0'"
# See sft/README.md's "Training on several dataset slices at once".
#
# training/loop.py is generic across every model in models/ -- it dispatches to
# models/{MODEL_NAME}/train_hooks.py for the model-specific parts (how to
# load the model, how to run forward+backward for one example). For
# mamba2_2_7b_memory specifically:
#   MODEL_NAME=mamba2_2_7b_memory make train ARGS="--data data/train_memory.pt"
