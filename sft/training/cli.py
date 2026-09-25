"""Experiment: shared

Training command-line setup and resume orchestration."""

import argparse
import importlib
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
from dotenv import load_dotenv

from training import checkpoints, loop
from training.datasets import (
    check_shares,
    datasets_fingerprint,
    group_specs,
    load_datasets,
    parse_data_spec,
)


warnings.filterwarnings("ignore", message=r".*_check_is_size.*", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).parents[2]))


@dataclass(frozen=True)
class ModelRuntime:
    """The model constants and training hooks selected for a CLI run."""

    model_id: str
    hooks: ModuleType


def default_model_name() -> str:
    load_dotenv()
    return os.getenv("MODEL_NAME", "mamba2_780m")


def load_model_runtime(model_name: str) -> ModelRuntime:
    model = importlib.import_module(f"models.{model_name}")
    hooks = importlib.import_module(f"models.{model_name}.train_hooks")
    return ModelRuntime(model.MODEL_ID, hooks)


def main(ckpt_dir: Path, model_name: str) -> None:
    parser = argparse.ArgumentParser(description=f"SFT for {model_name}")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument(
        "--model", default=None, help="Model ID (informational; actual ID comes from models/ file)"
    )
    parser.add_argument(
        "--data",
        action="append",
        default=None,
        help="Tokenized dataset from preparation/conversations.py (default: data/train.pt). Repeat to train on several slices at once; each may carry comma-separated per-slice overrides, e.g. --data 'data/train_cram.pt,share=35,chunk-len=512,batch-size=6,grad-checkpoint=1,shuffle=0'. share= is that slice's requested fraction of trained tokens (any units -- shares are normalised; give it for every slice or none, in which case each slice's own token count is used); chunk-len/batch-size/grad-checkpoint override --chunk-len/--batch-size/off for this slice only; shuffle=0 consumes the artifact in the order it was written, which is how a generator-side curriculum survives training.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=32,
        help="Linearly ramp the learning rate from 0 to --lr over this many optimizer steps, then hold at --lr -- 0 to disable. A pure function of global_step, so it resumes correctly with no extra checkpoint state.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many optimizer steps and save the final checkpoint (default: run to --epochs). Counts global_step, so a resumed run finishes the same budget rather than taking this many more.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="Skip examples longer than this (default: no limit)",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=None,
        help="Tokens per forward/backward chunk -- defaults to the model's own DEFAULT_CHUNK_LEN",
    )
    parser.add_argument(
        "--memory-window",
        type=int,
        default=None,
        help="Tokens per memory-subsystem write, for models that define set_memory_window (currently mamba2_2_7b_memory only; no-op otherwise) -- defaults to the model's own DEFAULT_MEMORY_WINDOW (1, i.e. a write every token, unless overridden). Must evenly divide --chunk-len. See docs/superpowers/specs/2026-07-02-chunked-memory-injection-design.md.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
        help="Number of examples to train in parallel (slot-based batching)",
    )
    parser.add_argument(
        "--accum-tokens",
        type=int,
        default=256,
        help="Target real tokens per slot to accumulate before each optimizer step -- converted internally to a chunk count (accum_tokens / chunk_len), so it means the same amount of real training regardless of --chunk-len. Total tokens per optimizer step end up ~accum_tokens * batch_size.",
    )
    parser.add_argument(
        "--ckpt-every-tokens",
        type=int,
        default=5000,
        help="Save checkpoint every N tokens of training",
    )
    parser.add_argument(
        "--keep-ckpts", type=int, default=50, help="Number of checkpoints to retain"
    )
    parser.add_argument(
        "--keep-full-state",
        type=int,
        default=2,
        help="Number of most-recent checkpoints to also save full internal model state for (mem_state.pt) -- lets resume continue mid-example slots exactly instead of restarting them from the beginning. 0 to disable. Only applies to models whose train_hooks define init_state (e.g. mamba2_2_7b_memory); no-op otherwise (falls back to always restarting mid-example slots).",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--eos-weight",
        type=float,
        default=5.0,
        help="Loss weight for EOS tokens (>1 to emphasise stopping)",
    )
    parser.add_argument(
        "--recall-weight",
        type=float,
        default=8.0,
        help="Loss weight multiplier for tokens marked True in the dataset's optional recall_masks tensors (preparation/chains.py/preparation/interference.py mark their spliced query-answer tokens) -- amplifies the recall training signal, which is otherwise a tiny fraction (~0.1%%) of all tokens. 1.0 reproduces the unweighted objective. No-op on datasets without recall_masks.",
    )
    parser.add_argument(
        "--recall-ramp-start",
        type=float,
        default=1.0,
        help="Recall-mask multiplier at step 0, ramping to --recall-weight over --recall-ramp-steps. Equal to --recall-weight to disable the ramp.",
    )
    parser.add_argument(
        "--recall-ramp-steps",
        type=int,
        default=32,
        help="Optimizer steps over which the recall multiplier ramps from --recall-ramp-start to --recall-weight, then holds. Keep this aligned with the model's beta-anneal window (mamba2_2_7b_memory's BETA_BIAS_ANNEAL_STEPS, 32): the anneal window is where the gradient decides whether the memory path is useful or gets suppressed, and a full-strength recall multiplier landing in it amplifies the loss spike rather than the signal. 0 disables the ramp (constant --recall-weight from step 0).",
    )
    parser.add_argument(
        "--recall-ramp-shape",
        choices=("linear", "geometric"),
        default="linear",
        help="Interpolation between --recall-ramp-start and --recall-weight: linear in the multiplier, or linear in its log (geometric), which spends more of the window near the low end.",
    )
    parser.add_argument(
        "--grad-ckpt-block",
        type=int,
        default=None,
        help="Checkpoint block size in tokens for slices with grad-checkpoint=1 (default: the model's own, currently 64). The backward pass recomputes one block's live graph at a time, so block must stay under what the card can hold as a live graph -- this box's ~52-token ceiling means local grad-checkpointed runs need 48 or less, while the default is sized for a rented card. Snapped down to a multiple of --memory-window by the model.",
    )
    parser.add_argument(
        "--mix-segment-tokens",
        type=int,
        default=1_000_000,
        help="Tokens one config group trains for before the mix hands over to whichever group is furthest behind its share. Only applies when --data slices disagree on chunk-len/batch-size/grad-checkpoint (slices that agree share a batch and interleave example-by-example instead); with a single config the budget is unbounded and the loop is the single-dataset one. Smaller mixes more finely but pays a slot drain per handover.",
    )
    parser.add_argument(
        "--head-weight",
        type=float,
        default=4.0,
        help="Loss weight multiplier at the first token after each backbone reset (example start, and each sleep for datasets with sleep_positions), decaying linearly to 1.0 over --head-tokens -- emphasises the empty-state regime, which is otherwise underweighted because most tokens sit deep inside long examples. 1.0 reproduces the unweighted objective.",
    )
    parser.add_argument(
        "--head-tokens",
        type=int,
        default=1024,
        help="Length of the --head-weight linear decay ramp, in tokens from each backbone reset",
    )
    parser.add_argument(
        "--freeze-lora",
        action="store_true",
        help="Freeze the parametric LoRA weights and optimize only the memory subsystem (front_end + injection modules). Isolates whether the memory can carry recall on its own when the parametric path can no longer re-absorb the niche. Checkpoints stay complete (LoRA held at its resumed values).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for everything not covered by the per-epoch data-shuffle seed (see run_training) -- LoRA init/dropout, and for models with per-sequence random state (e.g. mamba2_2_7b_memory's neural-memory init/reset) -- fixed by default so a run (or a crash) is reproducible; pass a different value to sample a different random init.",
    )
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Enable torch.autograd.set_detect_anomaly -- when a chunk's gradient comes back non-finite (the run's existing per-chunk check, see run_training), instead of just discarding it and continuing, autograd raises immediately with a traceback pointing at the exact forward op responsible, and the run stops there. Diagnostic only: real, not-small overhead (extra bookkeeping on every op during forward), and turns the normally-recoverable non-finite-gradient path into a hard stop -- use a dedicated short run to localize a real crash, not the long unattended one. See `make detect-anomaly`.",
    )
    args = parser.parse_args()
    runtime = load_model_runtime(model_name)
    model_id = runtime.model_id
    hooks = runtime.hooks
    args.model = args.model or model_id
    args.max_len = args.max_len if args.max_len is not None else math.inf
    args.data = args.data or ["data/train.pt"]
    if args.recall_ramp_shape == "geometric" and args.recall_ramp_start <= 0:
        parser.error("--recall-ramp-start must be > 0 for a geometric ramp")
    try:
        specs = [parse_data_spec(text, args.chunk_len, args.batch_size) for text in args.data]
        check_shares(specs)
    except ValueError as e:
        parser.error(str(e))
    for spec in specs:
        if spec.chunk_len is None:
            spec.chunk_len = hooks.DEFAULT_CHUNK_LEN
    torch.manual_seed(args.seed)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print(
            "--detect-anomaly enabled: forward pass will be slower, and the run will stop with a full traceback the first time a chunk's gradient is non-finite (instead of discarding that chunk and continuing)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(
            f"device: {device} -- {torch.cuda.get_device_name(device)} (index {torch.cuda.current_device()})"
        )
        print(
            f"  VRAM total:  {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB"
        )
        print(f"  VRAM free:   {torch.cuda.mem_get_info(device)[0] / 1024**3:.1f} GB")
    else:
        print(f"device: {device} (no CUDA/ROCm device found)")

    print(f"loading {args.model} ...")
    model, trainable_params = hooks.setup_training(
        device, args.lora_rank, args.lora_alpha, args.lora_dropout
    )

    if args.freeze_lora:
        # Freeze the parametric (LoRA) path and train only the memory subsystem
        # (front_end projections + injection modules): removes the parametric
        # re-absorption route so the memory alone must carry cross-sleep recall.
        # requires_grad stays True on everything so checkpoints remain complete
        # (save_checkpoint keys off requires_grad); only the optimizer's param
        # set shrinks, so LoRA values are held fixed at the resumed checkpoint.
        name_by_id = {id(p): n for n, p in model.named_parameters()}
        trainable_params = [
            p for p in trainable_params if "lora" not in name_by_id.get(id(p), "").lower()
        ]
        print(
            f"--freeze-lora: optimizing {len(trainable_params)} memory params "
            f"(front_end + injections); LoRA held fixed"
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    start_step = 0
    start_epoch = 0
    start_slot_states = None
    start_next_ptr = 0
    start_total_tokens = 0.0
    start_last_ckpt_tokens = 0.0
    start_full_state = None
    start_dataset_fingerprint = None
    start_group_idx = 0
    start_group_ptrs = None
    start_group_tokens = None
    groups = group_specs(specs)
    if args.resume:
        ckpt = checkpoints.latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            print(f"resuming from {ckpt}")
            checkpoints.load_checkpoint(model, ckpt)
            opt_path = ckpt / "optimizer.pt"
            if opt_path.exists():
                # A checkpoint saved under --freeze-lora carries AdamW state
                # over just the memory params; one saved under full training
                # covers the whole trainable set. Load whenever the saved
                # state matches the current optimizer's param count -- a
                # mismatch (e.g. full-training state resumed with
                # --freeze-lora) falls back to a fresh optimizer.
                saved = torch.load(opt_path, map_location=device, weights_only=True)
                n_current = sum(len(g["params"]) for g in optimizer.param_groups)
                n_saved = sum(len(g["params"]) for g in saved["param_groups"])
                if n_current == n_saved:
                    optimizer.load_state_dict(saved)
                else:
                    print(
                        f"starting a fresh optimizer (saved state covers {n_saved} params, current set has {n_current})"
                    )
            start_step = int(ckpt.name.split("-")[1])
            state_path = ckpt / "state.pt"
            state_batch_size = None
            if state_path.exists():
                state = torch.load(state_path, weights_only=True)
                start_epoch = state["epoch"]
                start_slot_states = state.get("slot_states")
                start_next_ptr = state.get("next_ptr", 0)
                start_dataset_fingerprint = state.get("dataset_fingerprint")
                start_total_tokens = state.get("total_tokens", 0.0)
                start_last_ckpt_tokens = state.get("last_ckpt_tokens", 0.0)
                state_batch_size = state.get("state_batch_size")
                start_group_idx = state.get("group_idx", 0) or 0
                start_group_ptrs = state.get("group_ptrs")
                start_group_tokens = state.get("group_tokens")
                # Legacy checkpoint format (single-example, no slot_states):
                # map old example_idx/chunk_pos to a single-slot slot_states.
                if start_slot_states is None:
                    example_idx = state.get("example_idx", 0)
                    chunk_pos = state.get("chunk_pos", 0)
                    start_slot_states = [(example_idx, chunk_pos)]
                    start_next_ptr = example_idx + (0 if chunk_pos > 0 else 1)
            # mem_state.pt (the saved full internal state -- see
            # save_checkpoint/rotate_full_state) only exists on recent
            # checkpoints and only when its batch size still matches this
            # run's --batch-size; missing or mismatched just means any
            # mid-example slot restarts from the beginning instead of
            # continuing exactly, not a broken resume. Not loaded here --
            # just the path is handed to run_training, which loads it
            # itself right before use (see run_training's docstring for
            # why: main()'s own frame outlives the entire training run, so
            # a local variable here bound to the loaded tensors would keep
            # them VRAM-resident for the whole run).
            mem_state_path = ckpt / "mem_state.pt"
            if mem_state_path.exists():
                resume_batch_size = groups[min(start_group_idx, len(groups) - 1)][0].batch_size
                if state_batch_size == resume_batch_size:
                    start_full_state = mem_state_path
                else:
                    print(
                        f"mem_state.pt batch size ({state_batch_size}) doesn't match "
                        f"the resumed slice's batch size ({resume_batch_size}) -- ignoring it, restarting mid-example slots from the beginning"
                    )
            print(
                f"resumed at step {start_step}, epoch {start_epoch + 1}, next_ptr {start_next_ptr}"
            )
        else:
            print("no checkpoint found, starting fresh")

    print(f"loading {len(specs)} dataset slice(s) ...")
    train_ids, train_masks, train_recall, train_sleeps = load_datasets(specs)
    if args.recall_weight != 1.0 and all(r is None for r in train_recall):
        print(
            f"warning: --recall-weight {args.recall_weight} given but no slice has recall_masks -- it will have no effect"
        )
    if (
        any(s is not None and len(s) for s in train_sleeps)
        and getattr(hooks, "sleep_slot", None) is None
    ):
        print(
            f"warning: a slice has sleep_positions but {model_name}'s train_hooks defines no sleep_slot -- sleeps will be ignored"
        )

    n = len(train_ids)
    print(f"train: {n}  epochs: {args.epochs}")
    for group in groups:
        cfg = group[0]
        print(
            f"  config group: {', '.join(s.path for s in group)}  chunk_len {cfg.chunk_len}  "
            f"batch {cfg.batch_size}  grad_ckpt {'on' if cfg.grad_checkpoint else 'off'}"
        )

    if args.resume and start_slot_states is not None:
        current_fp = datasets_fingerprint(specs)
        discard = False
        if start_dataset_fingerprint is None:
            # Older checkpoint format, saved before dataset_fingerprint
            # existed -- unlike an outright mismatch below, this is
            # ambiguous (could be the same dataset that was always in use,
            # or a swap that just happens to predate fingerprinting), so
            # ask rather than silently guessing either way.
            print(
                f"resume: checkpoint has no dataset fingerprint (older format) -- current --data is {current_fp}"
            )
            try:
                answer = (
                    input("Is this the same dataset the checkpoint was trained on? [Y/n] ")
                    .strip()
                    .lower()
                )
            except EOFError:
                answer = ""
                print(
                    "(no input available -- defaulting to 'n': safer to restart slots than risk reindexing into the wrong dataset)"
                )
            discard = answer.startswith("n")
        elif start_dataset_fingerprint != current_fp:
            print(f"resume: dataset changed ({start_dataset_fingerprint} vs {current_fp})")
            discard = True

        if discard:
            print(
                "resume: discarding slot_states/next_ptr from the checkpoint so slots start "
                "fresh against this dataset instead of reindexing into it with stale positions"
            )
            start_slot_states = None
            start_next_ptr = 0
            start_full_state = None
            start_group_idx = 0
            start_group_ptrs = None
            start_group_tokens = None

    loop.run_training(
        ckpt_dir,
        model_name,
        hooks,
        model,
        optimizer,
        trainable_params,
        train_ids,
        train_masks,
        train_recall,
        train_sleeps,
        device,
        args,
        start_epoch,
        start_slot_states,
        start_next_ptr,
        start_step,
        start_total_tokens,
        start_last_ckpt_tokens,
        start_full_state,
        specs=specs,
        start_group_idx=start_group_idx,
        start_group_ptrs=start_group_ptrs,
        start_group_tokens=start_group_tokens,
    )


if __name__ == "__main__":
    selected_model = default_model_name()
    main(Path(__file__).parents[2] / "models" / selected_model / "checkpoints", selected_model)
