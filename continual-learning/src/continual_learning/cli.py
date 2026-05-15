import argparse
from pathlib import Path
from .model import ContinualLearningModel, DEFAULT_MODEL
from .trainer import Trainer, TrainingConfig
import gradio as gr
from .ui import create_ui


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual Learning Model UI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--4bit", dest="load_in_4bit", action="store_true",
                        help="Load base model in 4-bit (requires bitsandbytes)")
    parser.add_argument("--8bit", dest="load_in_8bit", action="store_true",
                        help="Load base model in 8-bit (requires bitsandbytes)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore any existing checkpoint and start fresh")
    parser.set_defaults(resume=True)
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="Starting phase: 1 = train critic (base model frozen), "
                             "2 = train base model via critic reward signal")
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--base-lr", type=float, default=1e-6)
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory to save/load checkpoints (relative to cwd)")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Auto-save every N training steps; 0 to disable")
    parser.add_argument("--keep-checkpoints", type=int, default=1,
                        help="Number of numbered auto-save snapshots to keep; 0 to keep all")
    parser.add_argument(
        "--devices",
        default="0",
        help="Comma-separated CUDA GPU indices to use, e.g. '0' or '0,1'. "
             "Default: primary GPU only. Add '1' to include the iGPU, but only "
             "if your ROCm build was compiled for that architecture — mixed-arch "
             "systems often fail with 'invalid device function' otherwise.",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    device_ids = [int(x) for x in args.devices.split(",")]
    load_kwargs = dict(load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit,
                       device_ids=device_ids)

    checkpoint_available = False
    if args.resume:
        try:
            ContinualLearningModel._most_recent_snapshot(args.checkpoint_dir)
            checkpoint_available = True
        except (FileNotFoundError, ValueError):
            pass

    if checkpoint_available:
        print(f"Resuming from {args.checkpoint_dir}/ …")
        model = ContinualLearningModel.load_checkpoint(args.checkpoint_dir, **load_kwargs)
    else:
        if args.resume:
            print(f"No checkpoint found in {args.checkpoint_dir}/, starting fresh.")
        print(f"Loading {args.model} …")
        model = ContinualLearningModel(model_name=args.model, **load_kwargs)

    model.eval()

    config = TrainingConfig(
        critic_lr=args.critic_lr,
        base_model_lr=args.base_lr,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        keep_checkpoints=args.keep_checkpoints,
    )
    trainer = Trainer(model, config)

    ui = create_ui(model, trainer, initial_phase=args.phase)
    ui.launch(server_port=args.port, share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
