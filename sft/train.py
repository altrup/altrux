import argparse
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from lora import apply_lora, load_lora, save_lora

CKPT_DIR = Path("checkpoints")
MODEL_ID = "state-spaces/mamba2-780m"
TARGET_MODULES = ["in_proj", "out_proj"]


def latest_checkpoint() -> Path | None:
    if not CKPT_DIR.exists():
        return None
    steps = sorted(
        int(p.name.split("-")[1])
        for p in CKPT_DIR.iterdir()
        if p.is_dir() and p.name.startswith("step-")
    )
    return CKPT_DIR / f"step-{steps[-1]}" if steps else None


def rotate_checkpoints(keep: int) -> None:
    if not CKPT_DIR.exists():
        return
    steps = sorted(
        int(p.name.split("-")[1])
        for p in CKPT_DIR.iterdir()
        if p.is_dir() and p.name.startswith("step-")
    )
    for step in steps[:-keep]:
        shutil.rmtree(CKPT_DIR / f"step-{step}")


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> Path:
    path = CKPT_DIR / f"step-{step}"
    save_lora(model, path)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for mamba2-780m")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--model", default=MODEL_ID, help="Model ID or local path")
    parser.add_argument("--data", default="data/train.pt", help="Tokenized dataset from prepare_data.py")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accum-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--ckpt-every", type=int, default=100, help="Save checkpoint every N optimizer steps")
    parser.add_argument("--keep-ckpts", type=int, default=3, help="Number of checkpoints to retain")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading {args.model} ...")
    model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.float32)
    model = apply_lora(model, TARGET_MODULES, args.lora_rank, args.lora_alpha, args.lora_dropout)

    for param in model.parameters():
        param.requires_grad_(False)
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
            lora_params.append(param)

    print(f"lora trainable params: {sum(p.numel() for p in lora_params):,}")
    model = model.to(device)

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    start_step = 0
    if args.resume:
        ckpt = latest_checkpoint()
        if ckpt is not None:
            print(f"resuming from {ckpt}")
            load_lora(model, ckpt)
            opt_path = ckpt / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(
                    torch.load(opt_path, map_location=device, weights_only=True)
                )
            start_step = int(ckpt.name.split("-")[1])
            print(f"resumed at step {start_step}")
        else:
            print("no checkpoint found, starting fresh")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    all_ids: list[torch.Tensor] = data["ids"]
    all_masks: list[torch.Tensor] = data["masks"]
    n = len(all_ids)
    print(f"training on {n} examples × {args.epochs} epoch(s)")

    global_step = start_step
    model.train()
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        order = torch.randperm(n).tolist()
        accum_count = 0

        for i, idx in enumerate(order):
            ids = all_ids[idx].to(device)
            mask = all_masks[idx].to(device)

            if mask.sum() == 0:
                continue

            input_ids = ids[:-1].unsqueeze(0)
            target_ids = ids[1:].unsqueeze(0)
            loss_mask = mask[1:].float()

            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                reduction="none",
            )
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            (loss / args.accum_steps).backward()
            accum_count += 1

            is_last = i == len(order) - 1
            if accum_count == args.accum_steps or is_last:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                print(f"epoch {epoch + 1}  step {global_step:>6}  loss {loss.item():.4f}")

                if global_step % args.ckpt_every == 0:
                    path = save_checkpoint(model, optimizer, global_step)
                    rotate_checkpoints(args.keep_ckpts)
                    print(f"  saved {path}")

    path = save_checkpoint(model, optimizer, global_step)
    rotate_checkpoints(args.keep_ckpts)
    print(f"done. final checkpoint: {path}")


if __name__ == "__main__":
    main()
