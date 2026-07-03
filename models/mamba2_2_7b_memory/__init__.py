from pathlib import Path

from .model import (
    ASST_OPEN,
    MODEL_ID,
    SPECIAL_TOKENS,
    TARGET_LORA_MODULES,
    TOKENIZER_ID,
    USER_OPEN,
    Model,
    load_base,
    load_inference,
)


def post_load(model: Model, checkpoint_path: str | Path) -> None:
    # global_step isn't stored inside state.pt itself -- it's the
    # checkpoint directory's own name (see train.py's save_checkpoint),
    # e.g. ".../epoch-3/step-171" -> 171.
    step_dir = Path(checkpoint_path).name
    if step_dir.startswith("step-"):
        global_step = int(step_dir.split("-")[1])
        model.set_beta_anneal(global_step)
        print(f"beta anneal set for step {global_step}")


__all__ = [
    "MODEL_ID",
    "TOKENIZER_ID",
    "TARGET_LORA_MODULES",
    "USER_OPEN",
    "ASST_OPEN",
    "SPECIAL_TOKENS",
    "Model",
    "load_base",
    "load_inference",
    "post_load",
]
