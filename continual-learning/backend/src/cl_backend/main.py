import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cl_backend.config import ServerConfig
from cl_backend.model import ContinualLearningModel
from cl_backend.trainer import Trainer, TrainingConfig
from cl_backend.api.generate import router as generate_router
from cl_backend.api.train import router as train_router
from cl_backend.api.checkpoint import router as checkpoint_router


def _load_model(cfg: ServerConfig) -> ContinualLearningModel:
    load_in_4bit = cfg.quantize == "4bit"
    load_in_8bit = cfg.quantize == "8bit"
    checkpoint_dir = Path(cfg.checkpoint_dir)
    device_ids = cfg.device_ids()

    # Restrict visible GPUs before any CUDA init so accelerate cannot spill onto
    # excluded devices even if _build_max_memory sets their budget to 0.
    if device_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)

    if not cfg.no_resume and checkpoint_dir.exists():
        try:
            print(f"Resuming from checkpoint in {checkpoint_dir} …", flush=True)
            return ContinualLearningModel.load_checkpoint(
                checkpoint_dir,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                device_ids=device_ids,
            )
        except FileNotFoundError:
            print("No checkpoint found, starting fresh.", flush=True)

    return ContinualLearningModel(
        model_name=cfg.model,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device_ids=device_ids,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = ServerConfig()
    model = _load_model(cfg)
    model.eval()

    training_cfg = TrainingConfig(
        critic_lr=cfg.critic_lr,
        base_model_lr=cfg.base_lr,
        checkpoint_dir=cfg.checkpoint_dir,
        save_every=cfg.save_every,
        keep_checkpoints=cfg.keep_checkpoints,
    )
    trainer = Trainer(model, training_cfg)

    app.state.model = model
    app.state.trainer = trainer
    app.state.train_lock = asyncio.Lock()

    yield


app = FastAPI(title="Continual Learning API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(generate_router)
app.include_router(train_router)
app.include_router(checkpoint_router)


def start() -> None:
    import uvicorn
    cfg = ServerConfig()
    uvicorn.run("cl_backend.main:app", host="0.0.0.0", port=cfg.port, reload=False)
