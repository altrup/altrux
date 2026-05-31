from contextlib import asynccontextmanager

from fastapi import FastAPI

from .model.loader import registry
from .routers import health, inference, training


@asynccontextmanager
async def lifespan(app: FastAPI):
    await registry.startup()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(health.router)
app.include_router(inference.router)
app.include_router(training.router)
