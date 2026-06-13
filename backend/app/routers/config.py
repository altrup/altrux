from fastapi import APIRouter

from ..model.registry import registry

router = APIRouter()


@router.get("/config")
def get_config():
    return {
        "user_open": registry.user_open,
        "asst_open": registry.asst_open,
    }
