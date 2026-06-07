import json

from fastapi import APIRouter, HTTPException

from ..config import get_revise_data_path
from ..model.registry import registry
from ..schemas import ReviseRequest, ReviseResponse

router = APIRouter()


def _build_revise_line(base_messages: list[dict], suggestions: list[tuple[int, str]]) -> bytes:
    messages = [m.copy() for m in base_messages]
    tags_by_turn: dict[int, list[str]] = {}
    for at_turn, tag in suggestions:
        tags_by_turn.setdefault(at_turn, []).append(tag)
    for at_turn, tags in tags_by_turn.items():
        msg = messages[at_turn]
        msg["content"] = " ".join(tags) + " " + msg["content"]
        msg.pop("train", None)
    return (json.dumps({"messages": messages}) + "\n").encode()


@router.post("/session/revise", response_model=ReviseResponse)
async def session_revise(req: ReviseRequest) -> ReviseResponse:
    async with registry.lock:
        msgs = registry.messages
        if not msgs:
            raise HTTPException(status_code=400, detail="No messages in session")

        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        if len(assistant_msgs) < 1:
            raise HTTPException(status_code=400, detail="No assistant messages in session")

        base_messages = [
            ({"role": m["role"], "content": m["content"], "train": False}
             if m["role"] == "assistant"
             else {"role": m["role"], "content": m["content"]})
            for m in msgs
        ]

        at_turn = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        tag = f"<revise back={req.n}>{req.revision}</revise weight=0.5>"
        registry.revise_suggestions.append((at_turn, tag))
        line = _build_revise_line(base_messages, registry.revise_suggestions)

        path = get_revise_data_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        if registry.revise_entry_offset is None:
            with path.open("ab") as f:
                registry.revise_entry_offset = f.tell()
                f.write(line)
        else:
            with path.open("r+b") as f:
                f.seek(registry.revise_entry_offset)
                f.truncate()
                f.write(line)

    return ReviseResponse(ok=True)
