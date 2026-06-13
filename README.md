# altrux

Experiments with LLMs — specifically, building toward a model that revises its own output based on user feedback.

## Layout

```
altrux/
├── models/          # Model definitions — one Python file per model
│   └── mamba2_780m.py
├── backend/         # FastAPI inference server
├── frontend/        # React Router v7 chat UI
└── sft/             # LoRA supervised fine-tuning scripts
```

## Adding a new model

1. Create `models/my_model.py` — see `models/mamba2_780m.py` for the required interface (`MODEL_ID`, `TOKENIZER_ID`, `TARGET_LORA_MODULES`, `USER_OPEN`, `ASST_OPEN`, `Model`, `load_base`, `load_inference`)
2. Set `MODEL_NAME=my_model` in `backend/.env` and/or `sft/.env`
3. Start the backend or run sft — no other code changes needed

The frontend fetches `USER_OPEN`/`ASST_OPEN` automatically from `GET /config` at startup.

## Quickstart

```bash
# Backend
cd backend && make sync && make dev

# Frontend (separate terminal)
cd frontend && npm install && npm run dev

# Fine-tune
cd sft && make sync && make data && make train
```

See `backend/README.md`, `frontend/README.md`, and `sft/README.md` for full setup details.
