# continual-learning — frontend

React Router v7 chat UI for the continual-learning backend.

## Features

- Streaming chat with the backend model
- Session persistence — loads existing session on page load
- Session reset
- Health indicator (polls `/health` every 5 s)
- Light / dark / system theme toggle (next-themes)

## Setup

```bash
npm install
cp .env.example .env   # edit as needed
```

### Environment variables

| Variable           | Default                 | Description                                       |
| ------------------ | ----------------------- | ------------------------------------------------- |
| `VITE_BACKEND_URL` | `http://localhost:8000` | Backend API base URL                              |
| `VITE_USER_OPEN`   | `[USER] `               | User turn opener — must match `backend/.env`      |
| `VITE_ASST_OPEN`   | `[ASSISTANT] `          | Assistant turn opener — must match `backend/.env` |

## Development

```bash
npm run dev        # dev server at http://localhost:5173
```

The backend must be running at `VITE_BACKEND_URL` and have `CORS_ORIGINS` set to include the frontend origin (see `backend/.env`).

## Quality checks

Run in this order after any change:

```bash
npm run lint
npm run typecheck
npm run format
```

## Production build

```bash
npm run build
npm run start
```

Output: `build/client/` (static assets) + `build/server/` (SSR Node app).
