# Frontend

React + Vite + TypeScript web interface for the continual-learning backend.

## Setup

```bash
make install-frontend   # from repo root
# or:
cd frontend && npm install
```

## Dev server (with live reload / HMR)

```bash
make dev-frontend   # from repo root
# or:
cd frontend && npm run dev
```

Opens at `http://localhost:5173`. The Vite dev server proxies all `/generate`, `/train`, and `/checkpoint` requests to `http://localhost:8000`, so no CORS config needed — just make sure the backend is running.

## Build for production

```bash
cd frontend && npm run build   # outputs to frontend/dist/
```

The `dist/` folder can be served as static files from any web server or mounted inside FastAPI with `StaticFiles`.

## Structure

```
src/
├── App.tsx               # layout, wires generate ↔ train state
├── types.ts              # TypeScript types mirroring backend schemas
├── api/
│   ├── client.ts         # fetch/SSE helpers
│   ├── generate.ts       # /generate endpoint wrappers
│   └── train.ts          # /train + /checkpoint endpoint wrappers
├── components/
│   ├── GeneratePanel.tsx  # left column: prompt, slider, response display
│   ├── TrainPanel.tsx     # right column: phase, reward, train/save
│   └── HistoryTable.tsx   # last-N-steps table
├── hooks/
│   ├── useGenerate.ts    # SSE streaming → React state
│   └── useTrain.ts       # training mutations + history
└── __tests__/
    ├── setup.ts
    ├── GeneratePanel.test.tsx
    └── TrainPanel.test.tsx
```

## Testing

```bash
make test-frontend   # from repo root
# or:
cd frontend && npm run test
```
