"""FastAPI app entrypoint. Routes are mounted under /api to match the dashboard's
vite proxy (apps/dashboard/vite.config.ts: `/api` -> this service), so the dashboard
never needs to know gateway's host/port directly."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gateway.routers import module2, module3, module4

app = FastAPI(title="MetaCore Gateway")

# Dev-only: the dashboard is served by vite on a different origin than gateway's
# uvicorn port when run outside the docker-compose network.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(module2.router, prefix="/api")
app.include_router(module3.router, prefix="/api")
app.include_router(module4.router, prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
