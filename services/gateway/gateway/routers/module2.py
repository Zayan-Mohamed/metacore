"""Module 2 — score one situation against the trained EDL engine.

Shells out to services/learned/module2_auq_engine/infer_one.py, the same script a
developer runs by hand. Mirrors routers/module3.py: a dev-tool endpoint so the dashboard
can show real model output instead of a client-side approximation, not the live gRPC hot
path.

The script prints one JSON object on stdout (module3's scripts write a file instead —
this one is a single small payload, so a pipe is enough).

Requires the process running the gateway to have module2's deps (torch, numpy, pyyaml,
and ideally the `onnx` extra for onnxruntime) importable — true for
`uv run uvicorn gateway.main:app` from the repo root, not for the gateway's slim image.
Without onnxruntime the script falls back to the torch EDL head; the response shape is
identical either way and `backend` says which ran.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/module2", tags=["module2"])

MODULE2_DIR = Path(__file__).resolve().parents[3] / "learned" / "module2_auq_engine"
# The first call trains the baseline fixture and caches it; later calls are ~1s.
RUN_TIMEOUT_S = 300


class Module2RunRequest(BaseModel):
    novelty: float = Field(0.0, ge=0.0, le=1.0)
    observed_fraction: float = Field(1.0, ge=0.0, le=1.0)


class Module2Baselines(BaseModel):
    softmax: float
    mc_dropout: float
    edl: float


class Module2RunResult(BaseModel):
    u: float
    u_q: float
    evidence: float
    observed_fraction: float
    trigger: bool
    reason: str
    latency_ms: float
    baselines: Module2Baselines
    backend: str
    value_threshold: float


@router.post("/run", response_model=Module2RunResult)
def run_module2(req: Module2RunRequest) -> Module2RunResult:
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "infer_one.py",
                "--novelty",
                str(req.novelty),
                "--observed-fraction",
                str(req.observed_fraction),
            ],
            cwd=MODULE2_DIR,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Module 2 scoring timed out") from exc

    if proc.returncode != 0:
        detail = proc.stderr.strip()[-4000:] or "Module 2 scoring failed with no stderr output"
        raise HTTPException(status_code=500, detail=detail)

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        detail = proc.stdout.strip()[-4000:] or "Module 2 scoring produced no JSON on stdout"
        raise HTTPException(status_code=500, detail=detail) from exc

    return Module2RunResult(**payload)
