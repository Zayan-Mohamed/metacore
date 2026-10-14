"""Module 1 — assemble one state vector from the offline calibration artifacts.

Shells out to ``python -m module1.assemble`` (the same entrypoint a developer runs by
hand). Mirrors routers/module2.py: a dev-tool endpoint so the dashboard's
``/agent-state`` page can show a state assembled from the real ``data/`` artifacts
instead of the page's built-in representative values, not a live gRPC hot path.

The script prints one JSON object on stdout.

Unlike module2/module3, this endpoint needs no ML stack — ``module1.assemble`` is
standard-library + NumPy. It does need ``module1`` and ``metacore_contracts`` on the
path; PYTHONPATH is set explicitly here the same way Module 1's DVC stages and CI
lane set it, so the endpoint works even when the workspace is not ``uv sync``-ed.
When ``data/`` is absent the script falls back to the ``.synthetic/`` stand-in set
and says so in ``data_source``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/module1", tags=["module1"])

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE1_SRC = REPO_ROOT / "services" / "learned" / "module1_state_forecasting" / "src"
CONTRACTS_PY = REPO_ROOT / "packages" / "contracts" / "python"
RUN_TIMEOUT_S = 30

ISLANDS = ("eluvaitivu", "analaitivu", "nainativu", "delft")
SCENARIOS = ("normal", "cyclone", "blackout")


class Module1AssembleRequest(BaseModel):
    island: str = Field("eluvaitivu")
    scenario: str = Field("normal")
    timestamp: str | None = Field(None, description="ISO local hour, e.g. 2024-06-15T18:00")


class Module1AssembleResult(BaseModel):
    island: str
    island_name: str
    scenario: str
    scenario_id: str
    timestamp_lst: str
    out_of_distribution: bool
    degraded: bool
    schema_version: str
    embedding_dim: int
    has_embedding: bool
    node_count: int
    node_names: list[str]
    feature_names: list[str]
    node_features: list[list[float]]
    quality_mask: list[str]
    observed_fraction: float
    data_source: str
    generated_ms: float


@router.post("/assemble", response_model=Module1AssembleResult)
def assemble_state(req: Module1AssembleRequest) -> Module1AssembleResult:
    if req.island not in ISLANDS:
        raise HTTPException(status_code=422, detail=f"island must be one of {ISLANDS}")
    if req.scenario not in SCENARIOS:
        raise HTTPException(status_code=422, detail=f"scenario must be one of {SCENARIOS}")

    argv = [
        sys.executable, "-m", "module1.assemble",
        "--island", req.island,
        "--scenario", req.scenario,
        "--json",
    ]
    if req.timestamp:
        argv += ["--timestamp", req.timestamp]

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(MODULE1_SRC), str(CONTRACTS_PY), os.environ.get("PYTHONPATH", "")]
        ),
    }

    try:
        proc = subprocess.run(
            argv, cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=RUN_TIMEOUT_S, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Module 1 assembly timed out") from exc

    if proc.returncode != 0:
        detail = proc.stderr.strip()[-4000:] or "Module 1 assembly failed with no stderr output"
        raise HTTPException(status_code=500, detail=detail)

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        detail = proc.stdout.strip()[-4000:] or "Module 1 assembly produced no JSON on stdout"
        raise HTTPException(status_code=500, detail=detail) from exc

    return Module1AssembleResult(**payload)
