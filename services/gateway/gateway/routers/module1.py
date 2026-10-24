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


# ---------------------------------------------------------------- forecast (key function 2)
# Shells out to ``module1.infer.forecast``, which loads the trained ST-GNN v0.1 checkpoint. The
# checkpoint is git-ignored and produced by ``module1.train.run``; when it is missing the CLI exits
# 2 and this endpoint returns 503 with the command that makes it, rather than inventing a forecast.

WEIGHTINGS = ("equal", "hazard_heavy", "stress_heavy")


class Module1ForecastRequest(BaseModel):
    timestamp: str = Field("2025-11-28T08:00", description="ISO local hour inside 2024-2025")
    weighting: str = Field("equal")


class Module1ForecastNode(BaseModel):
    id: str
    island: str
    asset: str
    current: float
    history: list[float]
    forecast: list[float]
    persistence: list[float]
    target: list[float] | None
    components: dict[str, float]
    embedding: list[float]


class Module1ForecastResult(BaseModel):
    timestamp_lst: str
    split: str
    weighting: str
    weights: dict[str, float]
    variant: str
    horizon_h: int
    history_h: int
    tau: float
    embedding_dim: int
    has_embedding: bool
    ranking_t1: list[str]
    above_tau_t1: list[str]
    latency_ms: float
    data_source: str
    quality: str
    # M1 -> M2 contract fields, identical to /assemble's: node_features[i] is nodes[i]'s
    # 28-column state; quality_mask / observed_fraction feed M2's sensing axis.
    schema_version: str
    feature_names: list[str]
    node_features: list[list[float]]
    quality_mask: list[str]
    observed_fraction: float
    nodes: list[Module1ForecastNode]
    test_summary: dict[str, dict[str, float]] | None
    generated_ms: float


@router.post("/forecast", response_model=Module1ForecastResult)
def forecast_vulnerability(req: Module1ForecastRequest) -> Module1ForecastResult:
    if req.weighting not in WEIGHTINGS:
        raise HTTPException(status_code=422, detail=f"weighting must be one of {WEIGHTINGS}")
    argv = [
        sys.executable, "-m", "module1.infer.forecast",
        "--timestamp", req.timestamp,
        "--weighting", req.weighting,
        "--json",
    ]
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
        raise HTTPException(status_code=504, detail="Module 1 forecast timed out") from exc

    stderr = proc.stderr.strip()[-4000:]
    if proc.returncode == 2:  # no trained checkpoint
        raise HTTPException(status_code=503, detail=stderr)
    if proc.returncode == 3:  # hour outside the record or without 24 h of history
        raise HTTPException(status_code=422, detail=stderr)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=stderr or "Module 1 forecast failed")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        detail = proc.stdout.strip()[-4000:] or "Module 1 forecast produced no JSON on stdout"
        raise HTTPException(status_code=500, detail=detail) from exc
    return Module1ForecastResult(**payload)


# ---------------------------------------------------------------- M1 -> M4 topology (draft v1)
# The canonical network M4's twin compiles, and one hour's grid state for it. The topology is
# the committed pin, read as-is; the snapshot shells out to `module1.topology snapshot`, the same
# way /assemble and /forecast shell out, so the gateway carries no M1 logic of its own.

TOPOLOGY_PIN = CONTRACTS_PY / "metacore_contracts" / "schema" / "module1_topology_v1.json"


@router.get("/topology")
def get_topology() -> dict:
    if not TOPOLOGY_PIN.exists():
        raise HTTPException(status_code=503, detail="topology pin absent: run "
                            "python -m module1.topology build")
    return json.loads(TOPOLOGY_PIN.read_text())


class Module1SnapshotRequest(BaseModel):
    timestamp: str = Field("2025-11-28T19:00", description="ISO local hour inside 2024-2025")


@router.post("/snapshot")
def grid_state_snapshot(req: Module1SnapshotRequest) -> dict:
    argv = [sys.executable, "-m", "module1.topology", "snapshot", "--timestamp", req.timestamp]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(MODULE1_SRC), str(CONTRACTS_PY), os.environ.get("PYTHONPATH", "")]
        ),
    }
    try:
        proc = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=RUN_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Module 1 snapshot timed out") from exc
    if proc.returncode == 3:
        raise HTTPException(status_code=422, detail=proc.stderr.strip()[-2000:])
    if proc.returncode != 0:
        raise HTTPException(status_code=500,
                            detail=proc.stderr.strip()[-4000:] or "Module 1 snapshot failed")
    return json.loads(proc.stdout)


# ---------------------------------------------------------------- dashboard monitoring pages
# Read-only views for the Data health and Model graph pages. Each shells out to a
# Module 1 CLI or serves a committed artifact, so the gateway still carries no M1 logic.

def _run_json(argv: list[str], what: str) -> dict:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(MODULE1_SRC), str(CONTRACTS_PY), os.environ.get("PYTHONPATH", "")]
        ),
    }
    try:
        proc = subprocess.run([sys.executable, *argv], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=RUN_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail=f"Module 1 {what} timed out") from exc
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        detail = proc.stderr.strip()[-4000:] or f"Module 1 {what} produced no JSON"
        raise HTTPException(status_code=500, detail=detail) from exc


@router.get("/data-health")
def get_data_health() -> dict:
    """Every calibration gate, run now. A failing gate is reported in the body, not as a 500."""
    return _run_json(["-m", "module1.data.health", "--json"], "data health")


@router.get("/graph")
def get_graph() -> dict:
    """The forecaster's graph: nodes, edges, propagation matrix and the graph ablations."""
    return _run_json(["-m", "module1.graph", "--json"], "graph")
