"""Module 3 - the gating meta-policy: train a demo gate, step through one decision, play a storm.

Every route shells out to a script in services/learned/module3_metapolicy/scripts/, the same
scripts a developer runs by hand (see that module's README). These are dev-tool endpoints, not
the live gRPC hot path (that is gating_decision_svc, addressed separately via MODULE3_ADDR).

  POST /module3/run       scripts/demo_run.py   train a small PPO gate, score it vs the baselines
  POST /module3/generate  scripts/step_run.py   draw a situation and show the 15 gate inputs
  POST /module3/process   scripts/step_run.py   run the saved gate on that situation
  POST /module3/storm     scripts/storm_run.py  play one user-defined cyclone (Cyclone lab)

Requires the process running the gateway to have module3's deps (torch, gymnasium, numpy,
pyyaml) importable - true for `uv run uvicorn gateway.main:app` from the repo root (shared
workspace venv), not for the gateway's own slim Docker image.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/module3", tags=["module3"])

MODULE3_DIR = Path(__file__).resolve().parents[3] / "learned" / "module3_metapolicy"
STATE_DIR = MODULE3_DIR / "artifacts" / "dashboard"
RUN_TIMEOUT_S = 150
STEP_TIMEOUT_S = 30


class Module3RunRequest(BaseModel):
    seed: int = Field(0, ge=0, le=10_000)
    episode_len: int = Field(80, ge=20, le=200)
    budget_per_episode: float = Field(45, ge=1, le=200)
    train_episodes: int = Field(150, ge=10, le=300)
    eval_episodes: int = Field(12, ge=1, le=50)
    deliberation_cost: float = Field(0.05, ge=0, le=5)
    w_serve: float = Field(1.0, ge=0, le=10)
    w_viol: float = Field(2.0, ge=0, le=20)


class Module3RunResult(BaseModel):
    reward: dict[str, float]
    avg_deliberation_cost: dict[str, float]
    escalation_by_severity: dict[str, float]
    monotonic_nondecreasing: bool
    escalation_by_trigger_reason: dict[str, float]
    decisions: list[dict[str, Any]]
    decision_context: list[dict[str, Any]]
    notes: list[str] = []


class ObservationField(BaseModel):
    index: int
    name: str
    source: str
    meaning: str
    value: float


class Module3GenerateResult(BaseModel):
    step_index: int
    episode_len: int
    observation: list[ObservationField]
    raw: dict[str, Any]
    seed_used: int


class Module3ProcessResult(BaseModel):
    chosen: str
    requested: str
    budget_exhausted_fallback: bool
    action_probs: dict[str, float]
    epistemic_at_decision: float
    deliberation_cost: float
    plan: dict[str, Any]
    observation: list[ObservationField]
    raw: dict[str, Any]
    twin: dict[str, Any] | None = None


def _run_module3_script(args: list[str], timeout_s: int, output: Path) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [sys.executable, *args],
            cwd=MODULE3_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Module 3 step timed out") from exc
    if proc.returncode == 3:
        raise HTTPException(status_code=409, detail=proc.stderr.strip() or "step not ready")
    if proc.returncode != 0:
        detail = proc.stderr.strip()[-4000:] or "Module 3 step failed with no stderr output"
        raise HTTPException(status_code=500, detail=detail)
    if not output.exists():
        raise HTTPException(status_code=500, detail="Module 3 step produced no result file")
    return json.loads(output.read_text())


@router.post("/run", response_model=Module3RunResult)
def run_module3(req: Module3RunRequest) -> Module3RunResult:
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "request.json", Path(tmp) / "result.json"
        inp.write_text(json.dumps(req.model_dump()))
        payload = _run_module3_script(
            ["scripts/demo_run.py", str(inp), str(out)], RUN_TIMEOUT_S, out
        )
    # a new gate invalidates any situation generated for the previous one
    (STATE_DIR / "pending.json").unlink(missing_ok=True)
    return Module3RunResult(**payload)


@router.post("/generate", response_model=Module3GenerateResult)
def generate_situation() -> Module3GenerateResult:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "generated.json"
        payload = _run_module3_script(["scripts/step_run.py", "generate", str(out)], STEP_TIMEOUT_S, out)
    return Module3GenerateResult(**payload)


@router.post("/process", response_model=Module3ProcessResult)
def process_gate() -> Module3ProcessResult:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "processed.json"
        payload = _run_module3_script(["scripts/step_run.py", "process", str(out)], STEP_TIMEOUT_S, out)
    return Module3ProcessResult(**payload)


# ---------------------------------------------------------------- Cyclone lab
# Plays one user-defined storm through a trained gate (module3.serve.storm). Needs the trained
# checkpoints under artifacts/pipeline/final_islanded, which are not in git: without them the
# endpoint answers 409 with the reason instead of a stack trace.
STORM_TIMEOUT_S = 60


class StormParams(BaseModel):
    peak_wind_kmh: float | None = Field(None, ge=20, le=250)
    peak_frac: float | None = Field(None, ge=0.1, le=0.9)
    width_frac: float | None = Field(None, ge=0.03, le=0.40)
    trip_scale: float | None = Field(None, ge=0, le=0.6)
    comms_rate: float | None = Field(None, ge=0, le=0.6)
    diesel_loss_prob: float | None = Field(None, ge=0, le=1)
    grid_loss_prob: float | None = Field(None, ge=0, le=1)
    ood: bool | None = None


class Module3StormRequest(BaseModel):
    preset: str | None = Field(None, pattern="^(burevi_like|ditwah_like)$")
    random: bool = False
    params: StormParams = StormParams()
    seed: int = Field(0, ge=0, le=1_000_000)
    gate_seed: int = Field(0, ge=0, le=4)


@router.get("/storm/presets")
def storm_presets() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "presets.json"
        return _run_module3_script(
            ["scripts/storm_run.py", "presets", str(out)], STEP_TIMEOUT_S, out
        )


@router.post("/storm")
def run_storm(req: Module3StormRequest) -> dict[str, Any]:
    spec = {
        "preset": req.preset,
        "random": req.random,
        "params": req.params.model_dump(exclude_none=True),
        "seed": req.seed,
        "gate_seed": req.gate_seed,
    }
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "request.json", Path(tmp) / "result.json"
        inp.write_text(json.dumps(spec))
        return _run_module3_script(
            ["scripts/storm_run.py", str(inp), str(out)], STORM_TIMEOUT_S, out
        )
