"""Module 3 — trigger an on-demand train+eval run of the gating meta-policy.

Shells out to services/learned/module3_metapolicy/run_demo.py, the same script a
developer runs by hand (see that module's README). This is a dev-tool endpoint for
the "training only" starter, not the live gRPC hot path (that's gating-decision_svc,
addressed separately via MODULE3_ADDR) — it exists so the dashboard can drive a real
run instead of only showing the committed sample fixture.

Requires the process running the gateway to have module3's deps (torch, gymnasium,
numpy, pyyaml) importable — true for `uv run uvicorn gateway.main:app` from the repo
root (shared workspace venv), not for the gateway's own slim Docker image.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/module3", tags=["module3"])

MODULE3_DIR = Path(__file__).resolve().parents[3] / "learned" / "module3_metapolicy"
BASE_CONFIG_PATH = MODULE3_DIR / "config.yaml"
STATE_DIR = MODULE3_DIR / "artifacts" / "dashboard"
RUN_TIMEOUT_S = 90
STEP_TIMEOUT_S = 30


class Module3RunRequest(BaseModel):
    seed: int = Field(0, ge=0, le=10_000)
    episode_len: int = Field(80, ge=20, le=200)
    budget_per_episode: float = Field(45, ge=1, le=200)
    train_episodes: int = Field(150, ge=10, le=300)
    eval_episodes: int = Field(12, ge=1, le=50)
    deliberation_cost: float = Field(0.10, ge=0, le=5)
    benefit_scale: float = Field(4.0, ge=0, le=20)
    sensing_escalation_penalty: float = Field(-1.2, ge=-10, le=0)


class Module3RunResult(BaseModel):
    reward: dict[str, float]
    avg_deliberation_cost: dict[str, float]
    escalation_by_severity: dict[str, float]
    monotonic_nondecreasing: bool
    escalation_by_trigger_reason: dict[str, float]
    decisions: list[dict[str, Any]]
    decision_context: list[dict[str, Any]]


def _build_config(req: Module3RunRequest) -> dict:
    base = yaml.safe_load(BASE_CONFIG_PATH.read_text())
    return {
        **base,
        "seed": req.seed,
        "env": {
            **base["env"],
            "episode_len": req.episode_len,
            "budget_per_episode": req.budget_per_episode,
        },
        "reward": {
            **base["reward"],
            "deliberation_cost": req.deliberation_cost,
            "benefit_scale": req.benefit_scale,
            "sensing_escalation_penalty": req.sensing_escalation_penalty,
        },
        "train": {**base["train"], "episodes": req.train_episodes},
        "eval": {**base["eval"], "n_episodes": req.eval_episodes},
    }


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
    cfg = _build_config(req)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_config = Path(tmp) / "config.run.yaml"
        tmp_output = Path(tmp) / "result.json"
        tmp_config.write_text(yaml.safe_dump(cfg))
        policy_path = STATE_DIR / "policy.pt"
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "run_demo.py",
                    str(tmp_config),
                    str(tmp_output),
                    str(policy_path),
                ],
                cwd=MODULE3_DIR,
                capture_output=True,
                text=True,
                timeout=RUN_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="Module 3 run timed out") from exc

        if proc.returncode != 0:
            detail = proc.stderr.strip()[-4000:] or "Module 3 run failed with no stderr output"
            raise HTTPException(status_code=500, detail=detail)

        if not tmp_output.exists():
            raise HTTPException(status_code=500, detail="Module 3 run produced no result file")

        payload = json.loads(tmp_output.read_text())

    (STATE_DIR / "last_config.yaml").write_text(yaml.safe_dump(cfg))
    pending = STATE_DIR / "pending.json"
    if pending.exists():
        pending.unlink()

    return Module3RunResult(**payload)


@router.post("/generate", response_model=Module3GenerateResult)
def generate_situation() -> Module3GenerateResult:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_output = Path(tmp) / "generated.json"
        payload = _run_module3_script(
            ["interactive.py", "generate", str(tmp_output)],
            STEP_TIMEOUT_S,
            tmp_output,
        )
    return Module3GenerateResult(**payload)


@router.post("/process", response_model=Module3ProcessResult)
def process_gate() -> Module3ProcessResult:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_output = Path(tmp) / "processed.json"
        payload = _run_module3_script(
            ["interactive.py", "process", str(tmp_output)],
            STEP_TIMEOUT_S,
            tmp_output,
        )
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
        return _run_module3_script(["scripts/storm_run.py", "presets", str(out)], STEP_TIMEOUT_S, out)


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
        return _run_module3_script(["scripts/storm_run.py", str(inp), str(out)], STORM_TIMEOUT_S, out)
