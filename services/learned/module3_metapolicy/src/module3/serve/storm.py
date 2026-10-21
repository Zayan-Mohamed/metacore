"""Play one user-defined cyclone through a trained gate and the baselines, step by step.

This is what the dashboard's "Cyclone lab" page calls (via `scripts/storm_run.py` and the gateway).
The storm does not react to the controller, so the identical storm is replayed for the gate and for
every baseline: any difference in return is down to the decisions alone.
"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from module3.env.cyclone_env import OBS_FIELDS, CycloneGatingEnv
from module3.policy.recurrent_gate import RecurrentGate, build_inputs
from module3.scenario.cyclone import PROFILES, CycloneParams, sample_params
from module3.train.evaluate import BASELINES, _baseline_action

DEFAULT_CKPT_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "pipeline" / "final_islanded"

# the knobs the page exposes, with the range each one is allowed to take
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "peak_wind_kmh": (20.0, 250.0),
    "peak_frac": (0.1, 0.9),
    "width_frac": (0.03, 0.40),
    "trip_scale": (0.0, 0.6),
    "comms_rate": (0.0, 0.6),
    "diesel_loss_prob": (0.0, 1.0),
    "grid_loss_prob": (0.0, 1.0),
}
_CHECKED = {*PARAM_RANGES, "ood"}


class StormInputError(ValueError):
    """The request is malformed (unknown knob, value out of range, missing checkpoint)."""


def preset_table() -> dict:
    """Named storms plus the defaults and ranges for a custom one."""
    default = CycloneParams()
    return {
        "ranges": {k: list(v) for k, v in PARAM_RANGES.items()},
        "default": {k: getattr(default, k) for k in _CHECKED},
        "presets": {
            name: {k: getattr(p, k) for k in _CHECKED} for name, p in PROFILES.items()
        },
        "note": "burevi_like / ditwah_like are placeholder parameter sets, not real storm data",
    }


def build_params(spec: dict) -> CycloneParams:
    """Validate user knobs and build the storm. Unknown keys are an error, not ignored."""
    allowed = {f.name for f in fields(CycloneParams)}
    clean: dict = {}
    for key, val in spec.items():
        if key not in allowed or key not in _CHECKED:
            raise StormInputError(f"unknown storm parameter {key!r}")
        if key == "ood":
            clean[key] = bool(val)
            continue
        lo, hi = PARAM_RANGES[key]
        if not (lo <= float(val) <= hi):
            raise StormInputError(f"{key}={val} is outside the allowed range [{lo}, {hi}]")
        clean[key] = float(val)
    return replace(CycloneParams(name="custom"), **clean)


def load_gate(ckpt_dir: Path, gate_seed: int) -> tuple[RecurrentGate, dict]:
    path = Path(ckpt_dir) / f"gate_seed{gate_seed}.pt"
    if not path.is_file():
        raise StormInputError(
            f"no trained gate at {path} - the checkpoints are not in git; run the pipeline first"
        )
    blob = torch.load(path, weights_only=False, map_location="cpu")
    model = RecurrentGate(hidden=blob["hidden"], arch=blob["arch"]).eval()
    model.load_state_dict(blob["state_dict"])
    return model, blob["config"]


def _plan_view(plan: dict) -> dict:
    """The control action the chosen path proposed, in the M3 -> M4 ProposedControlAction shape."""
    return {
        "origin": str(plan.get("origin", "")),
        "load_shed": [
            {
                "node_id": str(c["node_id"]),
                "shed_fraction": round(float(c["shed_fraction"]), 4),
                "priority_tier": int(c.get("priority_tier", 0)),
            }
            for c in plan.get("load_shed") or []
        ],
        "dispatch": [
            {"node_id": str(c["node_id"]), "p_kw": float(c["p_kw"])}
            for c in plan.get("dispatch") or []
        ],
        "breakers": [
            {"edge_id": str(c["edge_id"]), "closed": bool(c.get("closed", True))}
            for c in plan.get("breakers") or []
        ],
        "rationale": str(plan.get("rationale", "")),
    }


def _play(env: CycloneGatingEnv, params, seed: int, policy: str, model, thr_u: float) -> dict:
    obs, _ = env.reset(seed=seed, options={"cyclone": params})
    ep = env._episode
    h = model.initial_state(1) if policy == "gate" else None
    prev_a, prev_r, truth = 0, 0.0, 0
    steps, ret, served = [], 0.0, 0.0
    done = False
    while not done:
        t = env._t
        p2 = None
        if policy == "gate":
            x = build_inputs(
                torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1),
                torch.tensor([[prev_a]]),
                torch.tensor([[prev_r]], dtype=torch.float32),
                torch.tensor([[0.0]]),
            )
            with torch.no_grad():
                logits, _, h = model(x, h)
            p2 = float(torch.softmax(logits[0, 0], dim=-1)[1])
            action = int(torch.argmax(logits[0, 0]).item())
        else:
            action = _baseline_action(policy, obs, truth, thr_u)
        seen = [round(float(v), 3) for v in obs]  # the 15 inputs the gate saw at this step
        obs, r, _, done, info = env.step(action)
        prev_a, prev_r = info["effective_action"], float(r)
        truth = info["n_tripped_truth"] + int(info["grid_lost_truth"])
        ret += r
        served += info["served_fraction"]
        steps.append(
            {
                "t": t,
                "wind_kmh": float(ep.wind[t]),
                "stage": info["severity"],
                "tripped": info["n_tripped_truth"],
                "grid_lost": bool(info["grid_lost_truth"]),
                "comms_up": info["comms_up"],
                "u": info["epistemic_uncertainty"],
                "trigger": info["trigger_reason"],
                "p_s2": p2,
                "action": info["effective_action"],
                "requested": info["requested_action"],
                "budget_fallback": info["budget_exhausted_fallback"],
                "reward": float(r),
                "served": info["served_fraction"],
                "approved": info["approved"],
                "violations": len(info["violating_buses"]),
                "obs": seen,
                "plan": _plan_view(info["proposed"]),
            }
        )
    n = len(steps)
    return {
        "return": ret,
        "s2_rate": sum(s["action"] for s in steps) / n,
        "served_fraction": served / n,
        "reject_rate": sum(not s["approved"] for s in steps) / n,
        "steps": steps,
    }


def simulate_storm(spec: dict) -> dict:
    """spec: {params: {...}, preset: str|None, seed: int, gate_seed: int, ckpt_dir: str|None,
    random: bool}. A preset is the starting point and `params` overrides it. `random: true` draws
    the storm from the training distribution instead (so the page can show 'a typical storm')."""
    seed = int(spec.get("seed", 0))
    ckpt = Path(spec.get("ckpt_dir") or DEFAULT_CKPT_DIR)
    model, cfg = load_gate(ckpt, int(spec.get("gate_seed", 0)))
    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(10_000 + seed))
    thr_u = float(cfg["eval"]["threshold_u"])

    if spec.get("random"):
        params = sample_params(np.random.default_rng(seed), env._scenario_cfg)
    else:
        base = dict(preset_table()["presets"].get(spec.get("preset") or "", {}))
        params = build_params({**base, **(spec.get("params") or {})})

    runs = {"gate": _play(env, params, seed, "gate", model, thr_u)}
    for name in BASELINES:
        runs[name] = _play(env, params, seed, name, None, thr_u)
    gate_run = runs.pop("gate")
    return {
        "params": {k: getattr(params, k) for k in _CHECKED},
        "seed": seed,
        "gate_seed": int(spec.get("gate_seed", 0)),
        "episode_len": len(gate_run["steps"]),
        "obs_fields": [
            {k: f[k] for k in ("index", "name", "source", "meaning")} for f in OBS_FIELDS
        ],
        "gate": gate_run,
        "baselines": {k: {m: v for m, v in r.items() if m != "steps"} for k, r in runs.items()},
        "notes": [
            "Same storm for every policy; the gate sees the full 15-number observation.",
            "breaker_rule and oracle are references, not deployable policies "
            "(oracle reads the true damage).",
        ],
    }
