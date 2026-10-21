"""Dashboard step-through: generate one gate input, then run the saved gate on it.

Train lives in `serve/demo.py` (it saves `gate_seed0.pt` into artifacts/dashboard/). These two
commands are the buttons after that: "Generate situation" and "Process gate".

Generate draws a random cyclone, plays the saved gate through it up to a random step, and shows the
15 numbers the gate sees there. Process replays the same storm to the same step (the storm is fixed
by its seed, so this is exact), lets the gate decide, and runs the chosen control path on the grid
twin. Replaying matters: the gate has a memory and a System 2 budget, so what it sees at step t
depends on what it did before step t.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import torch

from module3.env.cyclone_env import CycloneGatingEnv, describe_observation
from module3.policy.recurrent_gate import RecurrentGate, build_inputs
from module3.serve.storm import StormInputError, load_gate

STATE_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "dashboard"
CKPT_SEED = 0  # the dashboard always trains and reads the gate saved as gate_seed0.pt
PENDING = "pending.json"
COUNTER = "generate_count.txt"


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def gate_logits(model: RecurrentGate, obs, h, prev_action: int, prev_reward: float):
    """One gate step: returns (logits for [S1, S2], new memory). Same inputs as training."""
    x = build_inputs(
        torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1),
        torch.tensor([[prev_action]]),
        torch.tensor([[prev_reward]], dtype=torch.float32),
        torch.tensor([[0.0]]),
    )
    with torch.no_grad():
        logits, _, h = model(x, h)
    return logits[0, 0], h


def raw_view(env: CycloneGatingEnv, t: int) -> dict:
    """What the situation at step t looks like in plain terms (for the dashboard, not the gate)."""
    ep, m2 = env._episode, env._m2[t]
    ctx = ep.context(t)
    reason = str(m2.get("trigger_reason", "none"))
    return {
        "epistemic_uncertainty": float(m2["epistemic_uncertainty"]),
        "trigger_reason": reason,
        "observed_fraction": float(m2.get("observed_fraction", 1.0)),
        "competence_drop": bool(m2.get("competence_drop", reason != "none")),
        "severity": ctx["severity"],
        "max_node_vulnerability": float(ctx["max_node_vulnerability"]),
        "top_at_risk_nodes": list(ctx["top_at_risk_nodes"]),
        "wind_kmh": float(ctx["wind_kmh"]),
        "tripped_lines_true": len(ep.tripped[t]),
        "comms_up": bool(ep.comms_up[t]),
        "grid_lost_true": bool(ep.truth_grid_lost(t)),
        "budget_remaining": float(env._budget),
    }


def rollout_to(env: CycloneGatingEnv, model: RecurrentGate, seed: int, t: int):
    """Start storm `seed` and let the gate act greedily for steps 0..t-1. Returns the state at t."""
    obs, _ = env.reset(seed=seed)
    h = model.initial_state(1)
    prev_a, prev_r = 0, 0.0
    for _ in range(t):
        logits, h = gate_logits(model, obs, h, prev_a, prev_r)
        obs, r, _, _, info = env.step(int(torch.argmax(logits).item()))
        prev_a, prev_r = info["effective_action"], float(r)
    return obs, h, prev_a, prev_r


def _next_index(state_dir: Path) -> int:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / COUNTER
    try:
        n = int(path.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        n = 0
    path.write_text(str(n + 1), encoding="utf-8")
    return n


def generate(output_path: Path, state_dir: Path = STATE_DIR) -> dict:
    model, cfg = load_gate(state_dir, CKPT_SEED)  # raises StormInputError if not trained yet
    seed = int(cfg.get("seed", 0)) + 1 + _next_index(state_dir)
    rng = np.random.default_rng(seed)
    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(seed))
    t = int(rng.integers(0, env.episode_len))
    obs, *_ = rollout_to(env, model, seed, t)
    payload = {
        "step_index": t,
        "episode_len": env.episode_len,
        "observation": describe_observation(obs),
        "raw": raw_view(env, t),
        "seed_used": seed,
    }
    (state_dir / PENDING).write_text(json.dumps({"seed": seed, "t": t}), encoding="utf-8")
    output_path.write_text(json.dumps(_jsonable(payload)), encoding="utf-8")
    return payload


def process(output_path: Path, state_dir: Path = STATE_DIR) -> dict:
    model, cfg = load_gate(state_dir, CKPT_SEED)
    pending_path = state_dir / PENDING
    if not pending_path.is_file():
        raise StormInputError("no generated situation - click Generate first")
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    seed, t = int(pending["seed"]), int(pending["t"])

    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(seed))
    obs, h, prev_a, prev_r = rollout_to(env, model, seed, t)
    raw = raw_view(env, t)
    logits, _ = gate_logits(model, obs, h, prev_a, prev_r)
    probs = torch.softmax(logits, dim=-1).tolist()
    requested = int(torch.argmax(logits).item())
    _, reward, _, _, info = env.step(requested)

    origin = "SYSTEM2" if info["effective_action"] == 1 else "SYSTEM1"
    proposed = info["proposed"]
    payload = {
        "chosen": origin,
        "requested": "SYSTEM2" if requested == 1 else "SYSTEM1",
        "budget_exhausted_fallback": bool(info["budget_exhausted_fallback"]),
        "action_probs": {"SYSTEM1": float(probs[0]), "SYSTEM2": float(probs[1])},
        "epistemic_at_decision": float(raw["epistemic_uncertainty"]),
        "deliberation_cost": float(info["reward_parts"]["deliberation"]),
        "plan": {
            "action_id": str(uuid.uuid4()),
            "origin": origin,
            "breakers": proposed.get("breakers", []),
            "load_shed": proposed.get("load_shed", []),
            "dispatch": proposed.get("dispatch", []),
            "rationale": proposed.get("rationale", ""),
            "schema_version": "m3-out/0.1",
            "message_type": "ProposedControlAction",
        },
        "observation": describe_observation(obs),
        "raw": raw,
        "twin": {
            "approved": bool(info["approved"]),
            "violating_buses": list(info["violating_buses"]),
            "served_fraction": float(info["served_fraction"]),
            "reward": float(reward),
            "source": info["twin_source"],
        },
    }
    output_path.write_text(json.dumps(_jsonable(payload)), encoding="utf-8")
    return payload
