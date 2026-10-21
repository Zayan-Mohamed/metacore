"""Dashboard "Train" run: train one small gate with PPO and score it against the baselines.

Called by `scripts/demo_run.py` (gateway `POST /api/module3/run`). The output keeps the shape the
page has always read: reward / avg_deliberation_cost / escalation_by_severity / ... plus a handful
of `decisions` in the M3->M4 wire format, so the table of proposed actions and their verdicts
still renders. It is produced by the cyclone environment, the grid twin and a PPO-trained
recurrent gate (the same code the full pipeline uses), but at a size that finishes in about a
minute, so it is a demonstration run, not the five-seed evidence in the PP1 pack.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from module3.env.cyclone_env import CycloneGatingEnv
from module3.scenario.cyclone import HAZARD_STAGES
from module3.serve.step import CKPT_SEED, _jsonable, gate_logits
from module3.train.evaluate import BASELINES, evaluate_policy
from module3.train.pipeline import train_seed

TASKS_PER_UPDATE = 8
WARM_START = {"teacher": "breaker_rule", "tasks": 32, "epochs": 20}


def build_config(base: dict, req: dict) -> dict:
    """Apply the dashboard form to the pipeline config. Unknown request keys are ignored."""
    cfg = deepcopy(base)
    cfg["seed"] = int(req.get("seed", cfg.get("seed", 0)))
    cfg["env"]["episode_len"] = int(req.get("episode_len", cfg["env"]["episode_len"]))
    cfg["env"]["budget_per_episode"] = float(
        req.get("budget_per_episode", cfg["env"]["budget_per_episode"])
    )
    for key in ("deliberation_cost", "w_serve", "w_viol"):
        if key in req:
            cfg["reward"][key] = float(req[key])
    storms = int(req.get("train_episodes", 150))
    cfg["ppo"]["tasks_per_update"] = TASKS_PER_UPDATE
    cfg["ppo"]["updates"] = max(2, round(storms / TASKS_PER_UPDATE))
    cfg["policy"]["warm_start"] = dict(WARM_START)
    cfg["policy"]["trials_per_task"] = 1
    cfg["eval"]["episodes_random"] = int(req.get("eval_episodes", 12))
    return cfg


def _wire_records(cfg: dict, model, seed: int) -> tuple[list[dict], list[dict]]:
    """Play one storm with the trained gate; emit the first decision of each hazard stage as a
    ProposedControlAction + GatingDecision pair (M3_TO_M4_CONTRACT.md), plus dashboard context."""
    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(seed))
    obs, _ = env.reset(seed=seed)
    h = model.initial_state(1)
    prev_a, prev_r = 0, 0.0
    records: list[dict] = []
    context: list[dict] = []
    seen: set[str] = set()
    done = False
    while not done:
        logits, h = gate_logits(model, obs, h, prev_a, prev_r)
        obs, r, _, done, info = env.step(int(torch.argmax(logits).item()))
        prev_a, prev_r = info["effective_action"], float(r)
        stage = info["severity"]
        if stage in seen:
            continue
        seen.add(stage)
        n = len(seen) - 1
        action_id = f"m3-{seed:04d}-{n:02d}-{stage}"
        origin = "SYSTEM2" if info["effective_action"] == 1 else "SYSTEM1"
        proposed, parts = info["proposed"], info["reward_parts"]
        records.append(
            {
                "action_id": action_id,
                "origin": origin,
                "breakers": proposed.get("breakers", []),
                "load_shed": proposed.get("load_shed", []),
                "dispatch": proposed.get("dispatch", []),
                "rationale": proposed.get("rationale", ""),
                "schema_version": "m3-out/0.1",
                "message_type": "ProposedControlAction",
            }
        )
        records.append(
            {
                "action_id": action_id,
                "chosen": origin,
                "epistemic_at_decision": float(info["epistemic_uncertainty"]),
                "expected_survival_benefit": float(parts["serve"] - parts["violation"]),
                "deliberation_cost": float(parts["deliberation"]),
                "latency_ms": float(parts["latency_ms"]),  # a cost weight, not a measurement
                "budget_exhausted_fallback": bool(info["budget_exhausted_fallback"]),
                "schema_version": "m3-out/0.1",
                "message_type": "GatingDecision",
                "timestamp": 1_704_067_200.0 + float(n),
            }
        )
        context.append(
            {
                "action_id": action_id,
                "severity": stage,
                "trigger_reason": info["trigger_reason"],
                "epistemic_uncertainty": float(info["epistemic_uncertainty"]),
                "observed_fraction": float(info["observed_fraction"]),
                "competence_drop": info["trigger_reason"] != "none",
                "reward": float(r),
                "verdict": "APPROVE" if info["approved"] else "REJECT",
                "served_fraction": float(info["served_fraction"]),
                "violating_buses": list(info["violating_buses"]),
            }
        )
    return records, context


def run_demo(cfg: dict, state_dir: Path, log=lambda *_: None) -> dict:
    seed = int(cfg["seed"])
    model, _hist, _bc = train_seed(cfg, cfg["policy"]["arch"], seed, log)

    state_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": cfg["policy"]["arch"],
            "hidden": model.hidden,
            "config": cfg,
        },
        state_dir / f"gate_seed{CKPT_SEED}.pt",
    )

    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(10_000 + seed))
    thr = float(cfg["eval"]["threshold_u"])
    n_eval = int(cfg["eval"]["episodes_random"])
    storm_seeds = [1_000_000 * (seed + 1) + i for i in range(n_eval)]
    res = {p: evaluate_policy(env, p, storm_seeds, None, None, thr) for p in BASELINES}
    res["trained_policy"] = evaluate_policy(env, "gate", storm_seeds, None, model, thr)

    gate = res["trained_policy"]
    records, context = _wire_records(cfg, model, seed)
    payload = {
        "reward": {name: float(r["return"]) for name, r in res.items()},
        "avg_deliberation_cost": {
            name: float(res[name]["deliberation_cost"])
            for name in ("always_s2", "threshold", "trained_policy")
        },
        "escalation_by_severity": {s: float(gate["escalation_by_stage"][s]) for s in HAZARD_STAGES},
        "monotonic_nondecreasing": bool(gate["monotonic_nondecreasing"]),
        "escalation_by_trigger_reason": {
            k: float(v) for k, v in gate["escalation_by_reason"].items()
        },
        "decisions": records,
        "decision_context": context,
        "notes": [
            "A short demonstration run: one gate, PPO, warm-started from the breaker rule.",
            "breaker_rule and oracle are references; the oracle reads the true damage.",
            "Scored on the surrogate grid twin, not the real M4.",
        ],
    }
    return _jsonable(payload)
