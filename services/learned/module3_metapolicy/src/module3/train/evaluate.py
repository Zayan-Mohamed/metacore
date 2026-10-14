"""Scoring a gate (and the baselines it must beat) on identical storms.

Every policy is replayed on the SAME storms (same seeds -> same cyclone, same M2 stream),
so differences between policies are due to decisions alone. No number is reported without
the baselines beside it (repo convention).

Policies:
    always_s1      never deliberate
    always_s2      always request System 2 (budget forces S1 once it runs out)
    threshold      System 2 iff u > threshold_u        <- the 'why not just a threshold?' rule
    breaker_rule   System 2 iff a tripped feeder is visible and the storm is >= elevated
    oracle         System 2 iff a feeder is ACTUALLY tripped (cheats with ground truth;
                   an upper-ish bound on what any gate that cannot see truth could reach)
    gate           the trained RecurrentGate (greedy), memory carried through the storm
"""

from __future__ import annotations

import numpy as np
import torch

from module3.env.cyclone_env import CycloneGatingEnv
from module3.policy.recurrent_gate import RecurrentGate, build_inputs
from module3.scenario.cyclone import HAZARD_STAGES

BASELINES = ("always_s1", "always_s2", "threshold", "breaker_rule", "oracle")


def _baseline_action(name: str, obs: np.ndarray, truth_tripped: int, thr_u: float) -> int:
    if name == "always_s1":
        return 0
    if name == "always_s2":
        return 1
    if name == "threshold":
        return int(obs[0] > thr_u)
    if name == "breaker_rule":
        return int(obs[12] > 0 and obs[8] >= 0.33)
    if name == "oracle":
        return int(truth_tripped > 0)
    raise KeyError(name)


def run_episode(
    env: CycloneGatingEnv,
    policy: str,
    seed: int,
    cyclone=None,
    model: RecurrentGate | None = None,
    thr_u: float = 0.4,
) -> dict:
    """Play one storm with one policy. Returns the metrics for that storm."""
    obs, meta = env.reset(seed=seed, options={"cyclone": cyclone})
    h = model.initial_state(1) if model is not None else None
    prev_a, prev_r = 0, 0.0
    ret = served = delib = 0.0
    n = n_s2 = n_reject = n_fallback = n_viol_steps = 0
    by_stage = {s: [0, 0] for s in HAZARD_STAGES}
    by_reason = {r: [0, 0] for r in ("none", "value", "sensing", "both")}
    by_damage = {"no_trip": [0, 0], "trip": [0, 0]}  # steps with / without a feeder really out
    truth = 0
    done = False
    while not done:
        if policy == "gate":
            x = build_inputs(
                torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1),
                torch.tensor([[prev_a]]),
                torch.tensor([[prev_r]], dtype=torch.float32),
                torch.tensor([[0.0]]),
            )
            with torch.no_grad():
                logits, _, h = model(x, h)
            action = int(torch.argmax(logits[0, 0]).item())
        else:
            action = _baseline_action(policy, obs, truth, thr_u)
        obs, r, _, done, info = env.step(action)
        prev_a, prev_r = info["effective_action"], float(r)
        truth = info["n_tripped_truth"]
        ret += r
        served += info["served_fraction"]
        delib += info["reward_parts"]["deliberation"]
        n += 1
        n_s2 += info["effective_action"]
        n_reject += int(not info["approved"])
        n_fallback += int(info["budget_exhausted_fallback"])
        n_viol_steps += int(len(info["violating_buses"]) > 0)
        by_stage[info["severity"]][0] += 1
        by_stage[info["severity"]][1] += info["effective_action"]
        by_reason[info["trigger_reason"]][0] += 1
        by_reason[info["trigger_reason"]][1] += info["effective_action"]
        key = "trip" if info["n_tripped_truth"] > 0 else "no_trip"
        by_damage[key][0] += 1
        by_damage[key][1] += info["effective_action"]
    return {
        "return": ret,
        "served_fraction": served / n,
        "reject_rate": n_reject / n,
        "violation_step_rate": n_viol_steps / n,
        "s2_rate": n_s2 / n,
        "deliberation_cost": delib,
        "budget_fallbacks": n_fallback,
        "escalation_by_stage": by_stage,
        "escalation_by_reason": by_reason,
        "escalation_by_damage": by_damage,
        "ood": bool(meta["ood"]),
        "peak_wind_kmh": float(meta["peak_wind_kmh"]),
    }


def aggregate(episodes: list[dict]) -> dict:
    """Mean of per-storm metrics, plus escalation rates pooled over all storms."""
    keys = (
        "return",
        "served_fraction",
        "reject_rate",
        "violation_step_rate",
        "s2_rate",
        "deliberation_cost",
        "budget_fallbacks",
    )
    out = {k: float(np.mean([e[k] for e in episodes])) for k in keys}

    def pooled(field: str, names) -> dict:
        res = {}
        for nme in names:
            n = sum(e[field][nme][0] for e in episodes)
            s = sum(e[field][nme][1] for e in episodes)
            res[nme] = (s / n) if n else 0.0
        return res

    out["escalation_by_damage"] = pooled("escalation_by_damage", ("no_trip", "trip"))
    out["escalation_by_stage"] = pooled("escalation_by_stage", HAZARD_STAGES)
    out["escalation_by_reason"] = pooled(
        "escalation_by_reason", ("none", "value", "sensing", "both")
    )
    vals = [out["escalation_by_stage"][s] for s in HAZARD_STAGES]
    out["monotonic_nondecreasing"] = all(
        vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1)
    )
    return out


def evaluate_policy(
    env: CycloneGatingEnv,
    policy: str,
    seeds: list[int],
    cyclone=None,
    model: RecurrentGate | None = None,
    thr_u: float = 0.4,
) -> dict:
    eps = [run_episode(env, policy, s, cyclone, model, thr_u) for s in seeds]
    return aggregate(eps)
