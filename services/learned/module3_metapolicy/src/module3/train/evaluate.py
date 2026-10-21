"""Scoring a gate (and the baselines it must beat) on identical storms.

Every policy is replayed on the SAME storms (same seeds -> same cyclone, same M2 stream),
so differences between policies are due to decisions alone. No number is reported without
the baselines beside it (repo convention).

Policies:
    always_s1      never deliberate
    always_s2      always request System 2 (budget forces S1 once it runs out)
    threshold      System 2 iff u > threshold_u        <- the 'why not just a threshold?' rule
    breaker_rule   System 2 iff an outage is visible (a tripped feeder or a lost main supply)
                   and the storm is >= elevated
    oracle         System 2 iff an outage ACTUALLY exists (cheats with ground truth;
                   an upper-ish bound on what any gate that cannot see truth could reach)
    gate           the trained RecurrentGate (greedy), memory carried through the storm
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from module3.env.cyclone_env import CycloneGatingEnv
from module3.policy.recurrent_gate import RecurrentGate, build_inputs
from module3.scenario.cyclone import HAZARD_STAGES, sample_params

BASELINES = ("always_s1", "always_s2", "threshold", "breaker_rule", "oracle")


@dataclass(frozen=True)
class GateTuning:
    """Turns the gate's probability of System 2 into a stable decision.

    p2 = P(System 2) from the trained gate. Plain greedy is `p2 > 0.5`. Two refinements:
      * calibrated threshold - escalate when p2 > tau_on, de-escalate when p2 < tau_off. The
        defaults (0.5, 0.5) reproduce plain greedy; tau_on > tau_off is hysteresis: the gate
        needs more evidence to switch System 2 ON than to keep it ON, which stops flickering.
      * min_hold - after a switch, stay in the new mode at least this many steps.
    """

    tau_on: float = 0.5
    tau_off: float = 0.5
    min_hold: int = 0

    def step(self, p2: float, mode: int, held: int) -> tuple[int, int]:
        """Return (new_mode, steps_held_in_new_mode). `mode` is the gate's own last request."""
        if held >= self.min_hold:
            if mode == 0 and p2 > self.tau_on:
                return 1, 1
            if mode == 1 and p2 < self.tau_off:
                return 0, 1
        return mode, held + 1


def _baseline_action(name: str, obs: np.ndarray, truth_tripped: int, thr_u: float) -> int:
    if name == "always_s1":
        return 0
    if name == "always_s2":
        return 1
    if name == "threshold":
        return int(obs[0] > thr_u)
    if name == "breaker_rule":
        return int((obs[12] > 0 or obs[14] > 0) and obs[8] >= 0.33)
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
    tuning: GateTuning | None = None,
) -> dict:
    """Play one storm with one policy. Returns the metrics for that storm.

    `tuning` only affects policy "gate" (None = plain greedy argmax, as before).
    """
    obs, meta = env.reset(seed=seed, options={"cyclone": cyclone})
    h = model.initial_state(1) if model is not None else None
    prev_a, prev_r = 0, 0.0
    ret = served = delib = 0.0
    n = n_s2 = n_reject = n_fallback = n_viol_steps = 0
    by_stage = {s: [0, 0] for s in HAZARD_STAGES}
    by_reason = {r: [0, 0] for r in ("none", "value", "sensing", "both")}
    by_damage = {"no_trip": [0, 0], "trip": [0, 0]}  # steps with / without a feeder really out
    truth = 0
    flips, last_eff = 0, 0
    mode, held = 0, 10**6
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
            if tuning is None:
                action = int(torch.argmax(logits[0, 0]).item())
            else:
                p2 = float(torch.softmax(logits[0, 0], dim=-1)[1])
                mode, held = tuning.step(p2, mode, held)
                action = mode
        else:
            action = _baseline_action(policy, obs, truth, thr_u)
        obs, r, _, done, info = env.step(action)
        prev_a, prev_r = info["effective_action"], float(r)
        truth = info["n_tripped_truth"] + int(info["grid_lost_truth"])
        ret += r
        served += info["served_fraction"]
        delib += info["reward_parts"]["deliberation"]
        n += 1
        n_s2 += info["effective_action"]
        flips += int(info["effective_action"] != last_eff)
        last_eff = info["effective_action"]
        n_reject += int(not info["approved"])
        n_fallback += int(info["budget_exhausted_fallback"])
        n_viol_steps += int(len(info["violating_buses"]) > 0)
        by_stage[info["severity"]][0] += 1
        by_stage[info["severity"]][1] += info["effective_action"]
        by_reason[info["trigger_reason"]][0] += 1
        by_reason[info["trigger_reason"]][1] += info["effective_action"]
        key = "trip" if (info["n_tripped_truth"] > 0 or info["grid_lost_truth"]) else "no_trip"
        by_damage[key][0] += 1
        by_damage[key][1] += info["effective_action"]
    return {
        "return": ret,
        "served_fraction": served / n,
        "reject_rate": n_reject / n,
        "violation_step_rate": n_viol_steps / n,
        "s2_rate": n_s2 / n,
        "flips": flips,
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
        "flips",
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


@torch.no_grad()
def run_trials(
    env: CycloneGatingEnv,
    model: RecurrentGate,
    task_seed: int,
    n_trials: int,
    replay: str = "fresh",
) -> list[dict]:
    """Play ONE storm (task) `n_trials` times in a row, carrying the gate's memory across trials.

    Mirrors training (`ppo.collect_tasks`): memory, previous action and previous reward persist,
    and the first step of each later trial sees trial_done = 1. Greedy actions. Returns, per
    trial, the return plus the same storm played by two memoryless references (always System 1
    and the oracle) so the gate can be judged against what was achievable on that very trial.
    """
    rng = np.random.default_rng(task_seed)
    params = sample_params(rng, env._scenario_cfg)
    seeds = [
        task_seed if replay == "exact" else int(rng.integers(0, 2**31 - 1))
        for _ in range(n_trials)
    ]
    h = model.initial_state(1)
    prev_a, prev_r, done_flag = 0, 0.0, 0.0
    rows = []
    for k, s in enumerate(seeds):
        obs, _ = env.reset(seed=s, options={"cyclone": params})
        ret, n_s2, n, done = 0.0, 0, 0, False
        while not done:
            x = build_inputs(
                torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1),
                torch.tensor([[prev_a]]),
                torch.tensor([[prev_r]], dtype=torch.float32),
                torch.tensor([[done_flag]], dtype=torch.float32),
            )
            logits, _, h = model(x, h)
            a = int(torch.argmax(logits[0, 0]).item())
            obs, r, _, done, info = env.step(a)
            prev_a, prev_r, done_flag = a, float(r), 0.0
            ret += r
            n_s2 += info["effective_action"]
            n += 1
        done_flag = 1.0
        ref = {
            p: run_episode(env, p, s, params)["return"] for p in ("always_s1", "oracle")
        }
        rows.append({"trial": k + 1, "return": ret, "s2_rate": n_s2 / n, **ref})
    return rows


def evaluate_policy(
    env: CycloneGatingEnv,
    policy: str,
    seeds: list[int],
    cyclone=None,
    model: RecurrentGate | None = None,
    thr_u: float = 0.4,
    tuning: GateTuning | None = None,
) -> dict:
    eps = [run_episode(env, policy, s, cyclone, model, thr_u, tuning) for s in seeds]
    return aggregate(eps)
