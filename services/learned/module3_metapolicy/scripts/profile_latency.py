"""Measure how long each stage of one gating decision really takes.

Replaces the invented latency constants (S1 2 ms, S2 18 ms) in `pipeline.yaml: reward` with
measured numbers. One control step is:

    gate forward  ->  S1 or S2 plan  ->  twin / M4 verification

so we time each stage separately, over the same simulated storms the gate is trained on, and
report p50 / p95 / p99 / max in milliseconds (a mean hides the slow tail that decides whether
a real-time deadline is met).

    python scripts/profile_latency.py [--steps 3000] [--live-m4 http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from module3.control.planner import deliberative_plan
from module3.control.reactive import reactive_plan
from module3.env.cyclone_env import CycloneGatingEnv
from module3.policy.recurrent_gate import RecurrentGate, build_inputs
from module3.twin.live_m4 import LiveM4Twin
from module3.twin.surrogate import SurrogateTwin

ROOT = Path(__file__).resolve().parents[1]


def _ms(fn, *a, **k):
    t0 = time.perf_counter_ns()
    out = fn(*a, **k)
    return out, (time.perf_counter_ns() - t0) / 1e6


def _summary(xs: list[float]) -> dict:
    v = np.asarray(xs)
    return {
        "n": int(v.size),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
        "mean": float(v.mean()),
    }


def profile(cfg: dict, steps: int, live_url: str | None) -> dict:
    torch.set_num_threads(1)  # a deployed gate is one small model on one core
    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(0))
    model = RecurrentGate(hidden=int(cfg["policy"]["hidden"]), arch="gru").eval()
    surrogate = SurrogateTwin(env._tier_weights)
    live = LiveM4Twin(live_url, tier_weights=cfg.get("tier_weights")) if live_url else None

    samples: dict[str, list[float]] = {
        k: [] for k in ("gate", "s1_plan", "s2_plan", "twin_surrogate", "env_step")
    }
    if live:
        samples["m4_live_verify"] = []
    s2_by_case: dict[str, list[float]] = {"intact": [], "feeder_out": [], "grid_lost": []}

    h = model.initial_state(1)
    seed, n = 1_000, 0
    while n < steps:
        obs, _ = env.reset(seed=seed, options={"cyclone": None})
        seed += 1
        h = model.initial_state(1)
        done = False
        while not done and n < steps:
            ep, t = env._episode, env._t
            ctx = ep.context(t)
            reason = str(env._m2[t].get("trigger_reason", "none"))
            known_tripped, pv_k, dz_k, gl_k = ep.known(t)

            x = build_inputs(
                torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1),
                torch.tensor([[0]]),
                torch.tensor([[0.0]]),
                torch.tensor([[0.0]]),
            )
            with torch.no_grad():
                (logits, _, h), dt = _ms(model, x, h)
            samples["gate"].append(dt)
            action = int(torch.argmax(logits[0, 0]).item())

            _, dt = _ms(reactive_plan, ctx, reason)
            samples["s1_plan"].append(dt)
            plan2, dt = _ms(
                deliberative_plan,
                known_tripped,
                ctx,
                grid_lost=gl_k,
                pv_factor=pv_k,
                diesel_factor=dz_k,
                tier_weights=env._tier_weights,
            )
            samples["s2_plan"].append(dt)
            case = "grid_lost" if gl_k else ("feeder_out" if known_tripped else "intact")
            s2_by_case[case].append(dt)

            dist = ep.disturbance(t)
            _, dt = _ms(surrogate.evaluate, plan2, dist)
            samples["twin_surrogate"].append(dt)
            if live and not dist.grid_lost:
                _, dt = _ms(live.evaluate, plan2, dist)
                samples["m4_live_verify"].append(dt)

            (obs, _, _, done, _), dt = _ms(env.step, action)
            samples["env_step"].append(dt)
            n += 1

    out = {k: _summary(v) for k, v in samples.items() if v}
    out["s2_plan_by_case"] = {k: _summary(v) for k, v in s2_by_case.items() if v}
    return out


def render(res: dict) -> str:
    rows = ["| stage | n | p50 ms | p95 ms | p99 ms | max ms |", "|---|---|---|---|---|---|"]
    for k, v in res.items():
        if k == "s2_plan_by_case":
            continue
        rows.append(
            f"| {k} | {v['n']} | {v['p50']:.4f} | {v['p95']:.4f} | {v['p99']:.4f} "
            f"| {v['max']:.3f} |"
        )
    rows += ["", "System 2 planning time by situation:", ""]
    rows += ["| situation | n | p50 ms | p99 ms | max ms |", "|---|---|---|---|---|"]
    for k, v in res["s2_plan_by_case"].items():
        rows.append(f"| {k} | {v['n']} | {v['p50']:.4f} | {v['p99']:.4f} | {v['max']:.3f} |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--live-m4", default=None, help="gateway URL to also time the real M4")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "latency" / "profile.json"))
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("scenario", {})["p_grid_loss"] = 0.3 if not a.live_m4 else 0.0
    res = profile(cfg, a.steps, a.live_m4)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(render(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
