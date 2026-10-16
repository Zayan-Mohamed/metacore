"""Compare training recipes for the gate, seed by seed, on VALIDATION storms.

Why a sweep: a single run can look fine or fail by luck (seed-to-seed spread was the problem),
so each recipe is trained on several seeds and judged on three things:
    * mean return,
    * the WORST seed (is there a collapse?),
    * how many seeds beat the visible-outage rule.
Storm seeds here (offset 500_000_000) are disjoint from the final-report storms, so choosing a
recipe on this sweep does not leak into the final numbers.

Usage:  python scripts/stability_sweep.py [--seeds 0 1 2 3 4] [--workers 10] [--p-grid-loss 0.6]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
VAL_OFFSET = 500_000_000

# name -> overrides applied on top of pipeline.yaml (nested dict merge)
RECIPES: dict[str, dict] = {
    "A_baseline": {},
    "B_value_norm": {"ppo": {"value_norm": True}},
    "C_gentle": {
        "ppo": {
            "value_norm": True,
            "lr": 3e-4,
            "tasks_per_update": 32,
            "ent_coef": 0.02,
            "ent_coef_final": 0.002,
            "target_kl": 0.02,
            "critic_warmup_updates": 3,
        }
    },
    "D_gentle_warmstart": {
        "ppo": {
            "value_norm": True,
            "lr": 3e-4,
            "tasks_per_update": 32,
            "ent_coef": 0.02,
            "ent_coef_final": 0.002,
            "target_kl": 0.02,
            "critic_warmup_updates": 3,
        },
        "policy": {"warm_start": {"teacher": "breaker_rule", "tasks": 64, "epochs": 30}},
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out.get(k, {}), v) if isinstance(v, dict) else v
    return out


def run_one(args: tuple) -> dict:
    name, overrides, seed, p_grid_loss = args
    import torch

    torch.set_num_threads(1)
    from module3.train.pipeline import _evaluate_all, _train

    cfg = _merge(yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8")), overrides)
    cfg.setdefault("scenario", {})["p_grid_loss"] = p_grid_loss
    t0 = time.time()
    gate, hist, bc = _train(cfg, cfg["policy"]["arch"], seed, lambda *_: None)
    models = {"gate": gate, **({"gate_bc": bc} if bc is not None else {})}
    res = _evaluate_all(cfg, models, seed, n_random=40, n_profile=20, storm_offset=VAL_OFFSET)
    tail = hist[-10:]
    return {
        "recipe": name,
        "seed": seed,
        "random": {k: res["random"][k]["return"] for k in res["random"]},
        "ditwah": {k: res["ditwah_like"][k]["return"] for k in res["ditwah_like"]},
        "gate_s2_rate": res["random"]["gate"]["s2_rate"],
        "gate_s2_when_intact": res["random"]["gate"]["escalation_by_damage"]["no_trip"],
        "train_tail_entropy": float(np.mean([h["entropy"] for h in tail])),
        "train_tail_s2": float(np.mean([h["s2_rate"] for h in tail])),
        "seconds": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--recipes", nargs="*", default=list(RECIPES))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--p-grid-loss", type=float, default=0.6)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "sweep" / "latest.json"))
    a = ap.parse_args()
    jobs = [(r, RECIPES[r], s, a.p_grid_loss) for r in a.recipes for s in a.seeds]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        results = list(ex.map(run_one, jobs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nvalidation storms, p_grid_loss={a.p_grid_loss}, seeds={a.seeds}\n")
    hdr = (
        f"{'recipe':20s} {'mean':>7s} {'std':>6s} {'worst':>7s} {'>rule':>7s} "
        f"{'S2 rate':>8s} {'S2|intact':>10s}"
    )
    print(hdr + "   per-seed (gate return)")
    for r in a.recipes:
        rows = [x for x in results if x["recipe"] == r]
        g = np.array([x["random"]["gate"] for x in rows])
        rule = np.array([x["random"]["breaker_rule"] for x in rows])
        print(
            f"{r:20s} {g.mean():7.2f} {g.std(ddof=1) if len(g) > 1 else 0:6.2f} {g.min():7.2f} "
            f"{int((g > rule).sum())}/{len(g):<4d} "
            f"{np.mean([x['gate_s2_rate'] for x in rows]):8.2f} "
            f"{np.mean([x['gate_s2_when_intact'] for x in rows]):10.2f}   "
            + " ".join(f"{v:5.1f}" for v in g)
        )
        if any("gate_bc" in x["random"] for x in rows):
            b = np.array([x["random"]["gate_bc"] for x in rows])
            print(
                f"{'  (warm start only)':20s} {b.mean():7.2f} {b.std(ddof=1):6.2f} {b.min():7.2f}"
            )
    rule_all = np.array([x["random"]["breaker_rule"] for x in results])
    orc = np.array([x["random"]["oracle"] for x in results])
    s1 = np.array([x["random"]["always_s1"] for x in results])
    print(
        f"\nreference (same storms): always_s1 {s1.mean():.2f} | "
        f"outage rule {rule_all.mean():.2f} | oracle {orc.mean():.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
