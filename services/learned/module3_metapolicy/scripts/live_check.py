"""Score the FINAL trained gates on the real M4 firewall and on the surrogate, same storms.

The earlier live test (docs §22) used one gate of the pre-fix training run. This one covers all five
gates of the current recipe (`artifacts/pipeline/final_islanded`). The real M4 cannot represent a
lost main supply (finding F3), so storms with grid loss are excluded: the question here is whether
the surrogate we train on agrees with the firewall we are accountable to.

Needs the gateway:  uvicorn gateway.main:app --app-dir services/gateway --port 8000
    python scripts/live_check.py [--url http://127.0.0.1:8000] [--episodes 8] [--workers 3]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "artifacts" / "pipeline" / "final_islanded"


def run_seed(args: tuple) -> dict:
    seed, url, n_random, n_profile = args
    import torch

    torch.set_num_threads(1)
    from module3.policy.recurrent_gate import RecurrentGate
    from module3.train.pipeline import _evaluate_all
    from module3.twin.live_m4 import LiveM4Twin

    blob = torch.load(CKPT / f"gate_seed{seed}.pt", weights_only=False, map_location="cpu")
    model = RecurrentGate(hidden=blob["hidden"], arch=blob["arch"]).eval()
    model.load_state_dict(blob["state_dict"])
    cfg = deepcopy(blob["config"])
    cfg.setdefault("scenario", {})["p_grid_loss"] = 0.0
    sur = _evaluate_all(cfg, {"gate": model}, seed, None, n_random, n_profile)
    live = _evaluate_all(
        cfg, {"gate": model}, seed, LiveM4Twin(url, tier_weights=cfg.get("tier_weights")),
        n_random, n_profile,
    )
    return {
        "seed": seed,
        "cells": {
            suite: {
                pol: {
                    "live": live[suite][pol]["return"],
                    "surrogate": sur[suite][pol]["return"],
                    "live_reject": live[suite][pol]["reject_rate"],
                    "surrogate_reject": sur[suite][pol]["reject_rate"],
                    "live_served": live[suite][pol]["served_fraction"],
                    "surrogate_served": sur[suite][pol]["served_fraction"],
                }
                for pol in live[suite]
            }
            for suite in live
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=8, help="random storms per seed")
    ap.add_argument("--profile-episodes", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "live_check" / "latest.json"))
    a = ap.parse_args()
    jobs = [(s, a.url, a.episodes, a.profile_episodes) for s in a.seeds]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(run_seed, jobs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")

    suites = list(res[0]["cells"])
    pols = list(res[0]["cells"][suites[0]])
    print(f"\nreal M4 vs surrogate, same storms, no grid loss; seeds {a.seeds}\n")
    print(
        f"{'suite':14s} {'policy':13s} {'live':>7s} {'surrogate':>10s} "
        f"{'gap':>7s} {'max |gap|':>10s} {'live rej':>9s}"
    )
    worst = 0.0
    for suite in suites:
        for pol in pols:
            live = np.array([r["cells"][suite][pol]["live"] for r in res])
            sur = np.array([r["cells"][suite][pol]["surrogate"] for r in res])
            gap = live - sur
            worst = max(worst, float(np.abs(gap).max()))
            rej = np.mean([r["cells"][suite][pol]["live_reject"] for r in res])
            print(f"{suite:14s} {pol:13s} {live.mean():7.2f} {sur.mean():10.2f} {gap.mean():+7.3f} "
                  f"{np.abs(gap).max():10.3f} {rej:9.3f}")
    print(f"\nlargest per-seed gap anywhere: {worst:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
