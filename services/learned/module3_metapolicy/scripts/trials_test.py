"""Repeated-trials test: does the GRU get better at a storm the more it has seen of it?

This is the RL^2 claim proper. The memory test (memory_test.py) showed the GRU's state carries
information WITHIN one storm when observations are incomplete. Here the same storm (a "task") is
played K times in a row and the gate keeps its memory across trials. If memory is doing
meta-learning, the return should rise from trial 1 to trial K. A memoryless gate cannot improve
across trials by construction, so it is the control.

Two kinds of repetition (policy.replay):
    fresh   same storm parameters, new random damage each trial
    exact   the identical storm (same damage and timing): a rehearsal

Both run in the blind world (SCADA slots 12-14 hidden) so there is something to find out.
Arms: GRU and strictly memoryless MLP (no previous-action/reward feedback), each trained with
the same K trials per task.

Usage:  python scripts/trials_test.py [--seeds 0 1 2 3 4] [--trials 4] [--workers 10]
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
BLIND = [12, 13, 14]

# arm -> (arch, feedback)
ARCHS = {"gru": ("gru", True), "mlp_nofb": ("mlp", False)}


def run_one(args: tuple) -> dict:
    arch_name, replay, seed, n_trials, p_grid_loss, n_tasks = args
    import torch

    torch.set_num_threads(1)
    from module3.env.cyclone_env import CycloneGatingEnv
    from module3.train.evaluate import run_trials
    from module3.train.pipeline import _train

    arch, feedback = ARCHS[arch_name]
    cfg = deepcopy(yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8")))
    cfg.setdefault("scenario", {})["p_grid_loss"] = p_grid_loss
    cfg["env"]["hide_obs"] = BLIND
    cfg["policy"].update(feedback=feedback, trials_per_task=n_trials, replay=replay)
    t0 = time.time()
    gate, hist, _ = _train(cfg, arch, seed, lambda *_: None)
    gate.eval()
    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(10_000 + seed))
    tasks = [VAL_OFFSET + 1_000_000 * (seed + 1) + i for i in range(n_tasks)]
    per_task = [run_trials(env, gate, t, n_trials, replay) for t in tasks]
    keys = ("return", "s2_rate", "always_s1", "oracle")
    by_trial = {
        k: [float(np.mean([pt[j][k] for pt in per_task])) for j in range(n_trials)]
        for k in keys
    }
    return {
        "arm": arch_name,
        "replay": replay,
        "seed": seed,
        "by_trial": by_trial,
        "train_tail_entropy": float(np.mean([h["entropy"] for h in hist[-10:]])),
        "seconds": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--replays", nargs="*", default=["fresh", "exact"])
    ap.add_argument("--tasks", type=int, default=40, help="evaluation storms per run")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--p-grid-loss", type=float, default=0.6)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "trials_test" / "latest.json"))
    a = ap.parse_args()
    jobs = [
        (arch, rp, s, a.trials, a.p_grid_loss, a.tasks)
        for rp in a.replays
        for arch in ARCHS
        for s in a.seeds
    ]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(run_one, jobs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    head = " ".join(f"trial {j + 1:d}".rjust(8) for j in range(a.trials))
    print(f"\nblind world, {a.trials} trials per storm, {a.tasks} storms, seeds {a.seeds}\n")
    for rp in a.replays:
        print(f"[{rp}]  mean return per trial index            {head}   last-first")
        for arch in ARCHS:
            rows = [r for r in results if r["replay"] == rp and r["arm"] == arch]
            m = np.mean([r["by_trial"]["return"] for r in rows], axis=0)
            gain = np.array([
                r["by_trial"]["return"][-1] - r["by_trial"]["return"][0] for r in rows
            ])
            print(
                f"  {arch:10s} {'':32s}"
                + " ".join(f"{x:8.2f}" for x in m)
                + f"   {gain.mean():+.2f} (+/-{gain.std(ddof=1) if len(gain) > 1 else 0:.2f})"
            )
        for ref in ("always_s1", "oracle"):
            rows = [r for r in results if r["replay"] == rp and r["arm"] == "gru"]
            m = np.mean([r["by_trial"][ref] for r in rows], axis=0)
            print(f"  {ref:10s} {'':32s}" + " ".join(f"{x:8.2f}" for x in m))
        g = {r["seed"]: r for r in results if r["replay"] == rp and r["arm"] == "gru"}
        c = {r["seed"]: r for r in results if r["replay"] == rp and r["arm"] == "mlp_nofb"}
        d = np.array(
            [
                (g[s]["by_trial"]["return"][-1] - g[s]["by_trial"]["return"][0])
                - (c[s]["by_trial"]["return"][-1] - c[s]["by_trial"]["return"][0])
                for s in sorted(g)
            ]
        )
        print(
            f"  GRU improvement minus MLP improvement: {d.mean():+.2f}, "
            f"GRU larger on {int((d > 0).sum())}/{len(d)} seeds\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
