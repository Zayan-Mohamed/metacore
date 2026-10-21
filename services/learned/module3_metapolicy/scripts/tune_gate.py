"""Calibrate the gate's escalation threshold and add hysteresis, without retraining.

The trained gate outputs P(System 2). Greedy play ("escalate if P > 0.5") is what the earlier
results used. Here, per seed, we search a small grid of decision rules on VALIDATION storms
(offset 500_000_000, never used for the final report) and then score the winners, and the
greedy rule, on the final-report storms:

    greedy        P > 0.5                           (the earlier setting)
    calibrated    one threshold t: escalate iff P > t        (best validation return)
    hysteresis    escalate iff P > tau_on, drop iff P < tau_off, hold >= min_hold steps
                  (fewest switches whose validation return is within `--tol` of greedy)

"Flips" = System 1 <-> System 2 switches per storm: a chattering gate is hard to trust and a
real controller would wear out the hand-over.

    python scripts/tune_gate.py --ckpt artifacts/pipeline/final_islanded
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL_OFFSET = 500_000_000
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
TAU_ON = (0.5, 0.6, 0.7)
TAU_OFF = (0.2, 0.3, 0.4, 0.5)
HOLDS = (0, 2, 4)


def _grid():
    cal = [(t, t, 0) for t in THRESHOLDS]
    hyst = [(on, off, h) for on, off, h in itertools.product(TAU_ON, TAU_OFF, HOLDS) if off <= on]
    return cal, [c for c in hyst if c not in cal]


def _storm_seeds(seed: int, offset: int, n: int, suite_index: int = 0) -> list[int]:
    return [offset + 1_000_000 * (seed + 1) + 10_000 * suite_index + i for i in range(n)]


def run_seed(args: tuple) -> dict:
    ckpt_dir, seed, n_val, n_test, tol = args
    import torch

    torch.set_num_threads(1)
    from module3.env.cyclone_env import CycloneGatingEnv
    from module3.policy.recurrent_gate import RecurrentGate
    from module3.train.evaluate import GateTuning, evaluate_policy

    blob = torch.load(Path(ckpt_dir) / f"gate_seed{seed}.pt", weights_only=False)
    cfg = blob["config"]
    model = RecurrentGate(hidden=blob["hidden"], arch=blob["arch"]).eval()
    model.load_state_dict(blob["state_dict"])
    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(10_000 + seed))
    thr = float(cfg["eval"]["threshold_u"])

    def score(tuning, storms, spec=None):
        r = evaluate_policy(env, "gate", storms, spec, model, thr, tuning)
        return {
            "return": r["return"],
            "flips": r["flips"],
            "s2_rate": r["s2_rate"],
            "s2_intact": r["escalation_by_damage"]["no_trip"],
        }

    val = _storm_seeds(seed, VAL_OFFSET, n_val)
    cal, hyst = _grid()
    greedy_val = score(GateTuning(), val)
    table = {c: score(GateTuning(*c), val) for c in cal + hyst}

    best_cal = max(cal, key=lambda c: table[c]["return"])
    ok = [c for c in hyst + cal if table[c]["return"] >= greedy_val["return"] - tol]
    best_hyst = min(ok, key=lambda c: (table[c]["flips"], -table[c]["return"]))

    test = _storm_seeds(seed, 0, n_test)
    ditwah = _storm_seeds(seed, 0, 20, suite_index=2)
    chosen = {"greedy": (0.5, 0.5, 0), "calibrated": best_cal, "hysteresis": best_hyst}
    out = {"seed": seed, "chosen": {k: list(v) for k, v in chosen.items()}, "val": greedy_val}
    for name, c in chosen.items():
        out[name] = {
            "random": score(GateTuning(*c), test),
            "ditwah": score(GateTuning(*c), ditwah, "ditwah_like"),
        }
    out["grid"] = {f"{c[0]}/{c[1]}/{c[2]}": table[c] for c in table}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "artifacts" / "pipeline" / "final_islanded"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--tol", type=float, default=0.15, help="allowed validation-return loss")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "tuning" / "latest.json"))
    a = ap.parse_args()
    jobs = [(a.ckpt, s, a.n_val, a.n_test, a.tol) for s in a.seeds]
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(run_seed, jobs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\nchosen on validation storms; scored on final-report storms; seeds {a.seeds}\n")
    for suite in ("random", "ditwah"):
        print(f"[{suite}]")
        print(
            f"{'rule':12s} {'return':>14s} {'flips/storm':>14s} {'S2 rate':>9s} {'S2|intact':>10s}"
        )
        for name in ("greedy", "calibrated", "hysteresis"):
            rows = [r[name][suite] for r in res]
            ret = np.array([x["return"] for x in rows])
            fl = np.array([x["flips"] for x in rows])
            print(
                f"{name:12s} {ret.mean():7.2f}+/-{ret.std(ddof=1):4.2f} "
                f"{fl.mean():8.1f}+/-{fl.std(ddof=1):4.1f} "
                f"{np.mean([x['s2_rate'] for x in rows]):9.2f} "
                f"{np.mean([x['s2_intact'] for x in rows]):10.2f}"
            )
        print()
    print("per-seed chosen (tau_on / tau_off / hold) and test return:")
    for r in res:
        g, c, h = (r[k]["random"]["return"] for k in ("greedy", "calibrated", "hysteresis"))
        print(
            f"  seed {r['seed']}: calibrated {r['chosen']['calibrated']} -> {c:.2f}  | "
            f"hysteresis {r['chosen']['hysteresis']} -> {h:.2f}  | greedy {g:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
