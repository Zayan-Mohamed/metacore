"""Does the GRU's memory help? Train GRU and memoryless gates side by side in worlds where the
damage is progressively HIDDEN, on the same seeds and the same validation storms.

Why this test: with the SCADA slots (12-14) visible the gate is simply told what broke, so a
memoryless gate does as well as the GRU. Memory can only matter when the gate has to INFER the
storm's state from what it has seen so far. Two worlds hide information:
    blind        slots 12-14 hidden (which feeders tripped, generation, grid lost)
    very_blind   also severity and time-to-hazard hidden (slots 8, 9)
and each is run with and without the previous action/reward fed back. The feedback matters: the
previous reward reveals damage, so a "memoryless" gate that still gets it has one step of memory.
`_nofb` arms are therefore the strictly memoryless controls.

Usage:  python scripts/memory_test.py [--seeds 0 1 2 3 4] [--workers 10] [--p-grid-loss 0.6]
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
VERY_BLIND = [8, 9, 12, 13, 14]

# arm -> (arch, hidden slots, feedback)
ARMS: dict[str, tuple[str, list[int], bool]] = {
    "visible_gru": ("gru", [], True),
    "visible_mlp": ("mlp", [], True),
    "blind_gru": ("gru", BLIND, True),
    "blind_mlp": ("mlp", BLIND, True),
    "blind_gru_nofb": ("gru", BLIND, False),
    "blind_mlp_nofb": ("mlp", BLIND, False),
    "vblind_gru": ("gru", VERY_BLIND, True),
    "vblind_mlp": ("mlp", VERY_BLIND, True),
    "vblind_mlp_nofb": ("mlp", VERY_BLIND, False),
}
# (memory arm, memoryless control, label)
PAIRS = [
    ("visible_gru", "visible_mlp", "visible"),
    ("blind_gru", "blind_mlp", "blind (both get feedback)"),
    ("blind_gru", "blind_mlp_nofb", "blind: GRU vs strictly memoryless"),
    ("blind_gru_nofb", "blind_mlp_nofb", "blind, no feedback either side"),
    ("vblind_gru", "vblind_mlp", "very blind (both get feedback)"),
    ("vblind_gru", "vblind_mlp_nofb", "very blind: GRU vs strictly memoryless"),
]


def run_one(args: tuple) -> dict:
    arm, seed, p_grid_loss = args
    import torch

    torch.set_num_threads(1)
    from module3.train.pipeline import _evaluate_all, _train

    arch, hide, feedback = ARMS[arm]
    cfg = yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8"))
    cfg = deepcopy(cfg)
    cfg.setdefault("scenario", {})["p_grid_loss"] = p_grid_loss
    cfg["env"]["hide_obs"] = hide
    cfg["policy"]["feedback"] = feedback
    t0 = time.time()
    gate, hist, _ = _train(cfg, arch, seed, lambda *_: None)
    res = _evaluate_all(
        cfg, {"gate": gate}, seed, n_random=40, n_profile=20, storm_offset=VAL_OFFSET
    )
    tail = hist[-10:]
    return {
        "arm": arm,
        "seed": seed,
        "random": {k: res["random"][k]["return"] for k in res["random"]},
        "ditwah": {k: res["ditwah_like"][k]["return"] for k in res["ditwah_like"]},
        "burevi": {k: res["burevi_like"][k]["return"] for k in res["burevi_like"]},
        "gate_s2_rate": res["random"]["gate"]["s2_rate"],
        "gate_s2_when_intact": res["random"]["gate"]["escalation_by_damage"]["no_trip"],
        "train_tail_entropy": float(np.mean([h["entropy"] for h in tail])),
        "seconds": time.time() - t0,
    }


def _by_seed(results: list[dict], arm: str, suite: str = "random") -> dict[int, float]:
    return {r["seed"]: r[suite]["gate"] for r in results if r["arm"] == arm}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--p-grid-loss", type=float, default=0.6)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "memory_test" / "latest.json"))
    a = ap.parse_args()
    jobs = [(arm, s, a.p_grid_loss) for arm in a.arms for s in a.seeds]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as ex:
        results = list(ex.map(run_one, jobs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nvalidation storms, p_grid_loss={a.p_grid_loss}, seeds={a.seeds}\n")
    print(f"{'arm':18s} {'mean':>7s} {'std':>6s} {'worst':>7s} {'S2 rate':>8s} {'S2|intact':>10s}")
    for arm in a.arms:
        rows = [r for r in results if r["arm"] == arm]
        g = np.array([r["random"]["gate"] for r in rows])
        print(
            f"{arm:18s} {g.mean():7.2f} {g.std(ddof=1) if len(g) > 1 else 0:6.2f} {g.min():7.2f} "
            f"{np.mean([r['gate_s2_rate'] for r in rows]):8.2f} "
            f"{np.mean([r['gate_s2_when_intact'] for r in rows]):10.2f}"
        )
    ref = {
        k: np.mean([r["random"][k] for r in results])
        for k in ("always_s1", "breaker_rule", "oracle")
    }
    print(
        f"\nreference (same storms): always_s1 {ref['always_s1']:.2f} | "
        f"outage rule {ref['breaker_rule']:.2f} | oracle {ref['oracle']:.2f}"
    )

    print("\npaired by seed (same storms): memory arm minus control")
    print(f"{'comparison':44s} {'diff':>7s} {'GRU wins':>9s}  per-seed diffs")
    for mem, ctl, label in PAIRS:
        if mem not in a.arms or ctl not in a.arms:
            continue
        m, c = _by_seed(results, mem), _by_seed(results, ctl)
        d = np.array([m[s] - c[s] for s in sorted(m)])
        print(
            f"{label:44s} {d.mean():+7.2f} {int((d > 0).sum())}/{len(d):<7d}  "
            + " ".join(f"{x:+.1f}" for x in d)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
