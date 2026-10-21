"""Check the SurrogateTwin against the real M4 firewall.

Replays N random (disturbance, action) pairs on both and reports how often they agree on
(a) APPROVE/REJECT and (b) the exact set of violating buses, plus the served-fraction error.
This is the evidence that training on the surrogate is legitimate.

Needs a running gateway:
    uv run uvicorn gateway.main:app --app-dir services/gateway --port 8000
Usage:
    python scripts/validate_surrogate.py [--base-url http://127.0.0.1:8000] [--n 300] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from module3.twin import Disturbance, LiveM4Twin, SurrogateTwin
from module3.twin.topology import LINES, NODES, OVERHEAD_LINES, PRIORITY_TIER


def random_case(rng: random.Random) -> tuple[Disturbance, dict]:
    pool = list(OVERHEAD_LINES) * 3 + [ln.name for ln in LINES]  # bias toward overhead trips
    tripped = frozenset(rng.sample(pool, k=rng.choice([0, 0, 1, 1, 2, 3])))
    shed = []
    for node in rng.sample(NODES, k=rng.randint(0, 5)):
        shed.append(
            {
                "node_id": node,
                "shed_fraction": rng.choice([0.1, 0.35, 0.99, 1.0]),
                "priority_tier": PRIORITY_TIER[node],
            }
        )
    action = {
        "origin": "SYSTEM1",
        "breakers": [],
        "load_shed": shed,
        "dispatch": [],
        "rationale": "validation",
    }
    return Disturbance(
        tripped_lines=tripped,
        pv_factor=rng.choice([1.0, 0.4, 0.0]),
        diesel_factor=rng.choice([1.0, 0.6]),
    ), action


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sur, live = SurrogateTwin(), LiveM4Twin(args.base_url)
    n_dec = n_set = 0
    served_err = 0.0
    mismatches = []
    for i in range(args.n):
        dist, action = random_case(rng)
        s, m = sur.evaluate(action, dist), live.evaluate(action, dist)
        n_dec += int(s.approved == m.approved)
        same_set = set(s.violating_buses) == {b.upper() for b in m.violating_buses}
        n_set += int(same_set)
        served_err += abs(s.served_fraction - m.served_fraction)
        if s.approved != m.approved or not same_set:
            mismatches.append(
                {
                    "case": i,
                    "tripped": sorted(dist.tripped_lines),
                    "shed": {c["node_id"]: c["shed_fraction"] for c in action["load_shed"]},
                    "surrogate": s.violating_buses,
                    "live": m.violating_buses,
                }
            )
    summary = {
        "n": args.n,
        "decision_agreement": n_dec / args.n,
        "violating_set_agreement": n_set / args.n,
        "mean_abs_served_fraction_error": served_err / args.n,
        "mismatches_shown": mismatches[:10],
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
