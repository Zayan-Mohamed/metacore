"""The forecaster's graph as the dashboard's Model graph page shows it.

Nodes, edges, the exact propagation matrix the graph convolution multiplies by, and what the
graph contributed in the experiments (ST-GNN against its graph-free ablation, both weather
sources). Read from the code and the committed metrics files, never restated by hand.

    python -m module1.graph [--json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from module1.graph.build import build_graph, check_invariants
from module1.train import target as tgt

REPO_ROOT = Path(__file__).resolve().parents[6]
METRICS = REPO_ROOT / "reports" / "data-analysis-report" / "m1"
RUNS = {"nasa_power": "forecast_v0_1_metrics.json", "open_meteo": "forecast_v0_1_om_metrics.json"}


def ablations() -> dict:
    out: dict = {}
    for source, name in RUNS.items():
        path = METRICS / name
        if not path.exists():
            continue
        res = json.loads(path.read_text())["results"]
        out[source] = {}
        for w, r in res.items():
            t = {m: r["models"][m]["test"]["rmse_mean"] for m in
                 ("stgnn", "graph_free", "time_free", "no_weather", "lstm_1d_residual")}
            out[source][w] = {
                "rmse": {k: round(v, 4) for k, v in t.items()},
                "graph_gain_pct": round(100 * (1 - t["stgnn"] / t["graph_free"]), 1),
                "temporal_gain_pct": round(100 * (1 - t["stgnn"] / t["time_free"]), 1),
                "weather_gain_pct": round(100 * (1 - t["stgnn"] / t["no_weather"]), 1),
                "params": r["models"]["stgnn"]["train"].get("params"),
            }
    return out


def describe() -> dict:
    g = build_graph()
    check_invariants(g)
    a_hat = g.normalized()
    edges = [[g.node_ids[i], g.node_ids[j]] for i in range(g.n) for j in range(i + 1, g.n)
             if g.adjacency[i, j]]
    nodes = []
    for i in range(g.n):
        neigh = [g.node_ids[j] for j in range(g.n) if g.adjacency[i, j]]
        nodes.append({
            "id": g.node_ids[i], "island": g.node_island[i], "asset": g.node_asset[i],
            "degree": len(neigh), "neighbours": neigh,
            "self_weight": round(float(a_hat[i, i]), 4),
            "neighbour_weight": round(float(a_hat[i].sum() - a_hat[i, i]), 4),
            "exposure": tgt.EXPOSURE[g.node_asset[i]],
        })
    return {
        "nodes": nodes, "edges": edges,
        "a_hat": [[round(float(x), 4) for x in row] for row in a_hat],
        "rule": ("Edges only where a conductor runs: each asset to its own island's bus. No edge "
                 "between islands, because no cable joins them."),
        "layer": ("H' = H + ReLU(A_hat H W): each node's 64-number vector is transformed by W, "
                  "then mixed with its neighbours' using A_hat = D^-1/2 (A + I) D^-1/2. One layer "
                  "in v0.1; two in v0 over-smoothed each bus with its assets."),
        "ablations": ablations(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    d = describe()
    if args.json:
        print(json.dumps(d))
    else:
        for n in d["nodes"]:
            print(f"{n['id']:28s} degree {n['degree']}  self {n['self_weight']:.3f}  "
                  f"neighbours {n['neighbour_weight']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
