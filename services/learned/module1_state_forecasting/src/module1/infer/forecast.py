"""Forecast every node's vulnerability from one hour, with the trained ST-GNN v0.1.

    python -m module1.infer.forecast --timestamp 2025-11-28T08:00 --json

This is Module 1's second key function behind the dashboard: the gateway's ``/module1/forecast``
shells out to it, the same way ``/module1/assemble`` shells out to ``module1.assemble``.

It returns, per node, the last 24 hours of the (constructed) vulnerability score, the ST-GNN and
persistence forecasts for t+1 .. t+6, the target those hours actually reached, the hazard /
fragility / stress components, and the 64-d node embedding the M1 -> M2 contract reserves.

It always says which split the chosen hour belongs to. An hour from the training split is a
replay of data the model was fitted on, not evidence of skill, and the page must be able to say so.
The checkpoint is produced by ``module1.train.run --variant v0.1`` and is git-ignored; when it is
absent the CLI exits with code 2 and says how to make it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from module1.assemble import ISLAND_KEYS
from module1.assemble import assemble as assemble_state
from module1.models.stgnn import EMBEDDING_DIM, STGNN, persistence
from module1.train import dataset as ds
from module1.train import target as tgt
from module1.train.run import CONFIG, VARIANTS

VARIANT = "v0.1"
SERVICE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = SERVICE_ROOT.parents[2]
DEFAULT_CKPT_DIR = SERVICE_ROOT / "artifacts"
METRICS_JSON = REPO_ROOT / "reports" / "data-analysis-report" / "m1" / "forecast_v0_1_metrics.json"


class CheckpointMissing(FileNotFoundError):
    pass


def checkpoint_path(weighting: str, ckpt_dir: Path = DEFAULT_CKPT_DIR) -> Path:
    return ckpt_dir / f"stgnn_{VARIANT.replace('.', '_')}_{weighting}.pt"


def load_model(data: ds.ForecastData, weighting: str, ckpt_dir: Path = DEFAULT_CKPT_DIR) -> STGNN:
    path = checkpoint_path(weighting, ckpt_dir)
    if not path.exists():
        raise CheckpointMissing(
            f"no trained checkpoint at {path}. Train it with: python -m module1.train.run "
            f"--variant {VARIANT} --weightings {weighting} --out reports/data-analysis-report/m1 "
            f"--ckpt {ckpt_dir}"
        )
    a_hat = torch.tensor(data.graph.normalized(), dtype=torch.float32)
    model = STGNN(len(ds.FEATURES), CONFIG["horizon_h"], a_hat, CONFIG["hidden"],
                  **VARIANTS[VARIANT])
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval()


def split_of(data: ds.ForecastData, t: int) -> str:
    for name, (a, b) in data.split_index.items():
        if a <= t < b:
            return name
    return "none"


def contract_state(data: ds.ForecastData, timestamp: str, data_root=None) -> dict:
    """The M1 -> M2 contract fields for this hour, built by ``module1.assemble`` itself.

    One assembled state per island, concatenated in graph node order, so ``node_features``,
    ``quality_mask`` and ``observed_fraction`` are exactly what ``/module1/assemble`` publishes --
    one code path, not a copy that can drift. M2's sensing axis reads these.
    """
    island_id = {name: key for key, name in ISLAND_KEYS.items()}
    rows: list[list[float]] = []
    states = []
    for island in dict.fromkeys(data.graph.node_island):
        st = assemble_state(island_id[island], "normal", timestamp, data_root)
        if st["timestamp_lst"] != timestamp or st["node_names"] != [
            a for i, a in zip(data.graph.node_island, data.graph.node_asset, strict=True)
            if i == island
        ]:
            raise ValueError(f"assembled state for {island} does not match the forecast graph")
        rows.extend(st["node_features"])
        states.append(st)
    first = states[0]
    if any(s["quality_mask"] != first["quality_mask"] for s in states):
        raise ValueError("islands disagree on the quality mask")
    return {
        "schema_version": first["schema_version"],
        "feature_names": first["feature_names"],
        "node_features": rows,
        "quality_mask": first["quality_mask"],
        "observed_fraction": first["observed_fraction"],
    }


def forecast(model: STGNN, data: ds.ForecastData, timestamp: str, weighting: str,
             data_root=None) -> dict:
    """Pure function of a loaded model and the data: what the page shows for one hour."""
    hist, hor = CONFIG["history_h"], CONFIG["horizon_h"]
    matches = np.where(data.timestamps == timestamp)[0]
    if len(matches) == 0:
        raise ValueError(f"{timestamp!r} is not an hour in the record "
                         f"({data.timestamps[0]} .. {data.timestamps[-1]})")
    t = int(matches[0])
    if t - hist + 1 < 0:
        raise ValueError(f"{timestamp} needs {hist} hours of history before it")

    v = data.v[weighting]
    x = data.x[t - hist + 1: t + 1].copy()
    x[:, :, ds.FEATURES.index("vulnerability")] = v[t - hist + 1: t + 1]
    xb = torch.from_numpy(x[None].astype(np.float32))

    with torch.no_grad():
        model(xb)  # warm-up: the first call pays one-off allocation; NFR1 is about steady state
        started = time.perf_counter()
        pred = model(xb)[0].numpy()
        latency_ms = (time.perf_counter() - started) * 1000.0
        emb = model.embed(xb)[0].numpy()
    pers = persistence(xb, hor)[0].numpy()

    future = v[t + 1: t + 1 + hor] if t + hor < len(v) else None
    a, _ = data.split_index["train"]
    tau = float(np.quantile(v[a: data.split_index["train"][1]], CONFIG["event_quantile"]))

    nodes = []
    for j in range(data.graph.n):
        nodes.append({
            "id": data.graph.node_ids[j],
            "island": data.graph.node_island[j],
            "asset": data.graph.node_asset[j],
            "current": round(float(v[t, j]), 4),
            "history": [round(float(z), 4) for z in v[t - hist + 1: t + 1, j]],
            "forecast": [round(float(z), 4) for z in pred[j]],
            "persistence": [round(float(z), 4) for z in pers[j]],
            "target": None if future is None else [round(float(z), 4) for z in future[:, j]],
            "components": {k: round(float(data.components[k][t, j]), 4)
                           for k in ("hazard", "fragility", "stress")},
            "embedding": [round(float(z), 4) for z in emb[j]],
        })
    order = np.argsort(-pred[:, 0])
    return {
        "timestamp_lst": timestamp,
        "split": split_of(data, t),
        "weighting": weighting,
        "weights": dict(zip(("hazard", "fragility", "stress"), tgt.WEIGHTINGS[weighting],
                            strict=True)),
        "variant": VARIANT,
        "horizon_h": hor,
        "history_h": hist,
        "tau": round(tau, 4),
        "embedding_dim": EMBEDDING_DIM,
        "has_embedding": True,
        "ranking_t1": [data.graph.node_ids[i] for i in order],
        "above_tau_t1": [data.graph.node_ids[i] for i in range(data.graph.n) if pred[i, 0] > tau],
        "latency_ms": round(latency_ms, 3),
        "data_source": data.data_source,
        "quality": "QUALITY_INTERPOLATED",  # of the vulnerability score itself
        **contract_state(data, timestamp, data_root),
        "nodes": nodes,
    }


def test_summary(weighting: str) -> dict | None:
    """Headline held-out numbers for the page, straight from the committed metrics file."""
    if not METRICS_JSON.exists():
        return None
    res = json.loads(METRICS_JSON.read_text())["results"].get(weighting)
    if res is None:
        return None
    m = res["models"]
    return {k: {"rmse_mean": round(m[k]["test"]["rmse_mean"], 4),
                "onset_recall": round(m[k]["test"]["onset"]["onset_recall"], 3),
                "mase": round(m[k]["test"]["mase"], 3),
                "ditwah_rmse": round(m[k]["test"]["regimes"]["ditwah"]["rmse"], 4)}
            for k in ("stgnn", "persistence", "lstm_1d", "lstm_1d_residual")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-node vulnerability forecast for one hour.")
    ap.add_argument("--timestamp", required=True, help="ISO local hour, e.g. 2025-11-28T08:00")
    ap.add_argument("--weighting", default="equal", choices=sorted(tgt.WEIGHTINGS))
    ap.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    started = time.perf_counter()
    data = ds.build(args.data_root)
    try:
        model = load_model(data, args.weighting, Path(args.ckpt_dir))
    except CheckpointMissing as exc:
        print(str(exc), file=sys.stderr)
        return 2
    try:
        out = forecast(model, data, args.timestamp, args.weighting, args.data_root)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    out["test_summary"] = test_summary(args.weighting)
    out["generated_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    if args.json:
        json.dump(out, sys.stdout)
        sys.stdout.write("\n")
    else:
        print(f"{out['timestamp_lst']}  split={out['split']}  tau={out['tau']}")
        for n in out["nodes"]:
            print(f"  {n['id']:<24} now {n['current']:.3f}  t+1 {n['forecast'][0]:.3f}  "
                  f"t+6 {n['forecast'][-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
