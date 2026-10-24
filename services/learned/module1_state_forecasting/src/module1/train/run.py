"""Train the v0 vulnerability forecaster and its baselines; write metrics a reader can check.

    python -m module1.train.run --out reports/data-analysis-report/m1 --ckpt <artifacts dir>
    python -m module1.train.run --out reports/data-analysis-report/m1 --variant v0.1 --rescore

``--rescore`` recomputes the test metrics from the saved predictions without retraining (training
is seeded and deterministic, so retraining reproduces the same predictions anyway).

For every weighting in ``target.WEIGHTINGS`` this trains the ST-GNN and three ablations
(graph-free, time-free, no-weather), the 1-D LSTM baseline, and scores persistence -- all on the
same event-based split and the same seeded windows. The test quarter (Oct-Dec 2025) holds Cyclone
Ditwah and the Eluvaitivu hybrid decay, neither of which training ever sees.

Results are against a **constructed** label (see ``target.py``). The relative comparisons --
against baselines and between ablations -- are the defensible evidence; absolute errors are not.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from module1.graph.build import check_invariants
from module1.models.stgnn import EMBEDDING_DIM, LSTM1D, STGNN, persistence
from module1.train import dataset as ds
from module1.train import metrics as m
from module1.train import target as tgt

CONFIG = {
    "history_h": 24, "horizon_h": 6, "hidden": EMBEDDING_DIM, "batch": 64, "lr": 1e-3,
    "max_epochs": 25, "patience": 4, "seed": 7, "event_quantile": 0.95,
}
# lstm_1d_residual gives the 1-D baseline the same residual head as ST-GNN v0.1, so "beats the
# LSTM" measures architecture rather than output parameterisation (added after PR #12 review).
MODELS = ("stgnn", "graph_free", "time_free", "no_weather", "lstm_1d", "lstm_1d_residual")
# Architecture variants of the learned models (see models/stgnn.py). Baselines do not change.
VARIANTS = {"v0": {"gcn_layers": 2, "residual": False},
            "v0.1": {"gcn_layers": 1, "residual": True}}
# Test hours scored separately. Regimes are keyed on the first forecast hour (origin + 1).
STORM_HAZARD = 0.3  # any node's hazard above this
EVENT_WINDOWS = {"ditwah": ("2025-11-25T00:00", "2025-12-01T00:00")}  # Cyclone Ditwah, 25-30 Nov
# The ST-GNN is tested for a significant difference against each of these.
DM_BASELINES = ("persistence", "lstm_1d", "lstm_1d_residual")


def _seed(s: int) -> None:
    np.random.seed(s)
    torch.manual_seed(s)
    torch.use_deterministic_algorithms(True)


def _make(name: str, n_feat: int, a_hat: torch.Tensor, cfg: dict) -> nn.Module:
    h, hid = cfg["horizon_h"], cfg["hidden"]
    if name == "lstm_1d":
        return LSTM1D(h)
    if name == "lstm_1d_residual":
        return LSTM1D(h, residual=True)
    return STGNN(n_feat, h, a_hat, hid, use_graph=name != "graph_free",
                 use_time=name != "time_free", **VARIANTS[cfg["variant"]])


def _mask_weather(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    for f in ds.WEATHER_FEATURES:
        x[..., ds.FEATURES.index(f)] = 0.0
    return x


def _predict(model: nn.Module, x: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(torch.from_numpy(x[i:i + batch])).numpy())
    return np.concatenate(out)


def train_one(name: str, xtr, ytr, xva, yva, a_hat, cfg) -> tuple[nn.Module, dict]:
    _seed(cfg["seed"])
    model = _make(name, xtr.shape[-1], a_hat, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(cfg["seed"])
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr)
    best, best_state, bad, history = float("inf"), None, 0, []
    started = time.perf_counter()
    for _epoch in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        val = float(np.mean((_predict(model, xva) - yva) ** 2))
        history.append(val)
        if val < best - 1e-7:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, {"epochs": len(history), "best_val_mse": best, "val_curve": history,
                   "train_s": round(time.perf_counter() - started, 1),
                   "params": sum(p.numel() for p in model.parameters())}


def regimes(data, origins) -> dict[str, np.ndarray]:
    """Boolean masks over test origins: normal hours, storm hours, and each named event."""
    storm = data.components["hazard"][origins + 1].max(axis=1) > STORM_HAZARD
    stamps = data.timestamps[origins + 1]
    out = {"normal": ~storm, "storm": storm}
    for name, (start, end) in EVENT_WINDOWS.items():
        out[name] = (stamps >= start) & (stamps < end)
    return out


def evaluate(pred, yte, data, weighting, origins, tau) -> dict:
    v = data.v[weighting]
    a, b = data.split_index["test"]
    tr_a, tr_b = data.split_index["train"]
    scale = m.naive_scale(v[tr_a:tr_b])
    masks = regimes(data, origins)
    storm = masks["storm"]
    err = m.error_by_horizon(pred, yte)
    return {
        **err,
        "rmse_mean": float(np.mean(err["rmse"])),
        "mae_mean": float(np.mean(err["mae"])),
        **m.mase(pred, yte, scale),
        "mase_scale": scale,
        "regimes": {name: {"origins": int(k.sum()),
                           "rmse": float(np.sqrt(np.mean((pred[k] - yte[k]) ** 2))),
                           "mase": m.mase(pred[k], yte[k], scale)["mase"]}
                    for name, k in masks.items() if k.any()},
        "rmse_storm_hours": float(np.sqrt(np.mean((pred[storm] - yte[storm]) ** 2)))
        if storm.any() else None,
        "storm_origins": int(storm.sum()),
        "events": m.event_detection(pred, yte, tau),
        "onset": m.onset_lead_time(pred, v[a:b], origins - a, tau),
        "hit_at_1_h1": m.spatial_hit_rate(pred, yte, tau, 1, 0),
        "hit_at_3_h1": m.spatial_hit_rate(pred, yte, tau, 3, 0),
        "hit_at_1_h6": m.spatial_hit_rate(pred, yte, tau, 1, 5),
    }


def significance(preds: dict, yte, data, origins) -> dict:
    """Diebold-Mariano: ST-GNN against each baseline, over all test hours and per regime."""
    masks = {"all": np.ones(len(origins), bool), **regimes(data, origins)}
    return {base: {name: m.diebold_mariano(preds["stgnn"][k], preds[base][k], yte[k])
                   for name, k in masks.items() if k.sum() > 1}
            for base in DM_BASELINES if base in preds}


def score(res: dict, preds: dict, yte, data, wname, origins) -> None:
    for name, p in preds.items():
        res["models"][name]["test"] = evaluate(p, yte, data, wname, origins, res["tau"])
    res["significance_vs_stgnn"] = significance(preds, yte, data, origins)


def run_tag(variant: str, weather: str) -> str:
    """v0_1 for the NASA-trained paper model; v0_1_om for the Open-Meteo-trained live model."""
    return variant.replace(".", "_") + ("_om" if weather == "open_meteo" else "")


def rescore(out: Path, tag: str, data_root, weather: str = "nasa_power") -> int:
    path = out / f"forecast_{tag}_metrics.json"
    report = json.loads(path.read_text())
    data = ds.build(data_root, report.get("config", {}).get("weather", weather))
    for wname, res in report["results"].items():
        z = np.load(out / f"test_predictions_{tag}_{wname}.npz")
        preds = {name: z[name] for name in res["models"]}
        score(res, preds, z["true"], data, wname, z["origins"])
    path.write_text(json.dumps(report, indent=1))
    print(f"rescored {path}")
    return 0


def latency_ms(model: nn.Module, x: np.ndarray, reps: int = 200) -> dict:
    """Wall-clock for one streaming step (one window, all nodes), forward pass only."""
    one = torch.from_numpy(x[:1])
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(one)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(one)
            ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="metrics + predictions directory (tracked)")
    ap.add_argument("--ckpt", help="checkpoint directory (git-ignored); required unless --rescore")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--weightings", nargs="*", default=list(tgt.WEIGHTINGS))
    ap.add_argument("--variant", default="v0", choices=sorted(VARIANTS))
    ap.add_argument("--weather", default="nasa_power", choices=ds.WEATHER_SOURCES,
                    help="weather source to train on (ADR 0006); open_meteo tags outputs _om")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute test metrics from saved predictions; no training")
    args = ap.parse_args(argv)
    cfg = dict(CONFIG, variant=args.variant, weather=args.weather)
    tag = run_tag(args.variant, args.weather)
    if args.rescore:
        return rescore(Path(args.out), tag, args.data_root, args.weather)
    if not args.ckpt:
        ap.error("--ckpt is required when training")
    out, ckpt = Path(args.out), Path(args.ckpt)
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    data = ds.build(args.data_root, args.weather)
    check_invariants(data.graph)
    a_hat = torch.tensor(data.graph.normalized(), dtype=torch.float32)
    report: dict = {
        "config": cfg, "data_source": data.data_source, "splits": ds.SPLITS,
        "nodes": list(data.graph.node_ids), "features": list(ds.FEATURES),
        "target_assumptions": tgt.ASSUMPTIONS, "fitted": data.fitted, "results": {},
    }
    hist, hor = cfg["history_h"], cfg["horizon_h"]
    for wname in args.weightings:
        xtr, ytr, _ = ds.windows(data, "train", wname, hist, hor)
        xva, yva, _ = ds.windows(data, "val", wname, hist, hor)
        xte, yte, ote = ds.windows(data, "test", wname, hist, hor)
        a, b = data.split_index["train"]
        tau = float(np.quantile(data.v[wname][a:b], cfg["event_quantile"]))
        res: dict = {"tau": tau, "windows": {"train": len(xtr), "val": len(xva), "test": len(xte)},
                     "models": {}}
        preds = {"persistence": persistence(torch.from_numpy(xte), hor).numpy()}
        res["models"]["persistence"] = {"train": None}
        for name in MODELS:
            tr, va, te = (_mask_weather(z) for z in (xtr, xva, xte)) if name == "no_weather" \
                else (xtr, xva, xte)
            model, info = train_one(name, tr, ytr, va, yva, a_hat, cfg)
            preds[name] = _predict(model, te)
            res["models"][name] = {"train": info}
            if name == "stgnn":
                res["models"][name]["latency"] = latency_ms(model, te)
                torch.save(model.state_dict(), ckpt / f"stgnn_{tag}_{wname}.pt")
            print(f"[{wname}] {name}: {info['epochs']} epochs, val mse {info['best_val_mse']:.6f}",
                  flush=True)
        score(res, preds, yte, data, wname, ote)
        report["results"][wname] = res
        np.savez_compressed(out / f"test_predictions_{tag}_{wname}.npz", origins=ote, true=yte,
                            timestamps=data.timestamps[ote], **preds)
    (out / f"forecast_{tag}_metrics.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out / f'forecast_{tag}_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
