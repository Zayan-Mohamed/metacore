"""The dashboard's Live page in replay mode: the held-out test quarter, hour by hour, as if live.

Until the live worker runs (ADR 0006), the Live page steps through October-December 2025 with the
Open-Meteo-trained model's real stored forecasts. Nothing is recomputed in the browser except the
rolling scores. This module packs those forecasts, the realised target, the target's three
components and the hazard-driving weather into one committed JSON file, so a fresh clone can
replay without the git-ignored prediction archives.

Values are stored as integers in thousandths (``scale``) to keep the file near 1.5 MB.

    python -m module1.infer.replay build    # reads the local test_predictions archive
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from module1.train import dataset as ds
from module1.train.run import EVENT_WINDOWS, run_tag

REPO_ROOT = Path(__file__).resolve().parents[6]
REPORT_DIR = REPO_ROOT / "reports" / "data-analysis-report" / "m1"
WEIGHTING = "equal"
VARIANT, WEATHER = "v0.1", "open_meteo"
MODELS = ("stgnn", "lstm_1d_residual")  # persistence is the truth at the origin; not stored
SCALE = 1000
HISTORY_H = 24

# Quality of each input group in live-weather mode (ADR 0006, decision 1). The replay labels its
# inputs the way the live worker will, so the page does not change when the source does.
LIVE_QUALITY = {
    "weather": "QUALITY_INTERPOLATED",   # a numerical weather model's field, not a station
    "load": "QUALITY_INTERPOLATED",      # constructed from monthly ledger energy
    "fragility": "QUALITY_STALE",        # carried forward from the last ledger month
    "electrical": "QUALITY_MISSING",     # no telemetry exists
    "time_and_topology": "QUALITY_OBSERVED",
}

OUT = REPORT_DIR / f"replay_{run_tag(VARIANT, WEATHER)}_{WEIGHTING}.json"


def _ints(a: np.ndarray) -> list:
    return np.rint(np.asarray(a) * SCALE).astype(int).tolist()


def _weather(island: str, stamps: list[str]) -> dict[str, list[float | None]]:
    path = REPO_ROOT / "data" / "raw" / WEATHER / f"{island}_hourly.csv"
    with path.open() as fh:
        rows = {r["timestamp_lst"]: r for r in csv.DictReader(fh)}
    out: dict[str, list[float | None]] = {}
    for col in ("wind_50m_ms", "precip_mm_hr", "ghi_wh_m2"):
        out[col] = [round(float(rows[s][col]), 1) if rows[s][col] else None for s in stamps]
    return out


def build(report_dir: Path = REPORT_DIR) -> dict:
    tag = run_tag(VARIANT, WEATHER)
    z = np.load(report_dir / f"test_predictions_{tag}_{WEIGHTING}.npz")
    metrics = json.loads((report_dir / f"forecast_{tag}_metrics.json").read_text())
    res = metrics["results"][WEIGHTING]
    data = ds.build(None, WEATHER)

    origins = z["origins"]
    if not (np.diff(origins) == 1).all():
        raise ValueError("replay needs contiguous hourly origins")
    horizon = z["stgnn"].shape[-1]
    lo, hi = int(origins[0]) - HISTORY_H + 1, int(origins[-1]) + horizon + 1
    stamps = [str(s) for s in data.timestamps[lo:hi]]

    summary = {}
    for m in ("stgnn", "lstm_1d_residual", "persistence"):
        t = res["models"][m]["test"]
        summary[m] = {"rmse_mean": round(t["rmse_mean"], 4), "mase": round(t["mase"], 3),
                      "onset_recall": round(t["onset"]["onset_recall"], 3),
                      "ditwah_rmse": round(t["regimes"]["ditwah"]["rmse"], 4)}

    payload = {
        "kind": "replay",
        "note": ("Replay of the held-out test quarter (Oct-Dec 2025) with stored forecasts. "
                 "Not live: the live worker (ADR 0006) will serve the same shape from its "
                 "database."),
        "model": f"stgnn_{tag}_{WEIGHTING}",
        "variant": VARIANT,
        "weather_source": WEATHER,
        "weighting": WEIGHTING,
        "tau": round(res["tau"], 4),
        "horizon_h": int(horizon),
        "history_h": HISTORY_H,
        "scale": SCALE,
        "nodes": [{"id": n, "island": i, "asset": a} for n, i, a in
                  zip(data.graph.node_ids, data.graph.node_island, data.graph.node_asset,
                      strict=True)],
        "hours": stamps,
        "first_origin": HISTORY_H - 1,      # index into hours of the first forecast origin
        "truth": _ints(data.v[WEIGHTING][lo:hi]),
        "components": {k: _ints(data.components[k][lo:hi])
                       for k in ("hazard", "fragility", "stress")},
        "forecast": {m: _ints(z[m]) for m in MODELS},
        "weather": {isl: _weather(isl, stamps) for isl in sorted(set(data.graph.node_island))},
        "quality": LIVE_QUALITY,
        "events": {k: list(v) for k, v in EVENT_WINDOWS.items()},
        "test_summary": summary,
    }
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    payload = build()
    Path(args.out).write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.2f} MB, "
          f"{len(payload['hours'])} hours)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
