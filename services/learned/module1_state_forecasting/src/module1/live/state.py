"""Module 1 in live-weather mode (ADR 0006): today's inputs, today's forecast.

Weather is live (Open-Meteo, the source the live model was trained on). Everything electrical is
filled from the calibrated 2024-2025 dataset and labelled so:

* **Load** for an hour with no ledger month is the same calendar month of the latest ledger year,
  shaped hour by hour with the calibrated diurnal model and today's temperature, then scaled so the
  month's mean matches that ledger month. ``QUALITY_INTERPOLATED``.
* **Fragility** uses the ledger month before the hour's month when it exists. After the ledger
  ends it carries the last known value forward: ``QUALITY_STALE``. The last month we hold
  (December 2025) has the Eluvaitivu hybrid plant at about 3% of normal, so until CEB sends newer
  pages the live state treats that plant as still down.
* **Electrical readings** stay ``QUALITY_MISSING``.

Inputs are scaled with the constants fitted on the training split (``dataset.fit_site_stats``), so
a live hour reaches the model exactly as training hours did. The forecast is a pure function of
the weather rows and these frozen artifacts; its input hash makes that checkable.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from module1 import assemble as asm
from module1.data import load as loadmod
from module1.data.nasa_power import COLUMNS
from module1.data.scenarios import CORE_RATIO
from module1.models.stgnn import LSTM1D, STGNN
from module1.topology.snapshot import snapshot_from
from module1.topology.spec import build_spec
from module1.train import dataset as ds
from module1.train import target as tgt
from module1.train.run import CONFIG, VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[6]
ARTIFACTS = REPO_ROOT / "services" / "learned" / "module1_state_forecasting" / "artifacts"
WEATHER = "open_meteo"
WEIGHTING = "equal"
TAG = "v0_1_om"
MODEL_IDS = {"stgnn": f"stgnn_{TAG}_{WEIGHTING}",
             "lstm_1d_residual": f"lstm_1d_residual_{TAG}_{WEIGHTING}"}

QUALITY = {
    "weather": "QUALITY_INTERPOLATED",
    "load": "QUALITY_INTERPOLATED",
    "fragility_in_ledger": "QUALITY_INTERPOLATED",
    "fragility_carried": "QUALITY_STALE",
    "electrical": "QUALITY_MISSING",
    "time_and_topology": "QUALITY_OBSERVED",
}


class Frozen:
    """Everything fitted offline that the live path needs, loaded once per process."""

    def __init__(self) -> None:
        data = ds.build(None, WEATHER)
        self.graph = data.graph
        self.site_stats = data.fitted["site_stats"]
        self.a_hat = torch.tensor(data.graph.normalized(), dtype=torch.float32)
        a, b = data.split_index["train"]
        self.tau = float(np.quantile(data.v[WEIGHTING][a:b], CONFIG["event_quantile"]))

        root = REPO_ROOT / "data"
        tidy = root / "processed" / "ceb_generation_tidy.csv"
        params = json.loads((root / "processed" / "load_parameters.json").read_text())
        self.peakiness = params["calibration"]["solved_peakiness_exponent"]
        self.installed_kva = params["measured_inputs"]["installed_kva"]
        self.monthly = loadmod.read_monthly_island_energy(tidy)
        self.ledger_years = sorted({y for y, _, _ in self.monthly})
        nasa = loadmod.read_weather(root / "raw" / "nasa_power")
        self.median_c = {isl: statistics.median(t.values()) for isl, t in nasa.items()}
        self.nasa_temps = nasa
        self.hybrid = self._hybrid_months(tidy)
        # The contract state (what M2 scores) is scaled against each island's full NASA record,
        # exactly as module1.assemble builds it -- the distribution M2's real head was trained on.
        self.contract_refs = {
            isl: {k: np.array([float(r[k]) for r in ds._read(root / "raw" / "nasa_power"
                                                             / f"{isl}_hourly.csv")])
                  for k in ds.WEATHER}
            for isl in ds.ISLAND_ORDER}
        self.contract_peak = {isl: float(params["islands"][isl]["peak_kw"])
                              for isl in ds.ISLAND_ORDER}
        self.spec = build_spec(REPO_ROOT)

        n_feat = len(ds.FEATURES)
        h = CONFIG["horizon_h"]
        self.stgnn = STGNN(n_feat, h, self.a_hat, CONFIG["hidden"], **VARIANTS["v0.1"])
        self.stgnn.load_state_dict(torch.load(ARTIFACTS / f"{MODEL_IDS['stgnn']}.pt",
                                              map_location="cpu"))
        self.lstm = LSTM1D(h, residual=True)
        self.lstm.load_state_dict(torch.load(ARTIFACTS / f"{MODEL_IDS['lstm_1d_residual']}.pt",
                                             map_location="cpu"))
        self.stgnn.eval()
        self.lstm.eval()

    @staticmethod
    def _hybrid_months(tidy: Path) -> dict[str, float]:
        rows = ds._read(tidy)
        return {f"{r['year']}-{int(r['month_num']):02d}": float(r["units_kwh"])
                for r in rows if r["island_system"] == "Eluvaitivu-Hybrid"}


_FROZEN: Frozen | None = None


def frozen() -> Frozen:
    global _FROZEN
    if _FROZEN is None:
        _FROZEN = Frozen()
    return _FROZEN


# ------------------------------------------------------------ electrical fill-in

def ledger_month(fz: Frozen, stamp: str) -> tuple[str, bool]:
    """(year whose ledger month is used, True if that is the hour's own year)."""
    year = stamp[:4]
    if year in fz.ledger_years:
        return year, True
    return fz.ledger_years[-1], False


def live_load(fz: Frozen, island: str, stamps: list[str], temps: list[float | None]) -> np.ndarray:
    """Hourly island load for hours that may lie after the ledger, from the ledger's same month."""
    out = []
    for stamp, temp in zip(stamps, temps, strict=True):
        year, _ = ledger_month(fz, stamp)
        month = int(stamp[5:7])
        energy = fz.monthly[(year, month, island)]
        hours_in_month = loadmod.days_in_month(year, month) * 24
        # Mean shape of that ledger month, from the temperatures the load stage used for it.
        ref = [loadmod._shape(datetime.fromisoformat(s), t, fz.median_c[island], fz.peakiness)
               for s, t in fz.nasa_temps[island].items() if s.startswith(f"{year}-{month:02d}")]
        mean_shape = sum(ref) / len(ref)
        shape = loadmod._shape(datetime.fromisoformat(stamp), temp, fz.median_c[island],
                               fz.peakiness)
        out.append(energy / hours_in_month * shape / mean_shape)
    return np.array(out)


def live_decay(fz: Frozen, stamps: list[str]) -> tuple[np.ndarray, list[bool]]:
    """Hybrid ledger decay per hour (as in training) and whether each value is carried forward."""
    months = sorted(fz.hybrid)
    train_end = ds.SPLITS["train"][1][:7]
    median = float(np.median([fz.hybrid[m] for m in months if m <= train_end]))
    vals, carried = [], []
    for s in stamps:
        y, m = int(s[:4]), int(s[5:7])
        prev = f"{y - 1}-12" if m == 1 else f"{y}-{m - 1:02d}"
        known = prev in fz.hybrid
        use = prev if known else months[-1]
        vals.append(max(0.0, 1.0 - fz.hybrid[use] / median))
        carried.append(not known)
    return np.array(vals), carried


# ------------------------------------------------------------ the forecast

def current_hour() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:00")


def build_inputs(weather: dict[str, list[dict]], now: str) -> dict:
    """Model inputs for the 24 h ending at ``now``, plus the next 6 h of weather and load."""
    fz = frozen()
    hist, hor = CONFIG["history_h"], CONFIG["horizon_h"]
    first = next(iter(weather.values()))
    stamps = [r["timestamp_lst"] for r in first]
    if now not in stamps:
        raise ValueError(f"{now} is not in the fetched weather ({stamps[0]} .. {stamps[-1]})")
    t = stamps.index(now)
    if t - hist < 0 or t + hor >= len(stamps):
        raise ValueError(f"need {hist} h before and {hor} h after {now} in the fetched weather")

    site, hazard_isl, stress_isl, load_isl, weather_isl = {}, {}, {}, {}, {}
    for isl in ds.ISLAND_ORDER:
        rows = weather[isl]
        if [r["timestamp_lst"] for r in rows] != stamps:
            raise ValueError(f"{isl} weather is not aligned with the other islands")
        missing = [c for c in ds.WEATHER for r in rows[: t + hor + 1] if r[c] is None]
        if missing:
            raise ValueError(f"{isl}: missing weather values ({sorted(set(missing))})")
        x = {k: np.array([float(r[k]) for r in rows]) for k in ds.WEATHER}
        temps = [r["temp_2m_c"] for r in rows]
        load_kw = live_load(fz, isl, stamps, temps)
        st = fz.site_stats[isl]
        site[isl] = ds.site_features(x, load_kw, st)
        hazard_isl[isl] = tgt.hazard(x["wind_50m_ms"], x["precip_mm_hr"], st["hazard_ref"])
        kva = fz.installed_kva[f"{ledger_month(fz, now)[0]}/{isl}"]
        stress_isl[isl] = tgt.stress(load_kw, np.full(len(stamps), float(kva)))
        load_isl[isl] = load_kw
        weather_isl[isl] = x
    decay, carried = live_decay(fz, stamps)
    x_all, comp, v = ds.node_tensor(fz.graph, np.array(stamps), site, hazard_isl, stress_isl,
                                    decay)
    x_all[:, :, ds.FEATURES.index("vulnerability")] = v[WEIGHTING]
    window = x_all[t - hist + 1: t + 1][None].astype(np.float32)
    return {"fz": fz, "stamps": stamps, "t": t, "window": window, "v": v[WEIGHTING],
            "comp": comp, "load": load_isl, "weather": weather_isl,
            "fragility_carried": carried[t], "kva": {
                isl: fz.installed_kva[f"{ledger_month(fz, now)[0]}/{isl}"]
                for isl in ds.ISLAND_ORDER}}


def forecast(weather: dict[str, list[dict]], now: str, fetch_meta: dict | None = None,
             inputs: dict | None = None) -> dict:
    """One live step: inputs, both models, persistence, and everything the screen and M2-M4 need."""
    t0 = time.perf_counter()
    b = inputs or build_inputs(weather, now)
    fz, t, hor = b["fz"], b["t"], CONFIG["horizon_h"]
    xin = torch.from_numpy(b["window"])
    with torch.no_grad():
        t_inf = time.perf_counter()
        st_pred = fz.stgnn(xin)[0].numpy()
        latency_ms = (time.perf_counter() - t_inf) * 1000
        emb = fz.stgnn.embed(xin)[0].numpy()
        ls_pred = fz.lstm(xin)[0].numpy()
    g = fz.graph
    stamps = b["stamps"]
    nodes = []
    for j in range(g.n):
        nodes.append({
            "id": g.node_ids[j], "island": g.node_island[j], "asset": g.node_asset[j],
            "history": [round(float(x), 4) for x in b["v"][t - 23: t + 1, j]],
            "current": round(float(b["v"][t, j]), 4),
            "forecast": {
                "stgnn": [round(float(x), 4) for x in st_pred[j]],
                "lstm_1d_residual": [round(float(x), 4) for x in ls_pred[j]],
                "persistence": [round(float(b["v"][t, j]), 4)] * hor,
            },
            "components_now": {k: round(float(b["comp"][k][t, j]), 4) for k in b["comp"]},
            "embedding": [round(float(z), 4) for z in emb[j]],
        })
    islands = {}
    for isl in ds.ISLAND_ORDER:
        w = b["weather"][isl]
        span = slice(t - 23, t + hor + 1)
        islands[isl] = {
            "hours": stamps[span],
            "load_kw": [round(float(x), 2) for x in b["load"][isl][span]],
            "installed_kva": b["kva"][isl],
            **{k: [round(float(x), 2) for x in w[k][span]]
               for k in ("wind_50m_ms", "precip_mm_hr", "ghi_wh_m2", "temp_2m_c")},
        }
    quality = {
        "weather": QUALITY["weather"], "load": QUALITY["load"],
        "fragility": QUALITY["fragility_carried" if b["fragility_carried"] else
                             "fragility_in_ledger"],
        "electrical": QUALITY["electrical"], "time_and_topology": QUALITY["time_and_topology"],
    }
    return {
        "kind": "live",
        "issued_for": now,
        "horizon_hours": stamps[t + 1: t + 1 + hor],
        "tau": round(fz.tau, 4),
        "weighting": WEIGHTING,
        "models": MODEL_IDS,
        "weather_source": WEATHER,
        "ledger_year_used": ledger_month(fz, now)[0],
        "fragility_carried_forward": b["fragility_carried"],
        "input_sha256": hashlib.sha256(b["window"].tobytes()).hexdigest(),
        "latency_ms": round(latency_ms, 3),
        "nodes": nodes,
        "islands": islands,
        "above_tau_t1": [n["id"] for n in nodes if n["forecast"]["stgnn"][0] > fz.tau],
        "quality": quality,
        "fetch": fetch_meta,
        "generated_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def contract_state(fz: Frozen, inputs: dict, lag: int = 0) -> dict:
    """The M1 -> M2 contract fields for the live hour (or ``lag`` hours before it): 28 features
    per node, graph node order. Same formulas as ``module1.assemble``; only the values' source
    changes.
    """
    t = inputs["t"] - lag
    stamp = inputs["stamps"][t]
    temporal = asm._temporal(stamp)
    rows: list[list[float]] = []
    for isl in ds.ISLAND_ORDER:
        w = {k: float(inputs["weather"][isl][k][t]) for k in ds.WEATHER}
        ref = fz.contract_refs[isl]
        peak = fz.contract_peak[isl]
        load_kw = float(inputs["load"][isl][t])
        prev_kw = float(inputs["load"][isl][t - 1])
        cs_max = float(np.percentile(ref["ghi_clearsky_wh_m2"], 99.0))
        site = {
            "ghi_wh_m2_norm": min(w["ghi_wh_m2"] / cs_max, 1.5) if cs_max else 0.0,
            "clearsky_index": (min(w["ghi_wh_m2"] / w["ghi_clearsky_wh_m2"], 1.0)
                               if w["ghi_clearsky_wh_m2"] > 1.0 else 0.0),
            "wind_10m_ms_norm": asm._pnorm(w["wind_10m_ms"], ref["wind_10m_ms"]),
            "wind_50m_ms_norm": asm._pnorm(w["wind_50m_ms"], ref["wind_50m_ms"]),
            "temp_2m_c_norm": asm._zscore(w["temp_2m_c"], ref["temp_2m_c"]),
            "humidity_2m_pct_norm": w["humidity_2m_pct"] / 100.0,
            "precip_mm_hr_norm": asm._pnorm(w["precip_mm_hr"], ref["precip_mm_hr"]),
            "pressure_kpa_norm": asm._zscore(w["pressure_kpa"], ref["pressure_kpa"]),
            "load_kw_norm": load_kw / peak if peak else 0.0,
            "load_ramp_kw_per_h_norm": (load_kw - prev_kw) / peak if peak else 0.0,
        }
        for node in asm.ISLAND_ASSETS[isl]:
            topo = asm._topology_for_node(node)
            row = []
            for name in asm.FEATURE_NAMES:
                if name in asm._ELECTRICAL:
                    row.append(0.0)
                elif name == "pv_available_kw_norm":
                    row.append(site["ghi_wh_m2_norm"] if node == "pv" else 0.0)
                elif name in site:
                    row.append(site[name])
                elif name in temporal:
                    row.append(temporal[name])
                else:
                    row.append(topo[name])
            rows.append([round(v, 4) for v in row])
    mask = [asm.calibration_quality(n) for n in asm.FEATURE_NAMES]
    observed = sum(q == asm.QUALITY_OBSERVED for q in mask)
    return {"schema_version": str(asm.SCHEMA_VERSION), "feature_names": list(asm.FEATURE_NAMES),
            "node_features": rows, "quality_mask": mask,
            "observed_fraction": round(observed / len(mask), 4)}


def hybrid_down(fz: Frozen, stamp: str) -> tuple[bool, bool]:
    """(hybrid plant down that month?, carried forward from the last ledger month?)"""
    months = sorted(fz.hybrid)
    median = float(np.median(list(fz.hybrid.values())))
    key = stamp[:7]
    carried = key not in fz.hybrid
    ratio = fz.hybrid[months[-1] if carried else key] / median
    return ratio < CORE_RATIO, carried


def snapshots(fz: Frozen, inputs: dict, emitted_at: str | None = None) -> list[dict]:
    """GridStateSnapshot for now and each forecast hour, from the live load (for M4)."""
    t, hor = inputs["t"], CONFIG["horizon_h"]
    out = []
    for k in range(t, t + hor + 1):
        stamp = inputs["stamps"][k]
        down, carried = hybrid_down(fz, stamp)
        snap = snapshot_from(stamp, {isl: float(inputs["load"][isl][k]) for isl in ds.ISLAND_ORDER},
                             down, fz.spec, "live-weather", emitted_at, scenario="live-weather")
        snap["hybrid_status_carried_forward"] = carried
        out.append(snap)
    return out


def step(weather: dict[str, list[dict]], now: str, fetch_meta: dict | None = None) -> dict:
    """Everything one live hour produces in Module 1: forecast, M2 state, M4 snapshots."""
    inputs = build_inputs(weather, now)
    out = forecast(weather, now, fetch_meta, inputs)
    fz = inputs["fz"]
    out["contract_state"] = contract_state(fz, inputs)
    # M2's trigger confirms a competence drop over two consecutive hours (its config's
    # hysteresis); the previous hour lets it score the live hour as a stream, not a one-off.
    out["contract_state_prev"] = contract_state(fz, inputs, lag=1)
    out["snapshots"] = snapshots(fz, inputs)
    return out


def weather_rows_from_csv(path: Path) -> dict[str, list[dict]]:
    """Test and replay helper: a NASA-layout CSV (all islands) back into per-island rows."""
    import csv
    out: dict[str, list[dict]] = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["island"], []).append(
                {k: (None if r[k] == "" else (r[k] if k in ("island", "timestamp_lst")
                                              else float(r[k]))) for k in COLUMNS})
    return out
