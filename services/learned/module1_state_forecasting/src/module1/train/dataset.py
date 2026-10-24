"""Hourly node-feature tensor and the v0 vulnerability target, from the gated artifacts.

Builds ``X[T, N, F]`` and ``V[T, N]`` for the whole 2024-2025 record. Everything that is fitted --
normalisation statistics, hazard scaling bounds, the hybrid plant's median -- is fitted on the
**training split only** and then applied to all rows, so nothing about the validation or test
period reaches the model through its inputs or its label scale.

Split (event-based, by calendar, not random): train 2024-01 .. 2025-06, validation 2025-07 ..
2025-09, test 2025-10 .. 2025-12. The test quarter holds both real events in the record -- Cyclone
Ditwah (late November 2025) and the Eluvaitivu hybrid decay (Oct-Dec 2025) -- so neither is ever
seen in training. Windows are cut wholly inside one split, which leaves a natural embargo of
``history + horizon`` hours at each boundary.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from module1.assemble import _load_layout, _resolve_data_root
from module1.graph.build import ISLAND_ORDER, MicrogridGraph, build_graph
from module1.train import target as tgt

SPLITS = {
    "train": ("2024-01-01T00:00", "2025-06-30T23:00"),
    "val": ("2025-07-01T00:00", "2025-09-30T23:00"),
    "test": ("2025-10-01T00:00", "2025-12-31T23:00"),
}

WEATHER = ("ghi_wh_m2", "ghi_clearsky_wh_m2", "wind_10m_ms", "wind_50m_ms", "temp_2m_c",
           "humidity_2m_pct", "precip_mm_hr", "pressure_kpa")

# Model input columns. The 28-feature contract minus the five electrical features (all
# QUALITY_MISSING offline -- constant zeros carry no signal), plus fragility and the lagged label.
FEATURES: tuple[str, ...] = (
    "ghi_wh_m2_norm", "clearsky_index", "wind_10m_ms_norm", "wind_50m_ms_norm",
    "pv_available_kw_norm", "temp_2m_c_norm", "humidity_2m_pct_norm", "precip_mm_hr_norm",
    "pressure_kpa_norm", "load_kw_norm", "load_ramp_kw_per_h_norm",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "nominal_kv_norm", "critical_load", "is_bus", "is_pv", "is_wind", "is_bess", "is_diesel",
    "is_load", "fragility", "vulnerability",
)
WEATHER_FEATURES = FEATURES[:9]
GROUPS = {
    "weather": WEATHER_FEATURES,
    "demand": ("load_kw_norm", "load_ramp_kw_per_h_norm"),
    "temporal": ("hour_sin", "hour_cos", "doy_sin", "doy_cos"),
    "topology": FEATURES[15:23],
    "fragility": ("fragility",),
    "history": ("vulnerability",),
}


@dataclass
class ForecastData:
    timestamps: np.ndarray            # (T,) ISO strings
    x: np.ndarray                     # (T, N, F) float32
    v: dict[str, np.ndarray]          # weighting -> (T, N)
    components: dict[str, np.ndarray]  # "hazard" (T, N), "fragility" (T, N), "stress" (T, N)
    graph: MicrogridGraph
    split_index: dict[str, tuple[int, int]]
    data_source: str
    fitted: dict = field(default_factory=dict)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _hybrid_decay_lagged(tidy_csv: Path, stamps: np.ndarray, train_end: str) -> np.ndarray:
    """Per hour: 1 - (previous month's hybrid energy / training-period median)."""
    rows = [r for r in _read(tidy_csv) if r["island_system"] == "Eluvaitivu-Hybrid"]
    monthly = {f"{r['year']}-{int(r['month_num']):02d}": float(r["units_kwh"]) for r in rows}
    train_months = [m for m in monthly if m <= train_end[:7]]
    median = float(np.median([monthly[m] for m in train_months]))
    months = sorted(monthly)
    prev = {m: months[i - 1] if i else None for i, m in enumerate(months)}
    decay = {m: (0.0 if prev[m] is None else max(0.0, 1.0 - monthly[prev[m]] / median))
             for m in months}
    return np.array([decay[s[:7]] for s in stamps])


def fit_site_stats(x: dict[str, np.ndarray], load_kw: np.ndarray, tr: slice) -> dict:
    """Every scaling constant of one island's features, fitted on the training rows only.

    Training and the live worker both apply these through ``site_features``, so a live hour is
    scaled exactly as the hours the model learned from. Plain floats, so they serialise.
    """
    def p99(k: str) -> float:
        return float(np.percentile(x[k][tr], 99.0)) or 1.0

    def moments(k: str) -> list[float]:
        return [float(x[k][tr].mean()), float(x[k][tr].std()) or 1.0]

    return {
        "cs99": float(np.percentile(x["ghi_clearsky_wh_m2"][tr], 99.0)),
        "p99": {k: p99(k) for k in ("wind_10m_ms", "wind_50m_ms", "precip_mm_hr")},
        "moments": {k: moments(k) for k in ("temp_2m_c", "pressure_kpa")},
        "peak_kw": float(load_kw[tr].max()),
        "hazard_ref": tgt.hazard_reference(x["wind_50m_ms"][tr], x["precip_mm_hr"][tr]),
    }


def site_features(x: dict[str, np.ndarray], load_kw: np.ndarray, st: dict) -> dict[str, np.ndarray]:
    """The per-island model inputs for any run of consecutive hours, scaled by ``st``."""
    def zs(k: str) -> np.ndarray:
        mean, sd = st["moments"][k]
        return (x[k] - mean) / sd

    peak = st["peak_kw"]
    ghi_n = np.minimum(x["ghi_wh_m2"] / st["cs99"], 1.5)
    return {
        "ghi_wh_m2_norm": ghi_n,
        "clearsky_index": np.where(x["ghi_clearsky_wh_m2"] > 1.0,
                                   np.minimum(x["ghi_wh_m2"]
                                              / np.maximum(x["ghi_clearsky_wh_m2"], 1.0), 1.0),
                                   0.0),
        "wind_10m_ms_norm": np.minimum(x["wind_10m_ms"] / st["p99"]["wind_10m_ms"], 1.5),
        "wind_50m_ms_norm": np.minimum(x["wind_50m_ms"] / st["p99"]["wind_50m_ms"], 1.5),
        "pv_available_kw_norm": ghi_n,
        "temp_2m_c_norm": zs("temp_2m_c"),
        "humidity_2m_pct_norm": x["humidity_2m_pct"] / 100.0,
        "precip_mm_hr_norm": np.minimum(x["precip_mm_hr"] / st["p99"]["precip_mm_hr"], 1.5),
        "pressure_kpa_norm": zs("pressure_kpa"),
        "load_kw_norm": load_kw / peak,
        "load_ramp_kw_per_h_norm": np.diff(load_kw, prepend=load_kw[0]) / peak,
    }


def temporal_features(stamps: np.ndarray) -> dict[str, np.ndarray]:
    hours = np.array([int(s[11:13]) for s in stamps])
    doy = np.array([np.datetime64(s[:10]).astype("datetime64[D]").item().timetuple().tm_yday
                    for s in stamps])
    return {
        "hour_sin": np.sin(2 * np.pi * hours / 24), "hour_cos": np.cos(2 * np.pi * hours / 24),
        "doy_sin": np.sin(2 * np.pi * (doy - 1) / 365),
        "doy_cos": np.cos(2 * np.pi * (doy - 1) / 365),
    }


def node_tensor(graph: MicrogridGraph, stamps: np.ndarray, site: dict, hazard_isl: dict,
                stress_isl: dict, decay: np.ndarray):
    """(x[T, N, F], components, v[weighting]) from per-island features and target terms."""
    t_count, n = len(stamps), graph.n
    temporal = temporal_features(stamps)
    comp = {k: np.zeros((t_count, n)) for k in ("hazard", "fragility", "stress")}
    v = {k: np.zeros((t_count, n)) for k in tgt.WEIGHTINGS}
    x_all = np.zeros((t_count, n, len(FEATURES)), dtype=np.float32)
    col = {f: i for i, f in enumerate(FEATURES)}
    for j in range(n):
        isl, asset = graph.node_island[j], graph.node_asset[j]
        frag = tgt.fragility(asset, decay)
        comp["hazard"][:, j] = tgt.EXPOSURE[asset] * hazard_isl[isl]
        comp["fragility"][:, j] = frag
        comp["stress"][:, j] = stress_isl[isl]
        for k in tgt.WEIGHTINGS:
            v[k][:, j] = tgt.combine(hazard_isl[isl], frag, stress_isl[isl], asset, k)
        for f, arr in site[isl].items():
            if f == "pv_available_kw_norm" and asset != "pv":
                continue
            x_all[:, j, col[f]] = arr
        for f, arr in temporal.items():
            x_all[:, j, col[f]] = arr
        x_all[:, j, col["nominal_kv_norm"]] = 1.0
        x_all[:, j, col["critical_load"]] = 1.0 if asset == "bus" else 0.0
        if f"is_{asset}" in col:
            x_all[:, j, col[f"is_{asset}"]] = 1.0
        x_all[:, j, col["fragility"]] = frag
    return x_all, comp, v


# Weather sources the forecaster can be trained on (ADR 0006). Both are written in the NASA POWER
# file layout, so only the directory changes. The target's hazard term is computed from the same
# weather, so a model trained on one source predicts that source's target.
WEATHER_SOURCES = ("nasa_power", "open_meteo")


def build(data_root: str | Path | None = None, weather_source: str = "nasa_power") -> ForecastData:
    if weather_source not in WEATHER_SOURCES:
        raise ValueError(f"weather_source {weather_source!r} not in {WEATHER_SOURCES}")
    root = _resolve_data_root(data_root)
    load_csv, params_json, nasa_dir, _ = _load_layout(root)
    weather_dir = nasa_dir.parent / weather_source
    if not weather_dir.is_dir():
        raise FileNotFoundError(f"no {weather_source} weather at {weather_dir}")
    tidy_csv = (root / "processed" / "ceb_generation_tidy.csv" if (root / "processed").is_dir()
                else root / "ceb_generation_tidy.csv")
    params = json.loads(params_json.read_text())
    graph = build_graph()

    weather = {isl: _read(weather_dir / f"{isl}_hourly.csv") for isl in ISLAND_ORDER}
    stamps = np.array([r["timestamp_lst"] for r in weather[ISLAND_ORDER[0]]])
    for isl in ISLAND_ORDER:
        if [r["timestamp_lst"] for r in weather[isl]] != list(stamps):
            raise ValueError(f"{isl} weather is not aligned to the common hourly grid")

    loads: dict[str, np.ndarray] = {}
    for isl in ISLAND_ORDER:
        by_t = {r["timestamp_lst"]: float(r["load_kw"]) for r in _read(load_csv)
                if r["island"] == isl}
        loads[isl] = np.array([by_t[s] for s in stamps])

    split_index = {}
    for name, (a, b) in SPLITS.items():
        idx = np.where((stamps >= a) & (stamps <= b))[0]
        split_index[name] = (int(idx[0]), int(idx[-1]) + 1)
    tr0, tr1 = split_index["train"]

    w = {isl: {k: np.array([float(r[k]) for r in weather[isl]]) for k in WEATHER}
         for isl in ISLAND_ORDER}

    fitted: dict = {"hazard_ref": {}, "norm": {}, "peak_kw": {}, "site_stats": {}}
    site: dict[str, dict[str, np.ndarray]] = {}
    hazard_isl: dict[str, np.ndarray] = {}
    stress_isl: dict[str, np.ndarray] = {}
    for isl in ISLAND_ORDER:
        st = fit_site_stats(w[isl], loads[isl], slice(tr0, tr1))
        fitted["site_stats"][isl] = st
        fitted["peak_kw"][isl] = st["peak_kw"]
        fitted["hazard_ref"][isl] = st["hazard_ref"]
        site[isl] = site_features(w[isl], loads[isl], st)
        hazard_isl[isl] = tgt.hazard(w[isl]["wind_50m_ms"], w[isl]["precip_mm_hr"],
                                     st["hazard_ref"])
        kva = np.array([params["measured_inputs"]["installed_kva"][f"{s[:4]}/{isl}"]
                        for s in stamps])
        stress_isl[isl] = tgt.stress(loads[isl], kva)

    decay = _hybrid_decay_lagged(tidy_csv, stamps, SPLITS["train"][1])
    x_all, comp, v = node_tensor(graph, stamps, site, hazard_isl, stress_isl, decay)
    # "vulnerability" (the lagged label) is filled per weighting at window time; see windows().

    if not np.isfinite(x_all).all():
        raise ValueError("non-finite values in the feature tensor")
    return ForecastData(stamps, x_all, v, comp, graph, split_index,
                        "real" if root.name == "data" else "synthetic", fitted)


def windows(data: ForecastData, split: str, weighting: str, history: int, horizon: int,
            stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X[B, L, N, F], Y[B, N, H], origin_index[B]) with windows wholly inside ``split``."""
    a, b = data.split_index[split]
    x = data.x.copy()
    x[:, :, FEATURES.index("vulnerability")] = data.v[weighting]
    origins = np.arange(a + history - 1, b - horizon, stride)
    xs = np.stack([x[t - history + 1: t + 1] for t in origins]).astype(np.float32)
    ys = np.stack([data.v[weighting][t + 1: t + 1 + horizon].T for t in origins]).astype(np.float32)
    return xs, ys, origins
