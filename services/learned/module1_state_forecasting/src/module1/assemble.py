"""Assemble one Module 1 state vector from the offline calibration artifacts.

This is the *offline* half of M1's producer (ADR 0004): it turns the reconciled,
gated artifacts under ``data/`` into a contract-shaped ``StateRepresentation`` for a
chosen island / scenario / hour. It is deterministic, standard-library + NumPy only,
and does **not** touch the ST-GNN -- the learned 64-d ``node_embedding`` is left as a
flagged zero placeholder until a trained encoder exists.

    python -m module1.assemble --island eluvaitivu --scenario normal
    python -m module1.assemble --island eluvaitivu --scenario cyclone --json
    python -m module1.assemble --island delft --timestamp 2024-06-15T18:00

What is real here:

* ``load_kw_norm`` / ``load_ramp_kw_per_h_norm`` -- from
  ``data/processed/island_load_hourly.csv``, normalised by that island's own
  ``peak_kw`` in ``load_parameters.json``.
* the eight resource / meteorology features -- from
  ``data/raw/nasa_power/<Island>_hourly.csv``, normalised against that island's
  own two-year record.
* ``hour_sin/cos``, ``doy_sin/cos`` -- computed from ``timestamp_lst``. Genuinely
  ``QUALITY_OBSERVED``.
* the topology block -- from ``ISLAND_ASSETS`` below, derived from the CEB Jaffna
  ledger (which islands run a hybrid plant vs. diesel only).
* ``out_of_distribution`` -- read from ``data/processed/scenario_library.json``
  when a labelled window covers the chosen hour.

What is not real:

* the five ``electrical`` features -- the offline path has no power-flow simulator,
  so they are ``0.0`` and ``QUALITY_MISSING``. The contract says this is expected:
  the mask is a floor, the runtime simulator fills them per step.
* ``node_embedding`` -- zeros, ``has_embedding = false``.
* the node set is the island's generation assets plus a bus, not a full
  single-line diagram. Real per-bus topology arrives with the ST-GNN.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from metacore_contracts.state_schema import (
    EMBEDDING_DIM,
    FEATURE_NAMES,
    SCHEMA_VERSION,
    calibration_quality,
)

QUALITY_OBSERVED = "QUALITY_OBSERVED"
QUALITY_INTERPOLATED = "QUALITY_INTERPOLATED"
QUALITY_MISSING = "QUALITY_MISSING"

ScenarioName = Literal["normal", "cyclone", "blackout"]

# Frontend id  ->  the name used in the data files.
ISLAND_KEYS: dict[str, str] = {
    "eluvaitivu": "Eluvaitivu",
    "analaitivu": "Analaitivu",
    "nainativu": "Nainativu",
    "delft": "Delft-Neduntivu",
}

# Nodes per island: a bus plus the generation assets the CEB Jaffna ledger records.
# Only Eluvaitivu runs a hybrid (PV + storage) plant across 2024-2025; the other three
# islands are diesel-only. node[0] is always the bus -- that is what the dashboard shows.
ISLAND_ASSETS: dict[str, tuple[str, ...]] = {
    "Eluvaitivu": ("bus", "diesel", "pv", "bess"),
    "Analaitivu": ("bus", "diesel"),
    "Nainativu": ("bus", "diesel"),
    "Delft-Neduntivu": ("bus", "diesel"),
}

# A reference hour for the "normal" preset: mid-June, early evening (the load peak
# window), well away from any labelled scenario. Overridable with --timestamp.
DEFAULT_TIMESTAMP = "2024-06-15T18:00"

_ELECTRICAL = ("p_kw_norm", "q_kvar_norm", "voltage_pu", "soc_fraction", "asset_online")


def _repo_root() -> Path:
    # assemble.py -> module1 -> src -> module1_state_forecasting -> learned -> services -> repo root
    return Path(__file__).resolve().parents[5]


def _resolve_data_root(explicit: str | Path | None) -> Path:
    """Prefer the real DVC artifacts; fall back to the synthetic stand-in set."""
    if explicit is not None:
        return Path(explicit)
    root = _repo_root()
    real = root / "data"
    if (real / "processed" / "island_load_hourly.csv").exists():
        return real
    synth = root / ".synthetic"
    if (synth / "island_load_hourly.csv").exists():
        return synth
    return real  # let the open() failure below name the missing file


def _load_layout(data_root: Path) -> tuple[Path, Path, Path, Path]:
    """(load_csv, load_params_json, nasa_dir, scenario_json) for real or .synthetic roots."""
    if (data_root / "processed").is_dir():
        return (
            data_root / "processed" / "island_load_hourly.csv",
            data_root / "processed" / "load_parameters.json",
            data_root / "raw" / "nasa_power",
            data_root / "processed" / "scenario_library.json",
        )
    # .synthetic/ is flat
    return (
        data_root / "island_load_hourly.csv",
        data_root / "load_parameters.json",
        data_root / "nasa_power",
        data_root / "scenario_library.json",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _zscore(x: float, col: np.ndarray) -> float:
    sd = float(col.std())
    return 0.0 if sd == 0.0 else (x - float(col.mean())) / sd


def _pnorm(x: float, col: np.ndarray, pct: float = 99.0, cap: float = 1.5) -> float:
    ref = float(np.percentile(col, pct))
    return 0.0 if ref == 0.0 else min(x / ref, cap)


def _nearest_row(rows: list[dict[str, str]], stamp: str) -> tuple[int, dict[str, str]]:
    for i, r in enumerate(rows):
        if r["timestamp_lst"] == stamp:
            return i, r
    target = datetime.fromisoformat(stamp)
    idx = min(
        range(len(rows)),
        key=lambda i: abs(datetime.fromisoformat(rows[i]["timestamp_lst"]) - target),
    )
    return idx, rows[idx]


def _scenario_window(scenario_json: Path, island: str, stamp: str) -> dict | None:
    """The labelled scenario whose [start_month, end_month] covers `stamp` for `island`."""
    if not scenario_json.exists():
        return None
    lib = json.loads(scenario_json.read_text())
    month = stamp[:7]
    for sc in lib.get("scenarios", []):
        if sc.get("island") != island:
            continue
        if sc["start_month"] <= month <= sc["end_month"]:
            return sc
    return None


def _cyclone_hour(weather: list[dict[str, str]]) -> str:
    """The lowest-pressure hour in the record -- the closest thing to a cyclone the
    NASA POWER series contains. A falling surface pressure is the classic precursor."""
    idx = min(range(len(weather)), key=lambda i: float(weather[i]["pressure_kpa"]))
    return weather[idx]["timestamp_lst"]


def _temporal(stamp: str) -> dict[str, float]:
    dt = datetime.fromisoformat(stamp)
    hour = dt.hour + dt.minute / 60.0
    doy = dt.timetuple().tm_yday
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "doy_sin": math.sin(2 * math.pi * (doy - 1) / 365.0),
        "doy_cos": math.cos(2 * math.pi * (doy - 1) / 365.0),
    }


def _topology_for_node(node: str) -> dict[str, float]:
    is_flags = {f"is_{t}": 0.0 for t in ("bus", "pv", "wind", "bess", "diesel", "load")}
    if f"is_{node}" in is_flags:
        is_flags[f"is_{node}"] = 1.0
    return {
        # System base is the island LV nominal, so the bus sits at 1.0 p.u.
        "nominal_kv_norm": 1.0,
        # The island bus feeds critical load; the standalone gensets do not.
        "critical_load": 1.0 if node == "bus" else 0.0,
        **is_flags,
    }


def assemble(
    island_id: str,
    scenario: ScenarioName = "normal",
    timestamp: str | None = None,
    data_root: str | Path | None = None,
) -> dict:
    """Build one state. Returns a JSON-serialisable dict (see module docstring)."""
    started = time.perf_counter()
    if island_id not in ISLAND_KEYS:
        raise ValueError(f"unknown island {island_id!r}; have {sorted(ISLAND_KEYS)}")
    island = ISLAND_KEYS[island_id]

    root = _resolve_data_root(data_root)
    load_csv, load_params_json, nasa_dir, scenario_json = _load_layout(root)
    data_source = "real" if root.name == "data" else "synthetic"

    weather_all = _read_csv(nasa_dir / f"{island}_hourly.csv")
    load_all = [r for r in _read_csv(load_csv) if r["island"] == island]
    if not weather_all or not load_all:
        raise RuntimeError(f"no rows for {island} under {root}")

    # Choose the hour.
    if timestamp is not None:
        stamp = timestamp
    elif scenario == "cyclone":
        stamp = _cyclone_hour(weather_all)
    else:
        stamp = DEFAULT_TIMESTAMP

    w_idx, w = _nearest_row(weather_all, stamp)
    l_idx, ld = _nearest_row(load_all, stamp)
    stamp = w["timestamp_lst"]

    # Per-island normalisation references, from the island's own two-year record.
    wcol = {
        k: np.array([float(r[k]) for r in weather_all])
        for k in ("ghi_wh_m2", "ghi_clearsky_wh_m2", "wind_10m_ms", "wind_50m_ms",
                  "temp_2m_c", "humidity_2m_pct", "precip_mm_hr", "pressure_kpa")
    }
    params = json.loads(load_params_json.read_text())
    peak_kw = float(params["islands"][island]["peak_kw"])

    ghi = float(w["ghi_wh_m2"])
    ghi_cs = float(w["ghi_clearsky_wh_m2"])
    clearsky_max = float(np.percentile(wcol["ghi_clearsky_wh_m2"], 99.0))
    load_kw = float(ld["load_kw"])
    prev_kw = float(load_all[l_idx - 1]["load_kw"]) if l_idx > 0 else load_kw

    site = {
        "ghi_wh_m2_norm": min(ghi / clearsky_max, 1.5) if clearsky_max else 0.0,
        "clearsky_index": min(ghi / ghi_cs, 1.0) if ghi_cs > 1.0 else 0.0,
        "wind_10m_ms_norm": _pnorm(float(w["wind_10m_ms"]), wcol["wind_10m_ms"]),
        "wind_50m_ms_norm": _pnorm(float(w["wind_50m_ms"]), wcol["wind_50m_ms"]),
        "temp_2m_c_norm": _zscore(float(w["temp_2m_c"]), wcol["temp_2m_c"]),
        "humidity_2m_pct_norm": float(w["humidity_2m_pct"]) / 100.0,
        "precip_mm_hr_norm": _pnorm(float(w["precip_mm_hr"]), wcol["precip_mm_hr"]),
        "pressure_kpa_norm": _zscore(float(w["pressure_kpa"]), wcol["pressure_kpa"]),
        "load_kw_norm": load_kw / peak_kw if peak_kw else 0.0,
        "load_ramp_kw_per_h_norm": (load_kw - prev_kw) / peak_kw if peak_kw else 0.0,
    }
    temporal = _temporal(stamp)
    degraded = scenario == "blackout"

    nodes = ISLAND_ASSETS[island]
    node_features: list[list[float]] = []
    for node in nodes:
        topo = _topology_for_node(node)
        row: list[float] = []
        for name in FEATURE_NAMES:
            if name in _ELECTRICAL:
                row.append(0.0)  # QUALITY_MISSING offline -- no simulator
            elif name == "pv_available_kw_norm":
                row.append(site["ghi_wh_m2_norm"] if node == "pv" else 0.0)
            elif name == "soc_fraction":
                row.append(0.0)
            elif name in site:
                row.append(site[name])
            elif name in temporal:
                row.append(0.0 if degraded else temporal[name])
            elif name in topo:
                row.append(topo[name])
            else:  # unreachable while FEATURE_NAMES matches the six groups
                raise KeyError(name)
        node_features.append([round(v, 4) for v in row])

    quality_mask = [
        QUALITY_MISSING if (degraded and calibration_quality(n) == QUALITY_OBSERVED
                            and n in temporal)
        else calibration_quality(n)
        for n in FEATURE_NAMES
    ]
    observed = sum(1 for q in quality_mask if q == QUALITY_OBSERVED)

    window = _scenario_window(scenario_json, island, stamp)
    if scenario == "cyclone":
        scenario_id = f"nasa-power-pmin-{stamp[:4]}"
        ood = True
    elif window is not None:
        scenario_id = window["scenario_id"]
        ood = bool(window["out_of_distribution"])
    else:
        scenario_id = f"{island_id}-nominal-{stamp[:7]}"
        ood = False

    return {
        "island": island_id,
        "island_name": island,
        "scenario": scenario,
        "scenario_id": scenario_id,
        "timestamp_lst": stamp,
        "out_of_distribution": ood,
        "degraded": degraded,
        "schema_version": str(SCHEMA_VERSION),
        "embedding_dim": EMBEDDING_DIM,
        "has_embedding": False,
        "node_count": len(nodes),
        "node_names": list(nodes),
        "feature_names": list(FEATURE_NAMES),
        "node_features": node_features,
        "quality_mask": quality_mask,
        "observed_fraction": round(observed / len(FEATURE_NAMES), 4),
        "data_source": data_source,
        "generated_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def _print_table(state: dict) -> None:
    print(f"island        {state['island_name']}  ({state['island']})")
    print(f"scenario      {state['scenario']}  id={state['scenario_id']}")
    print(f"timestamp     {state['timestamp_lst']}")
    print(f"ood / degraded {state['out_of_distribution']} / {state['degraded']}")
    print(f"data source   {state['data_source']}")
    print(
        f"observed_frac {state['observed_fraction']}  "
        f"({sum(1 for q in state['quality_mask'] if q == QUALITY_OBSERVED)}/"
        f"{len(state['feature_names'])})"
    )
    print(f"nodes         {', '.join(state['node_names'])}")
    print()
    row0 = state["node_features"][0]
    width = max(len(n) for n in state["feature_names"])
    tags = {"QUALITY_OBSERVED": "obs", "QUALITY_INTERPOLATED": "int", "QUALITY_MISSING": "mis"}
    triples = zip(state["feature_names"], row0, state["quality_mask"], strict=True)
    for name, val, q in triples:
        print(f"  {name:<{width}}  {val:>9.4f}  {tags[q]}")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assemble one Module 1 state vector.")
    ap.add_argument("--island", required=True, choices=sorted(ISLAND_KEYS))
    ap.add_argument("--scenario", default="normal", choices=("normal", "cyclone", "blackout"))
    ap.add_argument("--timestamp", default=None, help="ISO local hour, e.g. 2024-06-15T18:00")
    ap.add_argument("--data-root", default=None, help="override data/ (or a .synthetic dir)")
    ap.add_argument("--json", action="store_true", help="emit the state as JSON on stdout")
    args = ap.parse_args(list(argv) if argv is not None else None)

    state = assemble(args.island, args.scenario, args.timestamp, args.data_root)
    if args.json:
        json.dump(state, sys.stdout)
        sys.stdout.write("\n")
    else:
        _print_table(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
