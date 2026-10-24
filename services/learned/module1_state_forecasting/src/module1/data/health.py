"""Data health: every calibration gate run now, plus the facts a reviewer would ask about.

The dashboard's Data health page calls this through the gateway. Each gate is the same function
its DVC stage runs (``validate.check``, ``nasa_power.check``, ``open_meteo.validate``,
``load.check``, ``scenarios.check``), so the page cannot report a pass the pipeline would fail.

What the page may show is limited on purpose. The CEB ledger is academic-use data, and the
hosting approval request promises the dashboard never displays it: ledger facts here are counts
and pass/fail only -- no energy, fuel, oil or cost figure. Load statistics are derived outputs.

    python -m module1.data.health [--json]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from module1.data import load, nasa_power, open_meteo, scenarios, validate

REPO_ROOT = Path(__file__).resolve().parents[6]
DATA = REPO_ROOT / "data"
PROCESSED = DATA / "processed"
TIDY = PROCESSED / "ceb_generation_tidy.csv"
SCHEMA = (REPO_ROOT / "packages" / "contracts" / "python" / "metacore_contracts" / "schema"
          / "module1_state_v1.json")
COMPARE_COLUMNS = ("ghi_wh_m2", "wind_50m_ms", "temp_2m_c", "precip_mm_hr")


def _file(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(REPO_ROOT)), "present": False}
    stat = path.stat() if path.is_file() else None
    files = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.is_file())
    newest = max(p.stat().st_mtime for p in files) if files else None
    return {"path": str(path.relative_to(REPO_ROOT)), "present": True,
            "bytes": stat.st_size if stat else sum(p.stat().st_size for p in files),
            "files": len(files),
            "modified": datetime.fromtimestamp(newest).isoformat(timespec="seconds")
            if newest else None}


def _gate(stage_id: str, title: str, what: str, fn, inputs: list[Path]) -> dict:
    t0 = time.perf_counter()
    try:
        failures = fn()
        error = None
    except Exception as exc:  # noqa: BLE001 -- a crashing gate is reported, not raised
        failures, error = [], f"{type(exc).__name__}: {exc}"
    return {
        "id": stage_id, "title": title, "checks": what,
        "status": "error" if error else ("pass" if not failures else "fail"),
        "failures": failures[:10], "failure_count": len(failures), "error": error,
        "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        "inputs": [_file(p) for p in inputs],
    }


def gates() -> list[dict]:
    load_csv = PROCESSED / "island_load_hourly.csv"
    return [
        _gate("ceb_reconcile", "CEB ledger reconciliation",
              "Annual totals match CEB's independent printed summary; per-row fuel and "
              "marine-transport identities hold.",
              lambda: validate.check(TIDY), [TIDY]),
        _gate("nasa_power_check", "NASA POWER weather",
              "Every hour present, no fill values, no negative irradiance, wind or rain, "
              "dark at local midnight and lit at local midday.",
              lambda: nasa_power.check(DATA / "raw" / "nasa_power"), [DATA / "raw" / "nasa_power"]),
        _gate("open_meteo_check", "Open-Meteo weather (live source)",
              "The NASA POWER checks plus the pinned-model rule.",
              lambda: open_meteo.validate(DATA / "raw" / "open_meteo"),
              [DATA / "raw" / "open_meteo"]),
        _gate("load_check", "Hourly load construction",
              "Every month adds back to the ledger exactly, no peak above installed capacity, "
              "Nainativu's reported maximum demand reproduced, no row claims to be observed.",
              lambda: load.check(load_csv, TIDY), [load_csv, PROCESSED / "load_parameters.json"]),
        _gate("scenario_check", "Scenario library",
              "The library reproduces from the ledger, ids are unique, the Eluvaitivu hybrid "
              "decay is present and labelled out-of-distribution.",
              lambda: scenarios.check(PROCESSED, TIDY),
              [PROCESSED / "events.csv", PROCESSED / "scenario_library.json"]),
    ]


def ledger_facts() -> dict:
    with TIDY.open() as fh:
        rows = list(csv.DictReader(fh))
    return {
        "rows": len(rows),
        "systems": sorted({r["island_system"] for r in rows}),
        "years": sorted({r["year"] for r in rows}),
        "months_per_system": len(rows) // max(1, len({r["island_system"] for r in rows})),
        "raw_workbook_on_this_machine": (DATA / "external" / "ceb_jaffna"
                                         / "Generation_2024_2025.xlsx").exists(),
        "governance": "Academic use only. Not in the repository; never displayed.",
    }


def weather_facts(source: str) -> dict:
    m = json.loads((DATA / "raw" / source / "manifest.json").read_text())
    distinct = {k: v["count"] for k, v in m["grid"]["measured_distinct_series"].items()}
    return {
        "source": m["source"], "model": m.get("model"),
        "range": [m["requested_start"], m["requested_end"]],
        "hours_per_site": next(iter(m["sites"].values()))["rows"],
        "missing_values": {k: v["missing_values"] for k, v in m["sites"].items()},
        "sites": len(m["sites"]), "distinct_series": distinct,
        "cells_to_sites": m["grid"]["cells_to_sites"],
    }


def load_facts() -> dict:
    p = json.loads((PROCESSED / "load_parameters.json").read_text())
    return {"quality": p["quality"],
            "islands": {k: {f: v[f] for f in ("peak_kw", "mean_kw", "load_factor")}
                        for k, v in p["islands"].items()}}


def scenario_facts() -> dict:
    lib = json.loads((PROCESSED / "scenario_library.json").read_text())
    ood = [s for s in lib["scenarios"] if s.get("out_of_distribution")]
    return {"counts": lib["counts"], "rule": lib["rule"],
            "out_of_distribution": [{k: s.get(k) for k in ("scenario_id", "island", "plant",
                                                            "start_month", "end_month")}
                                    for s in ood]}


def feature_quality() -> dict:
    schema = json.loads(SCHEMA.read_text())
    spec = schema.get("features", {})
    counts: dict[str, int] = {}
    groups: dict[str, dict[str, int]] = {}
    for name in schema["feature_names"]:
        f = spec.get(name, {})
        q, g = f.get("calibration_quality", "UNKNOWN"), f.get("group", "other")
        counts[q] = counts.get(q, 0) + 1
        groups.setdefault(g, {}).setdefault(q, 0)
        groups[g][q] += 1
    return {"total": len(schema["feature_names"]), "by_quality": counts, "by_group": groups}


def _series(source: str, island: str, column: str) -> np.ndarray:
    with (DATA / "raw" / source / f"{island}_hourly.csv").open() as fh:
        return np.array([float(r[column]) if r[column] else np.nan for r in csv.DictReader(fh)])


def source_comparison() -> dict:
    """NASA POWER against Open-Meteo, per island and variable: agreement and level."""
    out: dict = {}
    for island in (s.key for s in nasa_power.ISLAND_SITES):
        per = {}
        for col in COMPARE_COLUMNS:
            a, b = _series("open_meteo", island, col), _series("nasa_power", island, col)
            if col == "precip_mm_hr":
                b = b / 24.0  # NASA stores a mm/day rate under this name (Data Analysis Report)
            lags = {}
            for lag in (-1, 0, 1):
                x = a[max(0, lag):len(a) + min(0, lag)]
                y = b[max(0, -lag):len(b) - max(0, lag)]
                ok = np.isfinite(x) & np.isfinite(y)
                lags[lag] = float(np.corrcoef(x[ok], y[ok])[0, 1])
            best = max(lags, key=lags.get)
            per[col] = {"corr_same_hour": round(lags[0], 3), "corr_best": round(lags[best], 3),
                        "best_lag_h": best,
                        "mean_open_meteo": round(float(np.nanmean(a)), 3),
                        "mean_nasa_power": round(float(np.nanmean(b)), 3)}
        rain_om, rain_nasa = (_series("open_meteo", island, "precip_mm_hr"),
                              _series("nasa_power", island, "precip_mm_hr") / 24.0)
        per["annual_rain_mm"] = {
            "open_meteo": [round(float(np.nansum(rain_om[:8784]))),
                           round(float(np.nansum(rain_om[8784:])))],
            "nasa_power": [round(float(np.nansum(rain_nasa[:8784]))),
                           round(float(np.nansum(rain_nasa[8784:])))],
        }
        out[island] = per
    return {"note": ("NASA precipitation divided by 24 (stored as a mm/day rate). Positive lag "
                     "means Open-Meteo's series matches NASA's shifted one hour later."),
            "islands": out}


def report() -> dict:
    t0 = time.perf_counter()
    g = gates()
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": {"gates": len(g), "passed": sum(x["status"] == "pass" for x in g),
                    "failed": sum(x["status"] != "pass" for x in g)},
        "gates": g,
        "ledger": ledger_facts(),
        "weather": {"nasa_power": weather_facts("nasa_power"),
                    "open_meteo": weather_facts("open_meteo")},
        "load": load_facts(),
        "scenarios": scenario_facts(),
        "features": feature_quality(),
        "source_comparison": source_comparison(),
        "generated_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = report()
    if args.json:
        print(json.dumps(r))
    else:
        for x in r["gates"]:
            print(f"{x['status'].upper():5s} {x['title']} ({x['duration_ms']} ms)")
    return 0 if r["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
