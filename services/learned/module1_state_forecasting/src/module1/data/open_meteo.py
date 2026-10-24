"""Stage: Open-Meteo hourly weather for the four Jaffna islands (ADR 0006, live-weather mode).

NASA POWER cannot feed a forecast for today: on 24 Oct 2026 its hourly meteorology ended about
2.5 days back and its hourly irradiance on 30 Jul 2026. Open-Meteo serves the current hour, and
its Historical Forecast API holds the archived forecasts of the same models back to 2024, so the
model is trained on the same product it sees live. This module fetches that history (and, for the
live worker, the latest hours) and writes it in **exactly the NASA POWER file layout** -- same
sites, same columns, same units -- so the feature code does not change with the source.

Mapping to the NASA columns, and what it costs
----------------------------------------------
* ``shortwave_radiation`` (W/m^2, mean of the preceding hour) -> ``ghi_wh_m2``: numerically equal.
* ``ghi_clearsky_wh_m2`` is **computed**, not fetched. Open-Meteo lists
  ``shortwave_radiation_clear_sky`` but returns no values for it (checked live and historical, four
  models). It is the Haurwitz clear-sky model averaged over the same preceding hour, from the NOAA
  solar-position equations. CERES's clear-sky product includes aerosols; Haurwitz does not, so the
  two sources' clear-sky values differ in level. Within one source the feature is consistent.
* ``wind_speed_80m`` -> ``wind_50m_ms``: there is no 50 m level. The 80 m value is moved to 50 m
  with the 1/7 power law (x 0.935). After per-site normalisation a constant factor changes nothing
  the model sees; it keeps the column's name true.
* km/h -> m/s, hPa -> kPa. ``precipitation`` is a true preceding-hour sum in mm, whereas the NASA
  column stores a mm/day rate (see the Data Analysis Report); within each source it is consistent.
* Times are Sri Lanka Standard Time (UTC+05:30). POWER's are local solar time, about 11 minutes
  earlier at these longitudes. The column keeps the name ``timestamp_lst`` so layouts match.

Measured against the NASA pull, 2024-2025 (Nainativu; Delft agrees)
-----------------------------------------------------------------
* **Three distinct series for four sites, for every variable** (ICON cells 9.75/9.625/9.5 N;
  Eluvaitivu and Analaitivu share one). POWER gives one to two. The graph now sees some real
  inter-island weather contrast.
* **Irradiance is stamped one hour later than POWER's**: correlation 0.93 at lag 0, 0.98 with
  Open-Meteo shifted back one hour (end-of-hour mean versus POWER's start-of-hour). Each source
  is consistent with itself and the model is trained per source, so no shift is applied.
* **ICON rains about twice as much**: about 2,500 mm a year at each site, against 1,146 and
  1,605 mm from POWER (corrected) and a regional climate near 1,200-1,300 mm. A forecast-model
  wet bias, reported rather than corrected; hazard scaling is fitted on each source's own
  training split.
* **Cyclone Ditwah is present**: 50 m wind peaks at 19.0 m/s on 29 Nov 2025 01:00, against
  POWER's 19.7 m/s twelve hours earlier.

One weather model is pinned (``MODEL``) rather than Open-Meteo's ``best_match``, which may switch
models over time and would silently shift the training distribution. The grid cell Open-Meteo
answers from is recorded per site, and the number of distinct series is measured, as for POWER.

Usage:
    python -m module1.data.open_meteo fetch    <out_dir> [--start YYYYMMDD] [--end YYYYMMDD]
                                                         [--model NAME]  (evaluation only)
    python -m module1.data.open_meteo validate <out_dir>
    python -m module1.data.open_meteo live     <out_csv> [--past-days N]
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

from module1.data.nasa_power import COLUMNS, ISLAND_SITES, Site, check, write_site_csv

HISTORICAL_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_API = "https://api.open-meteo.com/v1/forecast"
MODEL = "icon_seamless"
TIMEZONE = "Asia/Colombo"
UTC_OFFSET_H = 5.5

DEFAULT_START = "20240101"
DEFAULT_END = "20251231"

# Open-Meteo hourly variable -> (NASA-layout column, conversion). Order is irrelevant: COLUMNS
# fixes the file layout. Clear-sky GHI is not here because it is computed (see the docstring).
HUB_SHEAR = (50 / 80) ** (1 / 7)
VARIABLES: dict[str, tuple[str, float]] = {
    "shortwave_radiation": ("ghi_wh_m2", 1.0),
    "wind_speed_10m": ("wind_10m_ms", 1 / 3.6),
    "wind_speed_80m": ("wind_50m_ms", HUB_SHEAR / 3.6),
    "temperature_2m": ("temp_2m_c", 1.0),
    "relative_humidity_2m": ("humidity_2m_pct", 1.0),
    "precipitation": ("precip_mm_hr", 1.0),
    "surface_pressure": ("pressure_kpa", 0.1),
}

# Haurwitz (1945) clear-sky GHI, W/m^2: 1098 cos z exp(-0.057 / cos z).
HAURWITZ_A, HAURWITZ_B = 1098.0, 0.057
CLEARSKY_SAMPLES = 12  # 5-minute samples across the preceding hour


# ------------------------------------------------------------ clear sky

def cos_zenith(lat: float, lon: float, utc: datetime) -> float:
    """NOAA general solar position (Spencer's series), accurate to well under a degree."""
    doy = utc.timetuple().tm_yday
    hour = utc.hour + utc.minute / 60 + utc.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    true_solar_min = hour * 60 + eqtime + 4 * lon
    hour_angle = math.radians(true_solar_min / 4 - 180)
    phi = math.radians(lat)
    return math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(hour_angle)


def clearsky_wh_m2(site: Site, local_stamp: str) -> float:
    """Mean Haurwitz GHI over the hour ending at ``local_stamp`` (SLST), as Wh/m^2."""
    end = datetime.fromisoformat(local_stamp) - timedelta(hours=UTC_OFFSET_H)
    total = 0.0
    for k in range(CLEARSKY_SAMPLES):
        t = end - timedelta(minutes=60 * (k + 0.5) / CLEARSKY_SAMPLES)
        cz = cos_zenith(site.latitude, site.longitude, t)
        if cz > 0:
            total += HAURWITZ_A * cz * math.exp(-HAURWITZ_B / cz)
    return round(total / CLEARSKY_SAMPLES, 2)


# ------------------------------------------------------------ fetching

def _request(url: str, attempts: int = 5) -> list | dict:
    """Open-Meteo rate-limits by request weight (429); back off rather than drop a month."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                payload = json.load(response)
            if isinstance(payload, dict) and payload.get("error"):
                raise ValueError(f"Open-Meteo rejected the request: {payload.get('reason')}")
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < attempts - 1:
                time.sleep(2 ** attempt * 15)
    raise RuntimeError(f"Open-Meteo request failed after {attempts} attempts: {last}") from last


def _query(extra: dict, model: str = MODEL) -> str:
    return urllib.parse.urlencode({
        "latitude": ",".join(str(s.latitude) for s in ISLAND_SITES),
        "longitude": ",".join(str(s.longitude) for s in ISLAND_SITES),
        "hourly": ",".join(VARIABLES),
        "models": model,
        "timezone": TIMEZONE,
        **extra,
    })


def to_rows(site: Site, block: dict) -> list[dict]:
    """One location's ``hourly`` block -> rows in the NASA POWER layout."""
    hourly = block["hourly"]
    rows = []
    for i, stamp in enumerate(hourly["time"]):
        row = {"island": site.key, "timestamp_lst": stamp,
               "ghi_clearsky_wh_m2": clearsky_wh_m2(site, stamp)}
        for variable, (column, factor) in VARIABLES.items():
            value = hourly.get(variable, [None] * len(hourly["time"]))[i]
            # Missing stays missing: 0.0 would be a plausible-looking lie for rain or sunshine.
            row[column] = None if value is None else round(value * factor, 3)
        rows.append(row)
    return rows


def _months(start: str, end: str) -> list[tuple[str, str]]:
    """Calendar-month chunks: a two-year, four-site hourly request is too heavy for one call."""
    first = datetime.strptime(start, "%Y%m%d").date()
    last = datetime.strptime(end, "%Y%m%d").date()
    out, cur = [], first
    while cur <= last:
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        out.append((cur.isoformat(), min(last, nxt - timedelta(days=1)).isoformat()))
        cur = nxt
    return out


def fetch(out_dir: Path, start: str = DEFAULT_START, end: str = DEFAULT_END,
          pause_s: float = 2.0, model: str = MODEL) -> dict[str, list[dict]]:
    rows_by_site: dict[str, list[dict]] = {s.key: [] for s in ISLAND_SITES}
    cells: dict[str, list[float]] = {}
    for a, b in _months(start, end):
        query = _query({"start_date": a, "end_date": b}, model)
        payload = _request(f"{HISTORICAL_API}?{query}")
        blocks = payload if isinstance(payload, list) else [payload]
        for site, block in zip(ISLAND_SITES, blocks, strict=True):
            rows_by_site[site.key].extend(to_rows(site, block))
            cells[site.key] = [block["latitude"], block["longitude"]]
        hours = len(blocks[0]["hourly"]["time"])
        print(f"{a[:7]}: {hours} hours x {len(blocks)} sites", flush=True)
        time.sleep(pause_s)

    out_dir.mkdir(parents=True, exist_ok=True)
    for site in ISLAND_SITES:
        write_site_csv(rows_by_site[site.key], out_dir / f"{site.key}_hourly.csv")
    write_manifest(out_dir, start, end, rows_by_site, cells, model)
    return rows_by_site


def write_manifest(out_dir: Path, start: str, end: str, rows_by_site: dict[str, list[dict]],
                   cells: dict[str, list[float]], model: str = MODEL) -> dict:
    by_cell: dict[str, list[str]] = {}
    for key, (lat, lon) in cells.items():
        by_cell.setdefault(f"{lat},{lon}", []).append(key)
    distinct = {}
    for column in COLUMNS[2:]:
        groups: dict[tuple, list[str]] = {}
        for key, rows in rows_by_site.items():
            groups.setdefault(tuple(r[column] for r in rows), []).append(key)
        distinct[column] = {"count": len(groups), "groups": sorted(groups.values())}
    manifest = {
        "source": "Open-Meteo Historical Forecast API",
        "endpoint": HISTORICAL_API,
        "model": model,
        "timezone": f"{TIMEZONE} (UTC+05:30)",
        "temporal": "hourly",
        "requested_start": start,
        "requested_end": end,
        "variables": {v: {"column": c, "factor": f} for v, (c, f) in VARIABLES.items()},
        "computed": {"ghi_clearsky_wh_m2": "Haurwitz clear-sky GHI, mean of the preceding hour"},
        "grid": {"cells_to_sites": by_cell, "distinct_cells": len(by_cell),
                 "site_count": len(ISLAND_SITES), "measured_distinct_series": distinct},
        "sites": {
            s.key: {"latitude": s.latitude, "longitude": s.longitude,
                    "returned_cell": cells.get(s.key), "rows": len(rows_by_site[s.key]),
                    "missing_values": sum(1 for r in rows_by_site[s.key] for c in COLUMNS[2:]
                                          if r[c] is None),
                    "file": f"{s.key}_hourly.csv"}
            for s in ISLAND_SITES
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def fetch_live(past_days: int = 2, forecast_days: int = 1) -> tuple[dict[str, list[dict]], dict]:
    """The latest hours for the live worker: observed-so-far plus the model's next day.

    Returns rows per site and fetch metadata (endpoint, model, fetch time, returned cells), which
    the worker stores with the rows so every forecast can be traced to the weather it used.
    """
    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
    query = _query({"past_days": past_days, "forecast_days": forecast_days})
    payload = _request(f"{LIVE_API}?{query}")
    blocks = payload if isinstance(payload, list) else [payload]
    rows = {s.key: to_rows(s, b) for s, b in zip(ISLAND_SITES, blocks, strict=True)}
    meta = {"endpoint": LIVE_API, "model": MODEL, "fetched_at": fetched_at,
            "cells": {s.key: [b["latitude"], b["longitude"]]
                      for s, b in zip(ISLAND_SITES, blocks, strict=True)}}
    return rows, meta


def validate(out_dir: Path) -> list[str]:
    """POWER's invariants (hour count, no negatives, dark at midnight, lit at midday), plus the
    pinned-model rule: a pull from another model must not pass as this one."""
    failures = check(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    if manifest.get("model") != MODEL:
        failures.append(f"manifest model {manifest.get('model')!r}, expected {MODEL!r}")
    return failures


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in ("fetch", "validate", "live"):
        print(__doc__, file=sys.stderr)
        return 2
    command, target = argv[1], Path(argv[2])
    opts = {argv[i]: argv[i + 1] for i in range(3, len(argv) - 1) if argv[i].startswith("--")}

    if command == "fetch":
        fetch(target, opts.get("--start", DEFAULT_START), opts.get("--end", DEFAULT_END),
              model=opts.get("--model", MODEL))
        return 0
    if command == "live":
        rows, meta = fetch_live(int(opts.get("--past-days", 2)))
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            for site_rows in rows.values():
                w.writerows({k: ("" if r[k] is None else r[k]) for k in COLUMNS} for r in site_rows)
        print(f"{sum(map(len, rows.values()))} rows from {meta['model']} at {meta['fetched_at']}")
        return 0

    failures = validate(target)
    if failures:
        print(f"OPEN-METEO CHECK FAILED — {len(failures)} problem(s):", file=sys.stderr)
        for failure in failures[:25]:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"open_meteo OK — {len(ISLAND_SITES)} sites, model {MODEL}, "
          f"{date.today().isoformat()} check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
