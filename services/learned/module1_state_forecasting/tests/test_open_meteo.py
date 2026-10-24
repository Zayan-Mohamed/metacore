"""The Open-Meteo stage must produce the NASA POWER file layout, in NASA POWER units.

If a conversion slips (km/h left as m/s, hPa as kPa) every downstream feature shifts with no
error anywhere, so the conversions are pinned here against a canned response. No network.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from module1.data import open_meteo as om
from module1.data.nasa_power import COLUMNS, ISLAND_SITES

NAI = next(s for s in ISLAND_SITES if s.key == "Nainativu")


def block(times: list[str], **values: list) -> dict:
    hourly = {"time": times}
    for v in om.VARIABLES:
        hourly[v] = values.get(v, [1.0] * len(times))
    return {"latitude": 9.66608, "longitude": 79.877945, "hourly": hourly}


def test_units_are_converted_to_the_nasa_layout() -> None:
    b = block(["2024-03-01T12:00"], wind_speed_10m=[36.0], wind_speed_80m=[36.0],
              surface_pressure=[1010.0], shortwave_radiation=[800.0], precipitation=[2.5])
    (row,) = om.to_rows(NAI, b)
    assert row["wind_10m_ms"] == pytest.approx(10.0)
    assert row["wind_50m_ms"] == pytest.approx(10.0 * (50 / 80) ** (1 / 7), abs=1e-3)
    assert row["pressure_kpa"] == pytest.approx(101.0)
    assert row["ghi_wh_m2"] == 800.0 and row["precip_mm_hr"] == 2.5
    assert set(row) == set(COLUMNS)


def test_missing_values_stay_missing() -> None:
    (row,) = om.to_rows(NAI, block(["2024-03-01T12:00"], precipitation=[None]))
    assert row["precip_mm_hr"] is None


def test_clear_sky_is_dark_at_night_and_plausible_at_noon() -> None:
    assert om.clearsky_wh_m2(NAI, "2024-03-21T00:00") == 0.0
    noon = om.clearsky_wh_m2(NAI, "2024-03-21T13:00")       # hour ending 13:00 SLST ~ solar noon
    assert 900 < noon < 1100                                # near-overhead sun at 9.6 N
    assert om.clearsky_wh_m2(NAI, "2024-03-21T07:00") < noon / 2


def test_months_cover_the_range_without_gaps() -> None:
    chunks = om._months("20240115", "20240310")
    assert chunks == [("2024-01-15", "2024-01-31"), ("2024-02-01", "2024-02-29"),
                      ("2024-03-01", "2024-03-10")]


def test_fetch_writes_files_that_pass_the_nasa_checks(tmp_path: Path, monkeypatch) -> None:
    times = [f"2024-01-01T{h:02d}:00" for h in range(24)]
    sun = [0.0] * 7 + [300.0] * 11 + [0.0] * 6
    payload = [block(times, shortwave_radiation=sun) for _ in ISLAND_SITES]
    monkeypatch.setattr(om, "_request", lambda url: payload)
    om.fetch(tmp_path, "20240101", "20240101", pause_s=0)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["model"] == om.MODEL and manifest["grid"]["distinct_cells"] == 1
    with (tmp_path / "Nainativu_hourly.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 24 and list(rows[0]) == list(COLUMNS)
    assert om.validate(tmp_path) == []


def test_validate_rejects_a_pull_from_another_model(tmp_path: Path, monkeypatch) -> None:
    times = [f"2024-01-01T{h:02d}:00" for h in range(24)]
    sun = [0.0] * 7 + [300.0] * 11 + [0.0] * 6
    monkeypatch.setattr(om, "_request",
                        lambda url: [block(times, shortwave_radiation=sun) for _ in ISLAND_SITES])
    om.fetch(tmp_path, "20240101", "20240101", pause_s=0)
    m = json.loads((tmp_path / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(dict(m, model="best_match")))
    assert any("expected" in f for f in om.validate(tmp_path))
