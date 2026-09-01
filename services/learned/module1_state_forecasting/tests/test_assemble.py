"""Tests for module1.assemble -- the offline state producer.

Most tests run against a tiny synthetic data root the test writes itself, so the lane
stays green on a clean clone with no DVC blobs. Two tests exercise the real
``data/`` artifacts and skip when they are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from metacore_contracts.state_schema import FEATURE_NAMES, calibration_quality
from module1 import assemble as asm

REPO_ROOT = Path(__file__).resolve().parents[4]
REAL_LOAD = REPO_ROOT / "data" / "processed" / "island_load_hourly.csv"

QUALITY_OBSERVED = "QUALITY_OBSERVED"
QUALITY_MISSING = "QUALITY_MISSING"
TEMPORAL = ("hour_sin", "hour_cos", "doy_sin", "doy_cos")


def _write_fixture(root: Path) -> None:
    """A flat (.synthetic-style) data root: two islands, a handful of hours each."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "nasa_power").mkdir(exist_ok=True)

    hours = [f"2024-06-15T{h:02d}:00" for h in range(6)] + ["2024-01-02T02:00"]
    load_rows = ["island,timestamp_lst,load_kw,quality"]
    weather = {}
    for island in ("Eluvaitivu", "Delft-Neduntivu"):
        cols = ["island,timestamp_lst,ghi_wh_m2,ghi_clearsky_wh_m2,wind_10m_ms,"
                "wind_50m_ms,temp_2m_c,humidity_2m_pct,precip_mm_hr,pressure_kpa"]
        for i, ts in enumerate(hours):
            load_rows.append(f"{island},{ts},{40 + 5 * i},QUALITY_INTERPOLATED")
            # the last hour is the pressure minimum, for the cyclone selector
            pressure = 101.3 - (2.5 if ts == "2024-01-02T02:00" else 0.1 * i)
            ghi = 0.0 if ts.endswith("T02:00") or i == 0 else 300 + 40 * i
            cols.append(
                f"{island},{ts},{ghi},{max(ghi, 1.0)},{6.0 + 0.1 * i},{7.2 + 0.1 * i},"
                f"{26 + 0.2 * i},{78 + i},{0.5 + 0.1 * i},{pressure}"
            )
        weather[island] = "\n".join(cols) + "\n"

    (root / "island_load_hourly.csv").write_text("\n".join(load_rows) + "\n")
    for island, text in weather.items():
        (root / "nasa_power" / f"{island}_hourly.csv").write_text(text)
    (root / "load_parameters.json").write_text(
        json.dumps({"islands": {
            "Eluvaitivu": {"peak_kw": 120.0},
            "Delft-Neduntivu": {"peak_kw": 900.0},
        }})
    )
    (root / "scenario_library.json").write_text(json.dumps({"scenarios": [
        {"scenario_id": "eluvaitivu-hybrid-decay-2025q4", "island": "Eluvaitivu",
         "start_month": "2025-10", "end_month": "2025-12", "out_of_distribution": True},
    ]}))


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic"
    _write_fixture(root)
    return root


def test_vector_shape_matches_contract(data_root: Path) -> None:
    s = asm.assemble("eluvaitivu", "normal", data_root=data_root)
    assert s["feature_names"] == list(FEATURE_NAMES)
    assert len(s["quality_mask"]) == len(FEATURE_NAMES)
    assert s["node_count"] == 4  # bus + diesel + pv + bess
    for row in s["node_features"]:
        assert len(row) == len(FEATURE_NAMES)
    assert s["embedding_dim"] == 64
    assert s["has_embedding"] is False
    assert s["schema_version"] == "1.0"


def test_mask_is_the_contract_floor_for_normal(data_root: Path) -> None:
    s = asm.assemble("eluvaitivu", "normal", data_root=data_root)
    for name, q in zip(FEATURE_NAMES, s["quality_mask"], strict=True):
        assert q == calibration_quality(name)
    assert s["observed_fraction"] == pytest.approx(12 / 28, abs=1e-4)


def test_electrical_features_are_missing_and_zero(data_root: Path) -> None:
    s = asm.assemble("eluvaitivu", "normal", data_root=data_root)
    for name in ("p_kw_norm", "q_kvar_norm", "voltage_pu", "soc_fraction", "asset_online"):
        idx = FEATURE_NAMES.index(name)
        assert s["quality_mask"][idx] == QUALITY_MISSING
        assert all(row[idx] == 0.0 for row in s["node_features"])


def test_blackout_drops_the_temporal_block(data_root: Path) -> None:
    s = asm.assemble("delft", "blackout", data_root=data_root)
    assert s["degraded"] is True
    for name in TEMPORAL:
        idx = FEATURE_NAMES.index(name)
        assert s["quality_mask"][idx] == QUALITY_MISSING
        assert all(row[idx] == 0.0 for row in s["node_features"])
    assert s["observed_fraction"] == pytest.approx(8 / 28, abs=1e-4)


def test_cyclone_selects_the_pressure_minimum_and_flags_ood(data_root: Path) -> None:
    s = asm.assemble("delft", "cyclone", data_root=data_root)
    assert s["out_of_distribution"] is True
    assert s["timestamp_lst"] == "2024-01-02T02:00"  # the seeded low-pressure hour
    idx = FEATURE_NAMES.index("pressure_kpa_norm")
    assert s["node_features"][0][idx] < -1.0  # well below the record mean


def test_temporal_encoding_is_on_the_unit_circle(data_root: Path) -> None:
    s = asm.assemble("eluvaitivu", "normal", timestamp="2024-06-15T05:00", data_root=data_root)
    row = s["node_features"][0]
    hs, hc = row[FEATURE_NAMES.index("hour_sin")], row[FEATURE_NAMES.index("hour_cos")]
    assert hs**2 + hc**2 == pytest.approx(1.0, abs=1e-3)


def test_only_eluvaitivu_carries_pv_and_storage(data_root: Path) -> None:
    elu = asm.assemble("eluvaitivu", "normal", data_root=data_root)
    delft = asm.assemble("delft", "normal", data_root=data_root)
    assert "pv" in elu["node_names"] and "bess" in elu["node_names"]
    assert "pv" not in delft["node_names"] and "bess" not in delft["node_names"]
    pv_idx = FEATURE_NAMES.index("pv_available_kw_norm")
    pv_node = elu["node_names"].index("pv")
    # pv_available is non-zero on the PV node in daylight, zero on the bus.
    day = asm.assemble("eluvaitivu", "normal", timestamp="2024-06-15T04:00", data_root=data_root)
    assert day["node_features"][pv_node][pv_idx] > 0.0
    assert day["node_features"][0][pv_idx] == 0.0


def test_deterministic_apart_from_timing(data_root: Path) -> None:
    a = asm.assemble("eluvaitivu", "cyclone", data_root=data_root)
    b = asm.assemble("eluvaitivu", "cyclone", data_root=data_root)
    a.pop("generated_ms")
    b.pop("generated_ms")
    assert a == b


def test_unknown_island_is_rejected(data_root: Path) -> None:
    with pytest.raises(ValueError):
        asm.assemble("atlantis", "normal", data_root=data_root)


@pytest.mark.skipif(not REAL_LOAD.exists(), reason="DVC artifacts absent; run `task data`")
def test_real_artifacts_assemble_and_are_in_range() -> None:
    s = asm.assemble("eluvaitivu", "normal")
    assert s["data_source"] == "real"
    assert s["observed_fraction"] == pytest.approx(12 / 28, abs=1e-4)
    site = dict(zip(s["feature_names"], s["node_features"][0], strict=True))
    assert 0.0 <= site["humidity_2m_pct_norm"] <= 1.0
    assert 0.0 <= site["load_kw_norm"] <= 1.5
    assert -6.0 <= site["pressure_kpa_norm"] <= 6.0


@pytest.mark.skipif(not REAL_LOAD.exists(), reason="DVC artifacts absent; run `task data`")
def test_real_cyclone_is_a_genuine_low_pressure_tail() -> None:
    s = asm.assemble("delft", "cyclone")
    idx = s["feature_names"].index("pressure_kpa_norm")
    assert s["node_features"][0][idx] < -2.0
    assert s["out_of_distribution"] is True
