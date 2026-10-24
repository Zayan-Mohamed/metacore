"""Live-weather mode (ADR 0006): today's inputs must be built exactly as training's were.

A live hour that reaches the model scaled differently from the training hours would produce a
confident number from the wrong distribution, with no error anywhere. These tests pin the live
path to the training path wherever both can be computed, using committed 2025 hours as "live".
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from module1.train import dataset as ds  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
PROCESSED = REPO / "data" / "processed"
needs_artifacts = pytest.mark.skipif(
    not (PROCESSED / "island_load_hourly.csv").exists()
    or not (REPO / "data" / "raw" / "open_meteo" / "manifest.json").exists(),
    reason="calibration artifacts absent -- run `task data`",
)
NOW = "2025-11-28T12:00"  # Ditwah, inside the ledger


def _om_rows(lo: str, hi: str) -> dict[str, list[dict]]:
    out = {}
    for isl in ds.ISLAND_ORDER:
        with (REPO / "data" / "raw" / "open_meteo" / f"{isl}_hourly.csv").open() as fh:
            out[isl] = [{k: (v if k in ("island", "timestamp_lst") else float(v))
                         for k, v in r.items()}
                        for r in csv.DictReader(fh) if lo <= r["timestamp_lst"] <= hi]
    return out


@pytest.fixture(scope="module")
def live():
    from module1.live import state
    return state


@needs_artifacts
def test_live_load_reproduces_the_load_stage_given_the_same_temperatures(live) -> None:
    fz = live.frozen()
    isl = "Nainativu"
    temps = fz.nasa_temps[isl]
    stamps = [s for s in temps if s.startswith("2025-07")]
    got = live.live_load(fz, isl, stamps, [temps[s] for s in stamps])
    with (PROCESSED / "island_load_hourly.csv").open() as fh:
        want = {r["timestamp_lst"]: float(r["load_kw"]) for r in csv.DictReader(fh)
                if r["island"] == isl and r["timestamp_lst"].startswith("2025-07")}
    assert np.allclose(got, [want[s] for s in stamps], atol=1e-3)


@needs_artifacts
def test_live_weather_inputs_equal_the_training_inputs(live) -> None:
    rows = _om_rows("2025-11-26T00:00", "2025-11-29T23:00")
    b = live.build_inputs(rows, NOW)
    data = ds.build(weather_source="open_meteo")
    t = int(np.where(data.timestamps == NOW)[0][0])
    window = data.x[t - 23: t + 1]
    load_cols = {ds.FEATURES.index(f) for f in ("load_kw_norm", "load_ramp_kw_per_h_norm",
                                                  "vulnerability")}
    weather_cols = [i for i in range(len(ds.FEATURES)) if i not in load_cols]
    assert np.array_equal(b["window"][0][..., weather_cols], window[..., weather_cols])


@needs_artifacts
def test_fragility_is_carried_forward_and_flagged_after_the_ledger(live) -> None:
    fz = live.frozen()
    # January 2026 legitimately uses December 2025's ledger (one-month lag); later months carry
    # that last known value forward and say so.
    last_known, carried_in = live.live_decay(fz, ["2026-01-15T12:00"])
    after, carried_after = live.live_decay(fz, ["2026-10-24T12:00"])
    assert carried_in == [False] and carried_after == [True]
    assert after[0] == pytest.approx(last_known[0])
    assert after[0] > 0.9                               # the plant at ~3% of normal
    assert live.hybrid_down(fz, "2026-10-24T12:00") == (True, True)


@needs_artifacts
def test_one_live_step_feeds_module2_and_module4(live) -> None:
    rows = _om_rows("2025-11-26T00:00", "2025-11-29T23:00")
    out = live.step(rows, NOW)
    assert out["issued_for"] == NOW and len(out["horizon_hours"]) == 6
    c = out["contract_state"]
    assert len(c["node_features"]) == 10 and all(len(r) == 28 for r in c["node_features"])
    from module1.assemble import assemble
    assert c["quality_mask"] == assemble("nainativu", "normal", NOW)["quality_mask"]
    assert [s["timestamp_lst"] for s in out["snapshots"]] == [NOW, *out["horizon_hours"]]
    assert all(s["topology_id"] == "northern-islands-v1" for s in out["snapshots"])
    for n in out["nodes"]:
        assert len(n["forecast"]["stgnn"]) == 6 and len(n["embedding"]) == 64


@needs_artifacts
def test_the_same_weather_gives_the_same_forecast(live) -> None:
    rows = _om_rows("2025-11-26T00:00", "2025-11-29T23:00")
    a, b = live.forecast(rows, NOW), live.forecast(rows, NOW)
    assert a["input_sha256"] == b["input_sha256"]
    assert [n["forecast"] for n in a["nodes"]] == [n["forecast"] for n in b["nodes"]]


@needs_artifacts
def test_a_gap_in_the_weather_is_refused_not_filled(live) -> None:
    rows = _om_rows("2025-11-26T00:00", "2025-11-29T23:00")
    rows["Delft-Neduntivu"][50]["wind_50m_ms"] = None
    with pytest.raises(ValueError, match="missing weather"):
        live.build_inputs(rows, NOW)
