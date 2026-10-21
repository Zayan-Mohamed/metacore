"""The offline calibration: persistence-miss table (C5) and the frozen artifact (C7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from verification.calibrate.misses import persistence_misses, read_hourly_load

REPO = Path(__file__).resolve().parents[3]
HOURLY = REPO / "data" / "processed" / "island_load_hourly.csv"

# 10 hours, two islands. Delft doubles over the day; Analaitivu dips then recovers.
HAND = {
    "Delft-Neduntivu": [100, 110, 120, 150, 200, 180, 160, 140, 120, 100],
    "Analaitivu": [50, 40, 45, 50, 55, 60, 50, 40, 30, 50],
}


def write_hand_csv(path: Path) -> Path:
    lines = ["island,timestamp_lst,load_kw,quality"]
    for island, values in HAND.items():
        for h, v in enumerate(values):
            lines.append(f"{island},2025-01-01T{h:02d}:00,{v},QUALITY_INTERPOLATED")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_hand_made_csv_gives_the_expected_misses(tmp_path: Path) -> None:
    load = read_hourly_load(write_hand_csv(tmp_path / "hand.csv"))
    assert load.islands == ("Analaitivu", "Delft-Neduntivu")
    table = persistence_misses(load, "2025-01-01T00:00", "2025-01-01T09:00", horizon=3)
    # 10 hours, a 3-hour window: start hours 00..06
    assert len(table) == 7
    assert table.timestamps[0] == "2025-01-01T00:00" and table.timestamps[-1] == "2025-01-01T06:00"
    delft, ana = 1, 0
    # t=00: Delft 100 -> 110, 120, 150: worst up +50 %, never below
    assert table.up[0, delft] == pytest.approx(0.50)
    assert table.down[0, delft] == pytest.approx(0.10)
    # t=00: Analaitivu 50 -> 40, 45, 50: worst down -20 %, worst up 0 %
    assert table.down[0, ana] == pytest.approx(-0.20)
    assert table.up[0, ana] == pytest.approx(0.0)
    # t=04: Delft 200 -> 180, 160, 140: falls all window
    assert table.up[4, delft] == pytest.approx(-0.10)
    assert table.down[4, delft] == pytest.approx(-0.30)


def test_no_row_uses_data_after_its_own_window(tmp_path: Path) -> None:
    load = read_hourly_load(write_hand_csv(tmp_path / "hand.csv"))
    table = persistence_misses(load, "2025-01-01T00:00", "2025-01-01T05:00", horizon=3)
    # The split ends at 05:00, so the last start hour whose window fits is 02:00
    assert table.timestamps[-1] == "2025-01-01T02:00"
    # Corrupt everything after the split: the table must not change
    later = load.kw.copy()
    later[6:] = 1e6
    from verification.calibrate.misses import HourlyLoad
    poisoned = HourlyLoad(load.timestamps, load.islands, later)
    again = persistence_misses(poisoned, "2025-01-01T00:00", "2025-01-01T05:00", horizon=3)
    assert np.array_equal(again.up, table.up) and np.array_equal(again.down, table.down)


def test_a_gap_in_one_island_is_refused(tmp_path: Path) -> None:
    path = write_hand_csv(tmp_path / "hand.csv")
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError):
        read_hourly_load(path)


def test_real_calibration_split_row_count() -> None:
    load = read_hourly_load(HOURLY)
    assert load.islands == ("Analaitivu", "Delft-Neduntivu", "Eluvaitivu", "Nainativu")
    table = persistence_misses(load, "2024-01-01T00:00", "2025-06-30T23:00")
    hours = (366 + 181) * 24  # 2024 is a leap year; Jan-Jun 2025 has 181 days
    assert len(table) == hours - 6
    assert table.up.shape == table.down.shape == (hours - 6, 4)
    assert np.all(table.up >= table.down)


# ------------------------------------------------------------------------------- C7: the artifact

def test_rebuilding_from_the_same_history_is_byte_identical() -> None:
    from verification.calibrate.build import build_artifact, render
    from verification.risk.calibration import DEFAULT_ARTIFACT
    from verification.risk.history import default_paths

    paths = default_paths()
    first = render(build_artifact(paths["hourly_load"], paths["meteorology"]))
    second = render(build_artifact(paths["hourly_load"], paths["meteorology"]))
    assert first == second
    # ...and identical to the committed artifact (what `calibrate check` asserts in CI)
    assert first == DEFAULT_ARTIFACT.read_text()


def test_the_artifact_records_provenance_and_loads() -> None:
    from verification.risk.calibration import Calibration

    cal = Calibration.load()
    assert cal.raw["schema"] == "m4-calibration/1"
    assert cal.raw["splits"]["calibration"] == ["2024-01-01T00:00", "2025-06-30T23:00"]
    assert cal.islands == ("Analaitivu", "Delft-Neduntivu", "Eluvaitivu", "Nainativu")
    assert len(cal.row_timestamps) == cal.up.shape[0] == cal.raw["rows"]["count"]
    assert cal.row_timestamps[-1] == "2025-06-30T17:00"  # last start whose window fits
    assert cal.seed == 20251128 and cal.horizon_h == 6


def test_stored_bands_match_their_rows() -> None:
    from verification.risk.bands import build_band, row_scores
    from verification.risk.calibration import Calibration

    cal = Calibration.load()
    s_up, s_down = row_scores(cal.up, cal.down)
    for key in ("h19|NE_MONSOON|CALM", "h03", "all"):
        idx = cal.rows_in(key)
        again = build_band(key, s_up[idx], s_down[idx], cal.alpha)
        assert again.n == cal.bands[key].n
        assert again.q_up == pytest.approx(cal.bands[key].q_up, abs=1e-4)
        assert again.q_down == pytest.approx(cal.bands[key].q_down, abs=1e-4)


def test_a_changed_input_makes_the_artifact_stale(tmp_path: Path) -> None:
    from verification.risk.calibration import Calibration, StaleCalibrationError
    from verification.risk.history import default_paths

    paths = default_paths()
    cal = Calibration.load()
    cal.check_inputs(paths["hourly_load"], paths["meteorology"])  # the real inputs match

    lines = HOURLY.read_text().splitlines()
    island, ts, kw, quality = lines[1].split(",")
    lines[1] = ",".join([island, ts, str(float(kw) + 0.001), quality])
    changed = tmp_path / "island_load_hourly.csv"
    changed.write_text("\n".join(lines) + "\n")
    with pytest.raises(StaleCalibrationError):
        cal.check_inputs(changed, paths["meteorology"])


def test_a_hand_edited_artifact_is_refused(tmp_path: Path) -> None:
    import json

    from verification.risk.calibration import DEFAULT_ARTIFACT, Calibration, StaleCalibrationError

    raw = json.loads(DEFAULT_ARTIFACT.read_text())
    raw["groups"]["h19|NE_MONSOON|CALM"]["q_up"] = 0.01  # a narrower band would approve more
    edited = tmp_path / "calibration_v1.json"
    edited.write_text(json.dumps(raw))
    with pytest.raises(StaleCalibrationError):
        Calibration.load(edited)


def test_the_check_cli_passes_on_the_committed_artifact() -> None:
    from verification.calibrate.__main__ import main

    assert main(["check"]) == 0


def test_the_wheel_ships_the_frozen_calibration() -> None:
    """The root .gitignore hides **/artifacts/*; the wheel must force-include the artifact."""
    import tomllib

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml")
                              .read_text())
    included = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["src/verification/artifacts/calibration_v1.json"] == \
        "verification/artifacts/calibration_v1.json"
