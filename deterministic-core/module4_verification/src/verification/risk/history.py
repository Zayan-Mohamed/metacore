"""Replay-mode access to the recorded history M4 calibrates on and replays (ADR 0004).

The system has replay and simulated modes only, so "now" is always a recorded hour. This module
reads the two recorded inputs M4 uses -- NASA POWER meteorology for the hazard tag and the
island load history for the realized misses that drive conformal PID -- straight from the data
folder. It never imports Module 1's code.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .situation import hazard_index, hazard_level


def repo_root(start: Path | None = None) -> Path | None:
    """The first ancestor holding ``data/processed`` (None outside a checkout)."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data" / "processed").is_dir():
            return parent
    return None


def default_paths() -> dict[str, Path]:
    root = repo_root()
    if root is None:
        raise FileNotFoundError("no data/processed above this module; pass paths explicitly")
    return {
        "hourly_load": root / "data" / "processed" / "island_load_hourly.csv",
        "meteorology": root / "data" / "raw" / "nasa_power",
    }


def read_meteorology(nasa_dir: Path | str, islands: tuple[str, ...],
                     timestamps: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(wind_50m_ms, precip_mm_hr), each shaped (len(timestamps), len(islands))."""
    wind = np.empty((len(timestamps), len(islands)))
    rain = np.empty_like(wind)
    for j, island in enumerate(islands):
        rows: dict[str, tuple[float, float]] = {}
        with open(Path(nasa_dir) / f"{island}_hourly.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["timestamp_lst"]] = (float(r["wind_50m_ms"]), float(r["precip_mm_hr"]))
        for t, ts in enumerate(timestamps):
            wind[t, j], rain[t, j] = rows[ts]
    return wind, rain


def hazard_levels(wind: np.ndarray, rain: np.ndarray,
                  refs: dict[str, dict[str, list[float]]],
                  islands: tuple[str, ...]) -> np.ndarray:
    """Per hour, the worst island's hazard level (0 NORMAL .. 3 EXTREME)."""
    idx = np.stack([hazard_index(wind[:, j], rain[:, j], refs[island])
                    for j, island in enumerate(islands)], axis=1)
    return hazard_level(idx.max(axis=1))
