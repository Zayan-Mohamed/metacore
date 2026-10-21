"""The persistence-miss table (Step 2 of the method).

M4's own guess for the next hours is persistence: "the next six hours look like now". A miss is
how far reality drifted from that guess, in percent of the current hour, so islands of very
different size are comparable. For every start hour t and every island i:

    r[i, h] = load[i, t + h] / load[i, t] - 1,      h = 1 .. horizon

and the row keeps the worst drift each way over the window -- ``up = max_h r`` and
``down = min_h r`` -- because the action must hold in every hour it stays in force. All islands
share one row, so a hot evening that lifts every island at once stays together.

Reads ``data/processed/island_load_hourly.csv`` (island, timestamp_lst, load_kw, quality). Uses
nothing from Module 1's code.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HORIZON_H = 6


@dataclass(frozen=True)
class HourlyLoad:
    """Hourly demand on a common, gap-free hourly axis: ``kw[t, i]``."""

    timestamps: tuple[str, ...]
    islands: tuple[str, ...]
    kw: np.ndarray

    def index_of(self, timestamp: str) -> int:
        return self.timestamps.index(timestamp)


@dataclass(frozen=True)
class MissTable:
    """One row per start hour: worst up and down drift per island over the next ``horizon``."""

    timestamps: tuple[str, ...]  # start hour of each row
    islands: tuple[str, ...]
    up: np.ndarray  # (rows, islands)
    down: np.ndarray  # (rows, islands)
    horizon: int = HORIZON_H

    def __len__(self) -> int:
        return len(self.timestamps)

    def select(self, mask: np.ndarray) -> MissTable:
        idx = np.flatnonzero(mask)
        return MissTable(tuple(self.timestamps[i] for i in idx), self.islands,
                         self.up[idx], self.down[idx], self.horizon)


def read_hourly_load(path: Path | str) -> HourlyLoad:
    """Reads the island load CSV and checks every island has the same gap-free hourly axis."""
    series: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            series.setdefault(row["island"], {})[row["timestamp_lst"]] = float(row["load_kw"])
    islands = tuple(sorted(series))
    stamps = tuple(sorted(series[islands[0]]))
    for island in islands:
        if tuple(sorted(series[island])) != stamps:
            raise ValueError(f"{island}: hourly axis differs from {islands[0]}'s")
    kw = np.array([[series[i][t] for i in islands] for t in stamps], dtype=float)
    if not np.all(np.isfinite(kw)) or np.any(kw <= 0.0):
        raise ValueError("hourly load must be finite and positive")
    return HourlyLoad(stamps, islands, kw)


def persistence_misses(load: HourlyLoad, start: str, end: str,
                       horizon: int = HORIZON_H) -> MissTable:
    """Miss rows for every start hour t with t and its whole window inside [start, end].

    ``end`` is inclusive and bounds the *window*, not just t: a row whose t + horizon falls after
    ``end`` would use data from outside the split, so it is left out.
    """
    stamps = load.timestamps
    lo = stamps.index(start)
    hi = stamps.index(end)
    starts = np.arange(lo, hi - horizon + 1)
    base = load.kw[starts]  # (rows, islands)
    ratios = np.stack([load.kw[starts + h] / base - 1.0 for h in range(1, horizon + 1)])
    return MissTable(
        timestamps=tuple(stamps[t] for t in starts),
        islands=load.islands,
        up=ratios.max(axis=0),
        down=ratios.min(axis=0),
        horizon=horizon,
    )
