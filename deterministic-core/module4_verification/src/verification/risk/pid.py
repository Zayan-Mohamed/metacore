"""Conformal PID control of the band during storms (Step 6, the online half).

Angelopoulos, Candes & Tibshirani (2023), "Conformal PID Control for Time Series Prediction",
NeurIPS. Every hour M4 can check one thing for free: did the six-hour window that started six
hours ago land inside the band M4 used for it? That miss signal err_t in {0, 1} steers an offset
added to the calibrated situation band:

    P  (quantile tracking)   offset += eta * (err_t - alpha)
    I  (integrator)          k_i * clip(sum of (err - alpha) over the last W windows, +/- cap)
    D  (scorecaster)         a fixed, hand-written margin per hazard level, applied before
                             the first big miss arrives -- "the weather report says frost"

The P part alone is Adaptive Conformal Inference on the band width (Gibbs & Candes 2021) and
gives the long-run guarantee: over any long run of hours, the share of windows outside the band
returns to alpha even when the world shifts. The scorecaster is a fixed rule, not a trained
model, so ADR 0003 holds. The state is a few numbers, replayed from recorded history up to "now"
and logged with every certificate, so a verdict replays exactly.

Storm mode: by default PID is applied only when the hour is at hazard level SEVERE or worse, or
was within the last STORM_MEMORY_H hours; on calm days the static calibrated band is used.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from .bands import Band, row_scores
from .calibration import Calibration
from .situation import STORM_FROM, Situation

ETA = 0.02  # P gain: one uncovered window widens the band by ~2 points
K_I = 0.005  # I gain per unit of accumulated excess misses
I_WINDOW = 72  # hours of misses the integrator remembers
I_CAP = 4.0  # saturation of the accumulated excess (in windows)
SCORECASTER = (0.0, 0.02, 0.08, 0.12)  # extra band per hazard level NORMAL..EXTREME
WARMUP_H = 14 * 24  # how far back the replay starts adapting
STORM_MEMORY_H = 24


@dataclass
class PIDState:
    """Everything the adapted band depends on, logged in the certificate."""

    p_offset: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0
    updates: int = 0
    recent_misses: int = 0
    recent_windows: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=I_WINDOW), repr=False)

    @property
    def extra(self) -> float:
        return self.p_offset + self.i_term + self.d_term

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("history")
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()} | {
            "extra": round(self.extra, 6)}


class ConformalPID:
    """The controller. ``observe`` one realized window at a time, ``band`` for the next hour."""

    def __init__(self, alpha: float, eta: float = ETA, k_i: float = K_I,
                 scorecaster: tuple[float, ...] = SCORECASTER) -> None:
        self.alpha = alpha
        self.eta = eta
        self.k_i = k_i
        self.scorecaster = scorecaster
        self.state = PIDState()

    def adapted(self, base: Band, hazard_level: int) -> Band:
        """The base band widened by P + I + the scorecaster for this hour's hazard level."""
        self.state.d_term = self.scorecaster[int(hazard_level)]
        return base.widened(self.state.extra, "+pid")

    def observe(self, used: Band, s_up: float, s_down: float) -> int:
        """Feeds back one realized window scored against the band that was used for it."""
        err = 0 if used.covers(s_up, s_down) else 1
        st = self.state
        st.p_offset += self.eta * (err - self.alpha)
        st.history.append(err - self.alpha)
        st.i_term = self.k_i * float(np.clip(sum(st.history), -I_CAP, I_CAP))
        st.updates += 1
        st.recent_windows = len(st.history)
        st.recent_misses = int(sum(1 for e in st.history if e > 0))
        return err


@dataclass(frozen=True)
class ReplayHistory:
    """Recorded hourly load and hazard levels on one axis (replay mode, ADR 0004)."""

    timestamps: tuple[str, ...]
    kw: np.ndarray  # (hours, islands), island order = Calibration.islands
    hazard: np.ndarray  # (hours,)

    @classmethod
    def load(cls, cal: Calibration, hourly_load=None, meteorology=None) -> ReplayHistory:
        from ..calibrate.misses import read_hourly_load
        from .history import default_paths, hazard_levels, read_meteorology

        paths = default_paths() if hourly_load is None or meteorology is None else {}
        load = read_hourly_load(hourly_load or paths["hourly_load"])
        if load.islands != cal.islands:
            raise ValueError(f"history islands {load.islands} != calibration {cal.islands}")
        wind, rain = read_meteorology(meteorology or paths["meteorology"], load.islands,
                                      load.timestamps)
        levels = hazard_levels(wind, rain, cal.raw["hazard"]["refs"], load.islands)
        return cls(load.timestamps, load.kw, np.asarray(levels, dtype=int))

    def index(self, timestamp: str) -> int:
        return self.timestamps.index(timestamp)

    def hazard_at(self, timestamp: str) -> int:
        return int(self.hazard[self.index(timestamp)])

    def realized_scores(self, start: int, horizon: int) -> tuple[float, float]:
        """(s_up, s_down) of the window starting at hour index ``start``."""
        base = self.kw[start]
        ratios = self.kw[start + 1 : start + 1 + horizon] / base - 1.0
        s_up, s_down = row_scores(ratios.max(axis=0)[None, :], ratios.min(axis=0)[None, :])
        return float(s_up[0]), float(s_down[0])


def in_storm(history: ReplayHistory, now: int) -> bool:
    lo = max(0, now - STORM_MEMORY_H)
    return bool(history.hazard[lo : now + 1].max() >= STORM_FROM)


def replay_pid(cal: Calibration, history: ReplayHistory, timestamp: str,
               warmup_h: int = WARMUP_H) -> tuple[Band, PIDState]:
    """Runs the controller over the recorded hours before ``timestamp`` and returns the band
    for ``timestamp`` with the controller state that produced it.

    Causal by construction: the window starting at hour t' is only observed once t' + horizon
    <= now, so no hour after ``timestamp`` is ever read.
    """
    now = history.index(timestamp)
    h = cal.horizon_h
    pid = ConformalPID(cal.alpha)
    used: dict[int, Band] = {}
    for t in range(max(0, now - warmup_h), now + 1):
        # Windows that completed by hour t: the one that started at t - h.
        done = t - h
        if done in used:
            pid.observe(used.pop(done), *history.realized_scores(done, h))
        ts = history.timestamps[t]
        base = cal.band_for(Situation.at(ts, int(history.hazard[t])))
        used[t] = pid.adapted(base, int(history.hazard[t]))
    return used[now], pid.state
