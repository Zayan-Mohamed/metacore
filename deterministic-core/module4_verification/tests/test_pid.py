"""Conformal PID band adaptation (C10)."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from verification.risk.bands import Band
from verification.risk.calibration import Calibration
from verification.risk.pid import ConformalPID, ReplayHistory, in_storm, replay_pid
from verification.risk.situation import Situation

ALPHA = 0.05


def _miss_rate(scores: np.ndarray, base: Band, pid: ConformalPID | None) -> float:
    misses = []
    for s in scores:
        band = pid.adapted(base, 0) if pid else base
        misses.append(0 if band.covers(s, 0.0) else 1)
        if pid:
            pid.observe(band, s, 0.0)
    return float(np.mean(misses))


def test_long_run_miss_rate_returns_to_alpha_after_a_shift() -> None:
    """Scores double halfway through: the static band breaks, PID tracks back to alpha."""
    rng = np.random.Generator(np.random.PCG64(1))
    calm = rng.exponential(0.1, 4000)
    storm = rng.exponential(0.2, 4000)
    base = Band("synthetic", 4000, float(np.quantile(calm, 1 - ALPHA)), 1.0, ALPHA)
    scores = np.concatenate([calm, storm])
    static = _miss_rate(scores[4000:], base, None)
    pid = ConformalPID(ALPHA, scorecaster=(0.0, 0.0, 0.0, 0.0))
    _miss_rate(scores[:4000], base, pid)
    adaptive = _miss_rate(scores[4000:], base, pid)
    assert static > 3 * ALPHA
    assert abs(adaptive - ALPHA) < 0.015


def test_the_scorecaster_widens_before_the_first_miss() -> None:
    pid = ConformalPID(ALPHA)
    base = Band("h19", 450, 0.3, 0.7, ALPHA)
    calm, severe = pid.adapted(base, 0), pid.adapted(base, 2)
    assert severe.q_up - calm.q_up == pytest.approx(0.08)
    assert pid.state.updates == 0


@cache
def replay() -> tuple[Calibration, ReplayHistory]:
    cal = Calibration.load()
    return cal, ReplayHistory.load(cal)


def test_replay_is_deterministic_and_causal() -> None:
    cal, hist = replay()
    a_band, a_state = replay_pid(cal, hist, "2025-11-28T19:00")
    b_band, b_state = replay_pid(cal, hist, "2025-11-28T19:00")
    assert a_band == b_band and a_state.to_json() == b_state.to_json()
    # Poison every hour after "now": the answer must not change
    later = hist.kw.copy()
    later[hist.index("2025-11-28T19:00") + 1 :] *= 10.0
    poisoned = ReplayHistory(hist.timestamps, later, hist.hazard)
    c_band, c_state = replay_pid(cal, poisoned, "2025-11-28T19:00")
    assert c_band == a_band and c_state.to_json() == a_state.to_json()


def test_ditwah_is_storm_mode() -> None:
    _, hist = replay()
    assert hist.hazard_at("2025-11-28T19:00") >= 2
    assert in_storm(hist, hist.index("2025-11-28T19:00"))
    assert not in_storm(hist, hist.index("2025-08-12T15:00"))


def test_pid_coverage_is_closer_to_target_than_static_on_the_test_months() -> None:
    """E2 in miniature: Oct-Dec 2025 (Ditwah and the hybrid decay), static vs PID."""
    cal, hist = replay()
    h = cal.horizon_h
    lo, hi = hist.index("2025-10-01T00:00"), hist.index("2025-12-31T17:00")
    pid = ConformalPID(cal.alpha)
    used_static: dict[int, Band] = {}
    used_pid: dict[int, Band] = {}
    miss_static, miss_pid = [], []
    for t in range(lo - 14 * 24, hi + 1):
        done = t - h
        if done in used_pid:
            s = hist.realized_scores(done, h)
            b_static, b_pid = used_static.pop(done), used_pid.pop(done)
            pid.observe(b_pid, *s)
            if done >= lo:
                miss_static.append(0 if b_static.covers(*s) else 1)
                miss_pid.append(0 if b_pid.covers(*s) else 1)
        base = cal.band_for(Situation.at(hist.timestamps[t], int(hist.hazard[t])))
        used_static[t] = base
        used_pid[t] = pid.adapted(base, int(hist.hazard[t]))
    static, adaptive = np.mean(miss_static), np.mean(miss_pid)
    assert abs(adaptive - cal.alpha) < abs(static - cal.alpha)
