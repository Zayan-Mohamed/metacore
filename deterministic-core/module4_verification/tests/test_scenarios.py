"""Seeded block-bootstrap scenarios (C9)."""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from verification.risk.calibration import Calibration
from verification.risk.scenarios import draw_scenarios
from verification.risk.situation import Situation

N = 688


@cache
def cal() -> Calibration:
    return Calibration.load()


def ditwah_evening_band():
    return cal().band_for(Situation.at("2025-11-28T19:00", hazard_level=3))


def test_same_seed_same_688_rows() -> None:
    band = ditwah_evening_band()
    a = draw_scenarios(cal(), band, N, seed=11)
    b = draw_scenarios(cal(), band, N, seed=11)
    assert len(a) == N
    assert np.array_equal(a.rows, b.rows)
    assert np.array_equal(a.hi, b.hi) and np.array_equal(a.lo, b.lo)
    c = draw_scenarios(cal(), band, N, seed=12)
    assert not np.array_equal(a.rows, c.rows)


def test_the_default_seed_is_the_artifacts() -> None:
    s = draw_scenarios(cal(), ditwah_evening_band(), 10)
    assert s.seed == cal().seed


def test_every_row_comes_from_the_matching_group_and_stays_in_the_band() -> None:
    band = ditwah_evening_band()
    s = draw_scenarios(cal(), band, N)
    assert set(s.rows) <= set(cal().rows_in(band.key))
    lo_f, hi_f = band.factors()
    assert np.all((s.hi >= lo_f) & (s.hi <= hi_f))
    assert np.all((s.lo >= lo_f) & (s.lo <= hi_f))
    assert np.all(s.lo <= s.hi)


def test_corner_rows_are_the_band_edges() -> None:
    band = ditwah_evening_band()
    top, bottom = draw_scenarios(cal(), band, N).corners()
    lo_f, hi_f = band.factors()
    assert np.all(top == hi_f) and np.all(bottom == lo_f)
    assert top.shape == (4,)


def test_island_correlation_is_kept() -> None:
    """Whole rows are drawn, so islands keep the co-movement they had in history."""
    band = cal().band_for(Situation.at("2025-03-10T15:00"))
    s = draw_scenarios(cal(), band, 5000, seed=3)
    members = cal().rows_in(band.key)
    data = np.corrcoef(cal().up[members].T)
    sampled = np.corrcoef((s.hi - 1.0).T)
    assert sampled == pytest.approx(data, abs=0.1)


def test_a_widened_band_spreads_the_scenarios() -> None:
    static = ditwah_evening_band()
    wide = static.widened(0.10)
    s0 = draw_scenarios(cal(), static, N)
    s1 = draw_scenarios(cal(), wide, N, static_band=static)
    assert np.array_equal(s0.rows, s1.rows)  # common random numbers
    assert s1.lo.min() < s0.lo.min()
    assert np.all(s1.lo <= s0.lo + 1e-12)
