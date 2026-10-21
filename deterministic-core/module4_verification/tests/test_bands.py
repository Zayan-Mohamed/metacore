"""Situation groups and split-conformal bands (C6)."""

from __future__ import annotations

import math
from collections import defaultdict
from functools import cache

import numpy as np
import pytest
from verification.calibrate.groups import group_rows
from verification.calibrate.misses import persistence_misses, read_hourly_load
from verification.risk.bands import (
    ALPHA,
    MIN_GROUP_ROWS,
    Band,
    build_band,
    conformal_quantile,
    lookup,
    row_scores,
)
from verification.risk.history import default_paths, hazard_levels, read_meteorology
from verification.risk.situation import Situation, hazard_scale_reference, season_of

# ------------------------------------------------------------------------------- synthetic


def test_textbook_quantile_on_known_misses() -> None:
    """Scores 1..100 at alpha = 0.1: k = ceil(101 * 0.9) = 91, so q is the 91st smallest."""
    scores = np.arange(100, 0, -1, dtype=float)  # order must not matter
    assert conformal_quantile(scores, 0.1) == 91.0
    # n = 19, alpha = 0.05: k = ceil(20 * 0.95) = 19 -> the maximum
    assert conformal_quantile(np.arange(1.0, 20.0), 0.05) == 19.0


def test_too_few_rows_give_an_unbounded_band() -> None:
    # n = 18 at alpha = 0.05 needs k = 19 > n: no finite quantile has the guarantee
    assert conformal_quantile(np.arange(18.0), 0.05) == math.inf


def test_coverage_on_exchangeable_synthetic_misses() -> None:
    """Fresh draws from the same distribution land inside the band at least 1 - alpha."""
    rng = np.random.Generator(np.random.PCG64(7))
    cal = rng.lognormal(size=(2000, 4)) - 1.0
    band = build_band("all", *row_scores(cal, -cal))
    fresh = rng.lognormal(size=(20000, 4)) - 1.0
    s_up, s_down = row_scores(fresh, -fresh)
    covered = np.mean((s_up <= band.q_up) & (s_down <= band.q_down))
    assert covered >= 1.0 - ALPHA - 0.005


def test_situation_keys_go_from_narrow_to_wide() -> None:
    s = Situation.at("2025-11-28T19:00", hazard_level=3)
    assert s.season == season_of("2025-11-28T19:00") == "NE_MONSOON"
    assert s.keys() == ["h19|NE_MONSOON|STORM", "h19|NE_MONSOON", "h19", "all"]


def test_lookup_falls_back_when_a_group_is_thin() -> None:
    table = {
        "h19|NE_MONSOON|STORM": Band("h19|NE_MONSOON|STORM", MIN_GROUP_ROWS - 1, 0.9, 0.9),
        "h19|NE_MONSOON": Band("h19|NE_MONSOON", 450, 0.3, 0.7),
        "all": Band("all", 13000, 1.2, 0.8),
    }
    band = lookup(table, Situation.at("2025-11-28T19:00", hazard_level=3))
    assert band.key == "h19|NE_MONSOON"
    calm = lookup(table, Situation.at("2025-11-28T19:00", hazard_level=0))
    assert calm.key == "h19|NE_MONSOON"  # no CALM group in this table either


def test_a_row_joins_its_neighbouring_hours() -> None:
    groups = group_rows(("2024-01-01T00:00", "2024-07-01T12:00"), np.array([0, 3]))
    assert set(groups["h23|NE_MONSOON|CALM"]) == {0}
    assert set(groups["h01"]) == {0}
    assert set(groups["h13|SW_MONSOON|STORM"]) == {1}
    assert set(groups["all"]) == {0, 1}


# ------------------------------------------------------------------------------- real data


@cache
def calibrated():
    paths = default_paths()
    load = read_hourly_load(paths["hourly_load"])
    wind, rain = read_meteorology(paths["meteorology"], load.islands, load.timestamps)
    end = load.index_of("2025-06-30T23:00")
    refs = {isl: hazard_scale_reference(wind[: end + 1, j], rain[: end + 1, j])
            for j, isl in enumerate(load.islands)}
    level = dict(zip(load.timestamps, hazard_levels(wind, rain, refs, load.islands).tolist(),
                     strict=True))
    cal = persistence_misses(load, "2024-01-01T00:00", "2025-06-30T23:00")
    groups = group_rows(cal.timestamps, np.array([level[t] for t in cal.timestamps]))
    s_up, s_down = row_scores(cal.up, cal.down)
    table = {k: build_band(k, s_up[i], s_down[i]) for k, i in groups.items()}
    return load, level, table


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def test_held_out_coverage_jul_to_sep_2025() -> None:
    """Coverage on the validation months, overall and per group.

    Overall coverage must reach 1 - alpha. Per group, the count covered must not be
    significantly below 1 - alpha (exact binomial, 0.1 %) in at least 80 % of groups: the hourly
    profile drifts a little between 2024 and 2025, and that residual drift is what conformal PID
    (C10) corrects online.
    """
    load, level, table = calibrated()
    val = persistence_misses(load, "2025-07-01T00:00", "2025-09-30T23:00")
    s_up, s_down = row_scores(val.up, val.down)
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ts, a, b in zip(val.timestamps, s_up, s_down, strict=True):
        band = lookup(table, Situation.at(ts, level[ts]))
        hits[band.key][0] += int(band.covers(a, b))
        hits[band.key][1] += 1
    covered = sum(c for c, _ in hits.values())
    total = sum(n for _, n in hits.values())
    assert covered / total >= 1.0 - ALPHA
    consistent = [_binom_cdf(c, n, 1.0 - ALPHA) > 1e-3 for c, n in hits.values()]
    assert np.mean(consistent) >= 0.8


@pytest.mark.parametrize(("hour", "rises"), [(15, True), (3, True), (19, False)])
def test_bands_follow_the_daily_ramp(hour: int, rises: bool) -> None:
    """15:00 and 03:00 guesses miss a rise (evening and morning ramps); 19:00 misses a fall."""
    _, _, table = calibrated()
    band = lookup(table, Situation.at(f"2025-11-28T{hour:02d}:00"))
    assert (band.q_up > band.q_down) is rises
