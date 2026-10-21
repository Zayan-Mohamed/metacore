"""Seeded scenarios for the next hours (Step 4 of the method).

Each scenario is one past six-hour window taken whole from the matching situation group -- a
block bootstrap with the block equal to the horizon (Kunsch 1989) -- so islands and hours that
moved together in history move together here. A window contributes, per island, the highest
and lowest demand it reached relative to its start hour; applied to this hour's snapshot that
gives two states to solve: the window's peak and its trough.

Checking only the peak and the trough is exact when a violation is monotone in each island's
demand over the window (voltage drop and feeder current grow with load; a battery exporting
into the feeder over-voltages as load falls). That holds for the radial 400 V feeders here and
is stated as an assumption of the method.

Everything is drawn from ``numpy.random.Generator(PCG64(seed))``: the same seed gives the same
rows, every time, on every machine.

Two corner rows close the set: every island at the top edge of the band, and every island at
the bottom edge. They are deterministic stress points reported beside the certificate, not
random draws, so they are not counted in k of N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bands import Band
from .calibration import Calibration


@dataclass(frozen=True)
class ScenarioSet:
    """``hi[s, i]`` / ``lo[s, i]``: demand multipliers for island i at scenario s's peak/trough."""

    hi: np.ndarray
    lo: np.ndarray
    rows: np.ndarray  # calibration row each scenario was drawn from
    band: Band
    seed: int
    islands: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.rows)

    def corners(self) -> tuple[np.ndarray, np.ndarray]:
        """(top, bottom): every island at the band's upper edge, and at its lower edge."""
        lo_f, hi_f = self.band.factors()
        k = len(self.islands)
        return np.full(k, hi_f), np.full(k, lo_f)


def draw_scenarios(cal: Calibration, band: Band, n: int, seed: int | None = None,
                   static_band: Band | None = None) -> ScenarioSet:
    """Draws ``n`` windows from ``band``'s group and clips them into the band.

    When conformal PID has widened (or narrowed) the band, pass the group's ``static_band``:
    each drawn miss is rescaled by the ratio of the adapted edge to the calibrated edge, so the
    scenarios spread as far as the band now says they should.
    """
    seed = cal.seed if seed is None else int(seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    members = cal.rows_in(static_band.key if static_band else band.key)
    rows = rng.choice(members, size=n, replace=True)
    up = cal.up[rows].copy()
    down = cal.down[rows].copy()
    if static_band is not None:
        up *= _ratio(band.q_up, static_band.q_up)
        down *= _ratio(band.q_down, static_band.q_down)
    lo_f, hi_f = band.factors()
    hi = np.clip(1.0 + up, lo_f, hi_f)
    lo = np.clip(1.0 + down, lo_f, hi_f)
    return ScenarioSet(hi=hi, lo=lo, rows=rows, band=band, seed=seed, islands=cal.islands)


def _ratio(adapted: float, static: float) -> float:
    if not np.isfinite(adapted) or not np.isfinite(static) or static <= 0.0:
        return 1.0
    return adapted / static
