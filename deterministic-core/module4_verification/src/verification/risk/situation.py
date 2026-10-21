"""Situation keys: time of day, season and hazard level.

A persistence miss depends on the situation it starts in: the 15:00 guess misses the evening
ramp, the 03:00 guess misses the morning one. Bands are therefore looked up per situation, with
a fallback to a wider group when the narrow one has too few rows (risk/bands.py).

Hazard is M4's own deterministic tag from NASA POWER meteorology (the same drivers M1's hazard
exposure uses): 50 m wind and rainfall, each scaled between its calibration-split median and
99.9th percentile, combined as a probabilistic OR and cut into the four HazardSeverity levels of
module1.proto. The scaling bounds are frozen in the calibration artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Months -> season. North-east monsoon as used in the thread's worked example (Oct to Jan); the
# first inter-monsoon and the south-west monsoon split the rest.
SEASONS: dict[int, str] = {
    10: "NE_MONSOON", 11: "NE_MONSOON", 12: "NE_MONSOON", 1: "NE_MONSOON",
    2: "INTER_MONSOON", 3: "INTER_MONSOON", 4: "INTER_MONSOON",
    5: "SW_MONSOON", 6: "SW_MONSOON", 7: "SW_MONSOON", 8: "SW_MONSOON", 9: "SW_MONSOON",
}

# module1.proto HazardSeverity, minus UNSPECIFIED.
HAZARD_LEVELS = ("NORMAL", "ELEVATED", "SEVERE", "EXTREME")
# Lower edges of ELEVATED, SEVERE, EXTREME on the [0, 1] hazard index.
HAZARD_CUTS = (0.4, 0.7, 0.9)
# The hazard part of a group key is two-valued: there are too few storm hours to split further.
STORM_FROM = HAZARD_LEVELS.index("SEVERE")

# +/- this many hours around the start hour count as "the same time of day".
HOUR_WINDOW = 1


def season_of(timestamp: str) -> str:
    """Season of an ISO local timestamp like ``2025-11-28T19:00``."""
    return SEASONS[int(timestamp[5:7])]


def hour_of(timestamp: str) -> int:
    return int(timestamp[11:13])


def hours_near(hour: int) -> frozenset[int]:
    """The start hours that count as the same time of day as ``hour`` (wraps midnight)."""
    return frozenset((hour + d) % 24 for d in range(-HOUR_WINDOW, HOUR_WINDOW + 1))


def hazard_bucket(level: int) -> str:
    return "STORM" if level >= STORM_FROM else "CALM"


@dataclass(frozen=True)
class Situation:
    """Where a verify call sits: its start hour, season and hazard level (0..3).

    ``hazard_level=None`` means the hazard tag is missing for this hour: the lookup then starts
    at the group without hazard rather than guessing a level.
    """

    hour: int
    season: str
    hazard_level: int | None = 0

    @classmethod
    def at(cls, timestamp: str, hazard_level: int | None = 0) -> Situation:
        return cls(hour=hour_of(timestamp), season=season_of(timestamp),
                   hazard_level=hazard_level)

    @property
    def hazard_name(self) -> str:
        return "UNKNOWN" if self.hazard_level is None else HAZARD_LEVELS[self.hazard_level]

    def keys(self) -> list[str]:
        """Group keys from narrowest to widest; bands.py takes the first with enough rows."""
        h = f"h{self.hour:02d}"
        wider = [f"{h}|{self.season}", h, "all"]
        if self.hazard_level is None:
            return wider
        return [f"{h}|{self.season}|{hazard_bucket(self.hazard_level)}", *wider]


# ------------------------------------------------------------------------------------ hazard

def hazard_scale_reference(wind: np.ndarray, rain: np.ndarray) -> dict[str, list[float]]:
    """(lo, hi) per driver: median and 99.9th percentile over the rows given (calibration)."""
    return {
        "wind_50m_ms": [round(float(np.median(wind)), 4),
                        round(float(np.percentile(wind, 99.9)), 4)],
        "precip_mm_hr": [round(float(np.median(rain)), 4),
                         round(float(np.percentile(rain, 99.9)), 4)],
    }


def _scale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def hazard_index(wind: np.ndarray, rain: np.ndarray, ref: dict[str, list[float]]) -> np.ndarray:
    """Hazard in [0, 1]: either driver alone can make the hour hazardous."""
    hw = _scale(np.asarray(wind, dtype=float), *ref["wind_50m_ms"])
    hr = _scale(np.asarray(rain, dtype=float), *ref["precip_mm_hr"])
    return 1.0 - (1.0 - hw) * (1.0 - hr)


def hazard_level(index: float | np.ndarray) -> np.ndarray:
    """Hazard index -> level 0 (NORMAL) .. 3 (EXTREME)."""
    return np.searchsorted(np.asarray(HAZARD_CUTS), np.asarray(index, dtype=float), side="right")
