"""Split-conformal bands per situation group (Step 3 of the method).

Each miss row gives two scores: how far demand rose above "now" at worst on any island
(``s_up = max_i up_i``) and how far it fell (``s_down = max_i -down_i``). The band edge for a
group is the finite-sample conformal quantile of its scores,

    k = ceil((n + 1)(1 - a)),   q = k-th smallest score,

which (Vovk et al. 2005; Lei et al. 2018) covers the next exchangeable miss with probability at
least 1 - a. Each edge takes a = alpha / 2, so by the union bound the two-sided band covers with
probability at least 1 - alpha. A group with fewer than MIN_GROUP_ROWS rows is not trusted on
its own: the lookup falls back to the next wider group (Situation.keys()).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .situation import Situation

MIN_GROUP_ROWS = 100
ALPHA = 0.05


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score; +inf when the group is too small for it."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if n == 0 or k > n:
        return math.inf
    return float(s[k - 1])


def row_scores(up: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (s_up, s_down): the worst rise and the worst fall across islands, never < 0."""
    return np.maximum(up.max(axis=1), 0.0), np.maximum((-down).max(axis=1), 0.0)


@dataclass(frozen=True)
class Band:
    """The calibrated band for one situation group, in fractions of the current hour's load."""

    key: str
    n: int
    q_up: float
    q_down: float
    alpha: float = ALPHA

    def factors(self) -> tuple[float, float]:
        """(lowest, highest) multiplier of the snapshot demand inside the band."""
        return max(0.0, 1.0 - self.q_down), 1.0 + self.q_up

    def covers(self, s_up: float, s_down: float) -> bool:
        return s_up <= self.q_up and s_down <= self.q_down

    def widened(self, extra: float, key_suffix: str = "") -> Band:
        """The same band with ``extra`` added to both edges (conformal PID, risk/pid.py)."""
        return Band(self.key + key_suffix, self.n, max(0.0, self.q_up + extra),
                    min(1.0, max(0.0, self.q_down + extra)), self.alpha)

    def to_json(self) -> dict:
        """JSON form; an unbounded edge (too few rows) is written as null, not Infinity."""
        return {"key": self.key, "n": self.n, "q_up": _edge(self.q_up),
                "q_down": _edge(self.q_down)}

    @classmethod
    def from_json(cls, g: dict, alpha: float = ALPHA) -> Band:
        def edge(x: float | None) -> float:
            return math.inf if x is None else float(x)

        return cls(g["key"], int(g["n"]), edge(g["q_up"]), edge(g["q_down"]), alpha)


def _edge(q: float) -> float | None:
    return None if math.isinf(q) else round(q, 4)


def build_band(key: str, s_up: np.ndarray, s_down: np.ndarray, alpha: float = ALPHA) -> Band:
    return Band(key, len(s_up), conformal_quantile(s_up, alpha / 2.0),
                conformal_quantile(s_down, alpha / 2.0), alpha)


def lookup(table: dict[str, Band], situation: Situation) -> Band:
    """The narrowest group for this situation that has enough rows."""
    for key in situation.keys():
        band = table.get(key)
        if band is not None and band.n >= MIN_GROUP_ROWS:
            return band
    raise KeyError(f"no calibrated band for {situation} (not even 'all')")
