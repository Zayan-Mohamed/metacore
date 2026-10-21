"""Assign miss rows to situation groups (Step 6, the offline half).

A row starting at hour ``hh`` belongs to the groups of every hour within +/- HOUR_WINDOW of it,
so each hour's group pools three start hours (Rule 1 in the thread's worked example). Rows also
carry their season (Rule 2) and their hazard bucket (Rule 3); every wider key in
Situation.keys() is filled too, so the lookup can fall back.
"""

from __future__ import annotations

import numpy as np

from ..risk.situation import HOUR_WINDOW, hazard_bucket, hour_of, season_of


def group_rows(timestamps: tuple[str, ...], hazard_levels: np.ndarray) -> dict[str, np.ndarray]:
    """Group key -> indices of the rows in it, sorted by key."""
    members: dict[str, list[int]] = {}
    for idx, (ts, level) in enumerate(zip(timestamps, hazard_levels, strict=True)):
        season = season_of(ts)
        bucket = hazard_bucket(int(level))
        hh = hour_of(ts)
        for d in range(-HOUR_WINDOW, HOUR_WINDOW + 1):
            h = f"h{(hh + d) % 24:02d}"
            for key in (f"{h}|{season}|{bucket}", f"{h}|{season}", h):
                members.setdefault(key, []).append(idx)
        members.setdefault("all", []).append(idx)
    return {k: np.array(v, dtype=int) for k, v in sorted(members.items())}
