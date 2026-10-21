"""Shared node priority-tier map, used by both System 1 and System 2.

Fixed priority map: N1-N3 critical (tier 1), N4-N6 important (2), rest shedable (3).
"""
from __future__ import annotations

PRIORITY = {f"N{i}": (1 if i <= 3 else 2 if i <= 6 else 3) for i in range(1, 13)}
CRITICAL_NODES = {n for n, t in PRIORITY.items() if t == 1}
