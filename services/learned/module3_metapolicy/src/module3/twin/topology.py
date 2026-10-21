"""Static description of the Delft 3-island grid, as M3 sees it.

This file is a *copy of facts* from `deterministic-core/.../models/delft_3island.dss`.
M3 may not import OpenDSS or anything from `deterministic-core` (ADR 0003: the firewall
owns the OpenDSS bindings), so the numbers are restated here and
`tests/test_twin_topology.py` parses the `.dss` text and fails if they ever drift apart.

What lives here:
  * which buses exist and how much load / generation sits on each,
  * which lines connect them (and which are overhead feeders that a cyclone can trip),
  * the priority tier of every node (tier 1 = most critical, shed last).

Nothing in this file is learned and nothing here simulates anything.
"""

from __future__ import annotations

from dataclasses import dataclass

SOURCE_BUS = "SourceBus"  # the stiff slack source; everything must connect back to it

# Real buses in the twin. NOTE: N7 and N10 do NOT exist (the .dss skips them), so any
# action naming them is rejected by M4 as MALFORMED_ACTION. The older flat code in
# `synthetic_context.NODE_POOL` still lists N1..N12; the pipeline uses this list instead.
NODES: tuple[str, ...] = ("N1", "N2", "N3", "N4", "N5", "N6", "N8", "N9", "N11", "N12")

# Active load per node, kW (from the `New Load.*` lines).
LOAD_KW: dict[str, float] = {
    "N1": 150.0,
    "N2": 120.0,
    "N3": 100.0,
    "N4": 100.0,
    "N5": 80.0,
    "N6": 60.0,
    "N8": 120.0,
    "N9": 100.0,
    "N11": 80.0,
    "N12": 90.0,
}

# Generators: node -> (name, kind, rated kW).
GENERATORS: dict[str, tuple[str, str, float]] = {
    "N1": ("Gen_Diesel_1", "diesel", 500.0),
    "N2": ("Gen_PV_1", "pv", 200.0),
    "N4": ("Gen_Diesel_2", "diesel", 300.0),
    "N5": ("Gen_PV_2", "pv", 150.0),
    "N8": ("Gen_Diesel_3", "diesel", 400.0),
    "N9": ("Gen_BESS_3", "bess", 200.0),
}

# Priority tier per node, same map the contract's `priority_tier` field carries
# (see `priority.py`): N1-N3 critical (1), N4-N6 important (2), the rest sheddable (3).
PRIORITY_TIER: dict[str, int] = {
    n: (1 if int(n[1:]) <= 3 else 2 if int(n[1:]) <= 6 else 3) for n in NODES
}


@dataclass(frozen=True)
class Line:
    name: str  # exactly the OpenDSS line name (without the "Line." prefix)
    bus1: str
    bus2: str
    kind: str  # "underground" | "overhead" | "subsea"
    norm_amps: float


LINES: tuple[Line, ...] = (
    Line("Line_Source_N1", SOURCE_BUS, "N1", "underground", 300.0),
    Line("Line_N1_N2", "N1", "N2", "underground", 250.0),
    Line("Line_N1_N3", "N1", "N3", "underground", 200.0),
    Line("Line_N4_N5", "N4", "N5", "underground", 200.0),
    Line("Line_N4_N6", "N4", "N6", "underground", 180.0),
    Line("Line_N8_N9", "N8", "N9", "underground", 150.0),
    Line("Line_N8_N11", "N8", "N11", "overhead", 100.0),
    Line("Line_N8_N12", "N8", "N12", "overhead", 100.0),
    Line("Line_1_2", "N2", "N4", "subsea", 150.0),
    Line("Line_2_3", "N5", "N8", "subsea", 150.0),
    Line("E_crit_1", "N1", "N8", "subsea", 200.0),
)

LINE_BY_NAME: dict[str, Line] = {ln.name: ln for ln in LINES}

# The lines a cyclone can realistically damage: wind hits overhead conductors, not cables.
OVERHEAD_LINES: tuple[str, ...] = tuple(ln.name for ln in LINES if ln.kind == "overhead")

# Which node sits at the far end of each overhead feeder (its exposure drives trip risk).
OVERHEAD_END_NODE: dict[str, str] = {"Line_N8_N11": "N11", "Line_N8_N12": "N12"}

# Per-node exposure to wind, 0..1. Nodes at the end of overhead feeders are the most exposed.
# (Modelling assumption, not measured data — see docs/module3.md.)
NODE_EXPOSURE: dict[str, float] = {
    n: (1.0 if n in OVERHEAD_END_NODE.values() else 0.6) for n in NODES
}

TOTAL_LOAD_KW = sum(LOAD_KW.values())


def tier_weight(tier: int, weights: dict[int, float]) -> float:
    """Weight of a priority tier in the 'priority-weighted load served' score."""
    return float(weights[int(tier)])
