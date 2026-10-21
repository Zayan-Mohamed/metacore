"""System 2 — the deliberative planner (critical-load ranking + load-shed prioritisation).

Compare the two control paths:

  System 1  (system1.py, reactive)  sees only "how exposed are the nodes?" and sheds in
                                    proportion. It does not know which lines are out, how much
                                    generation is left, or whether the main supply is gone.
  System 2  (this file, deliberate) works out what the grid actually looks like, then decides
                                    WHO to switch off, in what order, and how much.

What it decides, step by step:
  1. GROUPS    Split the buses into groups still joined by working lines. If the main supply
               is lost, each group has to stand on its own generators.
  2. SUPPLY    Capacity of a group = PV x pv_factor + diesel x diesel_factor + battery.
               (While the main supply is up, a group joined to it has unlimited supply and
               nothing needs shedding; a group cut off from it is simply dead.)
  3. SHORTFALL excess = demand - capacity x (1 - reserve_margin). The reserve margin is the
               headroom System 2 deliberately keeps so a small further loss does not cause a
               blackout. System 1 has no such margin.
  4. RANK      Give loads up in this order: lowest priority tier first; inside a tier, the
               largest load first (so fewer buses are touched); tier 1 (critical) only as a
               last resort, after every other load in the group is already off.
  5. SHED      Walk down the ranking, shedding each load fully until the excess is covered;
               the last load may be shed partially.
  6. DISPATCH  Ask the dispatchable units (diesel, battery) to cover what the PV does not.

Inputs  : the state the controller *believes* — tripped lines, grid_lost, PV and diesel
          factors. These are fresh if telemetry is up and STALE if it is down, which is why
          deliberating during a blackout of data does not help.
Output  : a ProposedControlAction-shaped dict (origin SYSTEM2): load_shed, dispatch,
          breakers=[] (storm-damaged lines are not re-closed), rationale.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from module3.twin.scoring import DEFAULT_TIER_WEIGHTS
from module3.twin.surrogate import (
    connected_groups,
    energized_buses,
    group_capacity_kw,
    unit_capacity_kw,
)
from module3.twin.topology import GENERATORS, LOAD_KW, NODES, PRIORITY_TIER

_EPS = 1e-6


def shed_priority_order(
    nodes: set[str] | list[str], tier_weights: Mapping[int, float] | None = None
) -> list[str]:
    """The order loads are given up: least important first, biggest first inside a tier."""
    w = tier_weights or DEFAULT_TIER_WEIGHTS
    return sorted(nodes, key=lambda n: (w[PRIORITY_TIER[n]], -LOAD_KW[n], n))


def critical_load_ranking(tier_weights: Mapping[int, float] | None = None) -> list[dict]:
    """Every load, most critical first — the 'critical-load ranking' as a readable table."""
    w = tier_weights or DEFAULT_TIER_WEIGHTS
    order = list(reversed(shed_priority_order(list(NODES), w)))
    return [
        {
            "rank": i + 1,
            "node_id": n,
            "tier": PRIORITY_TIER[n],
            "weight": float(w[PRIORITY_TIER[n]]),
            "load_kw": LOAD_KW[n],
        }
        for i, n in enumerate(order)
    ]


def deliberative_plan(
    known_tripped: frozenset[str],
    context: dict | None = None,
    *,
    grid_lost: bool = False,
    pv_factor: float = 1.0,
    diesel_factor: float = 1.0,
    tier_weights: Mapping[int, float] | None = None,
    reserve_margin: float = 0.05,
) -> dict:
    """Plan the shed and dispatch for the believed state. See the module docstring."""
    shed: dict[str, float] = {}
    notes: list[str] = []

    if not grid_lost:
        live = energized_buses(known_tripped)
        cut = [n for n in NODES if n not in live]
        for n in cut:
            shed[n] = 1.0
        notes.append(f"isolated={cut or 'none'}")
        stranded = [n for n in cut if n in GENERATORS]
        if stranded:
            notes.append(f"stranded_gen={stranded}")
        groups: list[set[str]] = []
    else:
        groups = connected_groups(known_tripped, frozenset(), grid_lost=True)
        for g in groups:
            cap = group_capacity_kw(g, pv_factor, diesel_factor)
            demand = sum(LOAD_KW[n] for n in g)
            if cap <= _EPS:
                for n in g:
                    shed[n] = 1.0
                notes.append(f"dead_group={sorted(g)}")
                continue
            excess = demand - cap * (1.0 - reserve_margin)
            for n in shed_priority_order(g, tier_weights):
                if excess <= _EPS:
                    break
                take = min(LOAD_KW[n], excess)
                shed[n] = take / LOAD_KW[n]
                excess -= take
            if sum(1 for n in g if shed.get(n, 0.0) > 0):
                notes.append(f"short_by_kw={max(0.0, demand - cap):.0f} group={sorted(g)}")

    load_shed = [
        {
            "node_id": n,
            # round UP: rounding down would shed a hair too little and risk a blackout
            "shed_fraction": min(1.0, math.ceil(shed[n] * 1e4) / 1e4),
            "priority_tier": PRIORITY_TIER[n],
        }
        for n in NODES
        if shed.get(n, 0.0) > 0.0
    ]

    dispatch: list[dict] = []
    if grid_lost:
        for g in groups:
            served = sum(LOAD_KW[n] * (1.0 - shed.get(n, 0.0)) for n in g)
            pv_out = sum(
                unit_capacity_kw(n, pv_factor, diesel_factor)
                for n in g
                if n in GENERATORS and GENERATORS[n][1] == "pv"
            )
            units = [n for n in g if n in GENERATORS and GENERATORS[n][1] != "pv"]
            total_cap = sum(unit_capacity_kw(n, pv_factor, diesel_factor) for n in units)
            need = max(0.0, served - pv_out)
            if total_cap <= _EPS:
                continue
            for n in units:
                share = unit_capacity_kw(n, pv_factor, diesel_factor) / total_cap
                p_kw = min(unit_capacity_kw(n, pv_factor, diesel_factor), need * share)
                dispatch.append({"node_id": n, "p_kw": float(round(p_kw, 2)), "q_kvar": 0.0})

    return {
        "origin": "SYSTEM2",
        "breakers": [],
        "load_shed": load_shed,
        "dispatch": dispatch,
        "rationale": "S2 plan: " + "; ".join(notes) if notes else "S2 plan: nothing to shed",
    }
