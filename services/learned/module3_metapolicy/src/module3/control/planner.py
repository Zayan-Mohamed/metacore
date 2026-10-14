"""System 2 for the Meta-RL pipeline — a topology-aware deliberative planner.

This is NOT the final System 2 (that is a separate ClickUp task: load-shed prioritisation
and critical-load ranking, replacing the Week-7 heuristic). It is the smallest planner that
makes the physics-based reward meaningful: it *thinks about the grid*.

Compare the two control paths:

  System 1  (system1.py, reactive)  sees only "how vulnerable are the nodes?" and sheds in
                                    proportion. It does not look at which lines are out.
  System 2  (this file, deliberate) reads the breaker/line status it can see, works out which
                                    buses are cut off from the source, and sheds exactly those.

Inputs  : the set of lines the controller *believes* are tripped (fresh if telemetry is up,
          stale if it is down — so deliberating on a blackout does not help), plus context.
Output  : a ProposedControlAction-shaped dict (origin SYSTEM2) — load_shed only; no breaker
          re-closing (storm-damaged lines stay out) and no dispatch (the stiff source covers
          generation in this twin).
"""

from __future__ import annotations

from module3.twin.surrogate import energized_buses
from module3.twin.topology import GENERATORS, NODES, PRIORITY_TIER


def deliberative_plan(known_tripped: frozenset[str], context: dict | None = None) -> dict:
    """Shed 100% of every bus that the believed topology says is cut off from the source."""
    live = energized_buses(known_tripped)
    load_shed = [
        {"node_id": n, "shed_fraction": 1.0, "priority_tier": PRIORITY_TIER[n]}
        for n in NODES
        if n not in live
    ]
    cut = [c["node_id"] for c in load_shed]
    # A cut-off bus that also hosts a generator cannot be made safe by shedding alone.
    stranded_gen = [n for n in cut if n in GENERATORS]
    note = f"S2 topology plan: isolated={cut or 'none'}"
    if stranded_gen:
        note += f" stranded_gen={stranded_gen}"
    return {
        "origin": "SYSTEM2",
        "breakers": [],
        "load_shed": load_shed,
        "dispatch": [],
        "rationale": note,
    }
