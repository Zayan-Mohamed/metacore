"""Priority-weighted load served — the 'survival' number the reward is built on.

    served = sum over nodes of  weight(tier) * load_kw * (1 - shed_fraction) * energized
    total  = sum over nodes of  weight(tier) * load_kw
    served_fraction = served / total          (0 = blackout, 1 = every kW served)

So serving a tier-1 (critical) kilowatt counts 3x a tier-3 kilowatt with the default
weights {1: 3, 2: 2, 3: 1}. Shedding a node, or losing it to a tripped feeder, costs
exactly its weighted load.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .topology import LOAD_KW, NODES, PRIORITY_TIER

DEFAULT_TIER_WEIGHTS: dict[int, float] = {1: 3.0, 2: 2.0, 3: 1.0}


def shed_fractions(action: dict) -> dict[str, float]:
    """Collapse an action's `load_shed` list into {node_id: shed_fraction in 0..1}.

    If a node appears twice the larger shed wins (the twin applies commands in order, so
    the last wins; taking the max is the conservative reading for scoring).
    """
    out: dict[str, float] = {}
    for cmd in action.get("load_shed") or []:
        node = str(cmd.get("node_id"))
        frac = max(0.0, min(1.0, float(cmd.get("shed_fraction", 0.0))))
        out[node] = max(out.get(node, 0.0), frac)
    return out


def priority_weighted_served(
    shed: Mapping[str, float],
    energized: Iterable[str],
    weights: Mapping[int, float] = DEFAULT_TIER_WEIGHTS,
) -> float:
    """Return the served fraction (0..1) for a shed plan and a set of energised buses."""
    live = set(energized)
    served = 0.0
    total = 0.0
    for node in NODES:
        w = float(weights[PRIORITY_TIER[node]]) * LOAD_KW[node]
        total += w
        if node in live:
            served += w * (1.0 - float(shed.get(node, 0.0)))
    return served / total if total > 0 else 0.0
