"""SurrogateTwin — a fast, reduced-order stand-in for the OpenDSS firewall.

Why it exists: PPO needs hundreds of thousands of grid evaluations; the real firewall takes
~13 ms each (it recompiles the circuit every call), so training against it would take hours.
This model reproduces the *verdict logic* the real twin showed in our probes, in microseconds.

What it models (and nothing more):
  1. CONNECTIVITY. Starting at the source bus, walk every line that is not tripped. A bus
     that cannot be reached is de-energised.
  2. VOLTAGE VIOLATION. A de-energised bus reads 0 pu in OpenDSS, which breaks the 0.95 pu
     limit -> UNDERVOLTAGE. The one exception found by probing: a bus whose every incident
     line is out AND that has no generator AND whose load is fully shed is simply not part
     of the solved circuit, so it reports nothing.
  3. MALFORMED ACTIONS. Naming a bus or line the twin does not have is rejected outright.

What it does NOT model (be honest in the paper):
  * Thermal overload and over-voltage from extreme dispatch. The planner never produces
    such setpoints, so the surrogate never flags them.
  * Generation adequacy. The twin has a stiff 100 MVA slack source, so PV derating or
    diesel loss never causes a violation in the real twin either. Those disturbances are
    carried in the scenario and the gate's observation, not in the verdict.

It is validated, not trusted: `scripts/validate_surrogate.py` replays random scenarios
against the live M4 and reports the agreement rate.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .base import Disturbance, TwinOutcome
from .scoring import DEFAULT_TIER_WEIGHTS, priority_weighted_served, shed_fractions
from .topology import (
    GENERATORS,
    LINE_BY_NAME,
    LINES,
    LOAD_KW,
    NODES,
    SOURCE_BUS,
)

_EPS = 1e-6


def energized_buses(tripped: frozenset[str], extra_open: frozenset[str] = frozenset()) -> set[str]:
    """Buses reachable from the source through lines that are neither tripped nor opened."""
    adj: dict[str, list[str]] = defaultdict(list)
    for ln in LINES:
        if ln.name in tripped or ln.name in extra_open:
            continue
        adj[ln.bus1].append(ln.bus2)
        adj[ln.bus2].append(ln.bus1)
    seen = {SOURCE_BUS}
    stack = [SOURCE_BUS]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


class SurrogateTwin:
    """Microsecond-scale grid verdict. Implements the `Twin` protocol."""

    def __init__(self, tier_weights: Mapping[int, float] | None = None) -> None:
        self.tier_weights = dict(tier_weights or DEFAULT_TIER_WEIGHTS)

    def evaluate(self, action: dict, disturbance: Disturbance) -> TwinOutcome:
        # Lines opened by the action itself (breaker commands) behave like trips.
        opened = frozenset(
            str(b.get("edge_id"))
            for b in (action.get("breakers") or [])
            if not b.get("closed", True)
        )
        shed = shed_fractions(action)

        # --- malformed commands: unknown bus / line -> M4 rejects the whole action ---
        malformed: list[str] = []
        for b in action.get("breakers") or []:
            if str(b.get("edge_id")) not in LINE_BY_NAME:
                malformed.append(str(b.get("edge_id")))
        for node in shed:
            if node not in LOAD_KW:
                malformed.append(node)
        for d in action.get("dispatch") or []:
            if str(d.get("node_id")) not in GENERATORS:
                malformed.append(str(d.get("node_id")))
        if malformed:
            return TwinOutcome(
                approved=False,
                violating_buses=sorted(set(malformed)),
                served_fraction=0.0,
                energized=frozenset(),
                source="surrogate",
                detail={"malformed": sorted(set(malformed))},
            )

        live = energized_buses(disturbance.tripped_lines, opened)

        violating: list[str] = []
        for node in NODES:
            if node in live:
                continue
            incident_all_out = all(
                (ln.name in disturbance.tripped_lines or ln.name in opened)
                for ln in LINES
                if node in (ln.bus1, ln.bus2)
            )
            residual_kw = LOAD_KW[node] * (1.0 - shed.get(node, 0.0))
            has_gen = node in GENERATORS
            floating_ok = incident_all_out and not has_gen and residual_kw <= _EPS
            if not floating_ok:
                violating.append(node)

        served = priority_weighted_served(shed, live & set(NODES), self.tier_weights)
        return TwinOutcome(
            approved=not violating,
            violating_buses=violating,
            served_fraction=served,
            energized=frozenset(live),
            source="surrogate",
        )
