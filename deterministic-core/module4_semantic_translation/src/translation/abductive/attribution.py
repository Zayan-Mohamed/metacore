"""Abductive Attribution: Maps physical violations back to proposed action components.

ZERO ML DEPENDENCIES.

Entities come from the topology pin (packages/contracts/.../module1_topology_v1.json), not from
name patterns: which bus each line joins, which bus each generator and load sits on, and which
island everything belongs to. A violation is attributed to the action component that touches
the same element, else the same feeder chain, else the same island -- the islands are separate
grids, so a command on one island cannot cause a violation on another.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..types import ProposedControlAction, Violation

PIN_NAME = "module1_topology_v1.json"


def find_pin(start: Path | None = None) -> Path | None:
    """The shared topology pin, found by walking up to the repo's packages/contracts."""
    here = (start or Path(__file__)).resolve()
    for parent in here.parents:
        candidate = (parent / "packages" / "contracts" / "python" / "metacore_contracts"
                     / "schema" / PIN_NAME)
        if candidate.is_file():
            return candidate
    return None


@dataclass
class TopologyEntities:
    """Every name the network defines, and where each sits (all upper-case)."""

    island_of: dict[str, str] = field(default_factory=dict)
    bus_of: dict[str, str] = field(default_factory=dict)  # generator / load -> bus
    line_buses: dict[str, tuple[str, str]] = field(default_factory=dict)
    islands: set[str] = field(default_factory=set)

    @classmethod
    def from_pin(cls, pin: dict) -> TopologyEntities:
        t = cls()
        for b in pin["buses"]:
            t.island_of[b["node_id"].upper()] = b["island"]
            t.islands.add(b["island"])
        for line in pin["lines"]:
            a, b = line["from_node"].upper(), line["to_node"].upper()
            t.line_buses[line["edge_id"].upper()] = (a, b)
            t.island_of[line["edge_id"].upper()] = t.island_of[a]
        for asset in pin["assets"]:
            t.bus_of[asset["element"].upper()] = asset["bus"].upper()
            t.island_of[asset["element"].upper()] = asset["island"]
        for load in pin["loads"]:
            t.bus_of[load["element"].upper()] = load["bus"].upper()
            t.island_of[load["element"].upper()] = load["island"]
        return t

    @classmethod
    def load(cls, path: Path | str | None = None) -> TopologyEntities | None:
        path = Path(path) if path else find_pin()
        return cls.from_pin(json.loads(path.read_text())) if path else None

    def names(self) -> set[str]:
        """Every element id and island name a grounded sentence may mention."""
        return set(self.island_of) | self.islands

    def buses_of(self, element: str) -> set[str]:
        """The bus(es) an element touches: itself, a line's two ends, a unit's bus."""
        e = element.upper()
        if e in self.line_buses:
            return set(self.line_buses[e])
        if e in self.bus_of:
            return {self.bus_of[e]}
        return {e}


class AbductiveAttributor:
    """Infers causal attribution from action commands to violations."""

    @staticmethod
    def attribute_violations(
        action: ProposedControlAction,
        violations: list[Violation],
        topology: TopologyEntities | None = None,
    ) -> list[Violation]:
        """Enriches violations with the plausible attributed_component from the action."""
        topology = topology if topology is not None else TopologyEntities.load()
        components: list[tuple[str, str]] = (
            [(f"dispatch.{d.node_id}", d.node_id) for d in action.dispatch]
            + [(f"load_shed.{ls.node_id}", ls.node_id) for ls in action.load_shed]
            + [(f"breaker.{b.edge_id}", b.edge_id) for b in action.breakers if not b.closed]
        )
        enriched: list[Violation] = []
        for v in violations:
            attributed = _attribute(v.element_id, components, topology)
            if not attributed:
                attributed = (f"action.{action.origin.lower()}" if action.origin
                              else "action.unspecified")
            v_copy = v.model_copy()
            v_copy.attributed_component = attributed
            enriched.append(v_copy)
        return enriched


def _attribute(element: str, components: list[tuple[str, str]],
               topology: TopologyEntities | None) -> str:
    target = element.upper()
    # 1. The command names the violating element itself
    for label, ref in components:
        if ref.upper() == target:
            return label
    if topology is None:
        return ""
    # 2. The command acts on a bus the violating element touches (a unit on that bus, a line
    #    ending there, a shed on that feeder group)
    touched = topology.buses_of(target)
    for label, ref in components:
        if topology.buses_of(ref) & touched:
            return label
    # 3. Same island: the only grid the command can reach
    island = topology.island_of.get(target)
    for label, ref in components:
        if island and topology.island_of.get(ref.upper()) == island:
            return label
    return ""
