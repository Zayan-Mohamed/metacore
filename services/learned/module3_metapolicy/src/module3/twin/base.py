"""The `Twin` interface — the only thing the environment knows about grid physics.

Two interchangeable backends implement it:

  * `SurrogateTwin` (surrogate.py) — fast reduced-order model on M3's side. Used for
    TRAINING, where we need hundreds of thousands of evaluations.
  * `LiveM4Twin` (live_m4.py) — asks the real M4 firewall over HTTP. Used for the final
    TEST of a trained policy. This is the "hybrid" plan: train cheap, test on the real thing.

Both take the same inputs and return the same `TwinOutcome`, so swapping them changes
nothing else in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Disturbance:
    """What the cyclone has done to the grid at one step. Produced by `scenario.cyclone`.

    Fields:
        tripped_lines: names of lines that are physically out (damaged), e.g. {"Line_N8_N11"}.
        pv_factor:     0..1 multiplier on PV output (wind cut-out). 1 = unaffected.
        diesel_factor: 0..1 multiplier on diesel availability. 1 = fully available.
        comms_up:      False when telemetry is lost (M2 reports this as `sensing`).
    """

    tripped_lines: frozenset[str] = frozenset()
    pv_factor: float = 1.0
    diesel_factor: float = 1.0
    comms_up: bool = True


@dataclass
class TwinOutcome:
    """What the grid did in response to an action under a disturbance.

    Fields:
        approved:         True if the firewall would APPROVE (no violations).
        violating_buses:  buses with a voltage/thermal violation.
        served_fraction:  priority-weighted share of load actually served, 0..1
                          (see `reward.priority_weighted_served`).
        energized:        buses that are electrically connected to the source.
        latency_ms:       solve latency reported by the twin (informational).
        source:           "surrogate" or "live_m4".
        detail:           backend-specific extras (violation list, etc.).
    """

    approved: bool
    violating_buses: list[str]
    served_fraction: float
    energized: frozenset[str]
    latency_ms: float = 0.0
    source: str = "surrogate"
    detail: dict = field(default_factory=dict)


class Twin(Protocol):
    def evaluate(self, action: dict, disturbance: Disturbance) -> TwinOutcome:
        """Apply `action` (ProposedControlAction-shaped dict) under `disturbance`."""
        ...
