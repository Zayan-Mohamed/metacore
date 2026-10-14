"""The physical reward — what the gate is actually trying to maximise.

    reward = w_serve * served_fraction          (priority-weighted load served, 0..1)
           - w_viol  * violating_bus_fraction   (share of buses the firewall flags)
           - deliberation_cost  [System 2 only] (budget/compute spent thinking)
           - w_latency * path_latency / budget  (slower path costs more; S1 2 ms, S2 18 ms)

The first two terms come from the grid twin and ARE the 'survival' signal. The last two are
the price of deliberating. This replaces the hand-written 7-case table in `gating_env.py`
(which stays, unchanged, for the dashboard demo and as an ablation baseline).
"""

from __future__ import annotations

from dataclasses import dataclass

from module3.twin.base import TwinOutcome
from module3.twin.topology import NODES


@dataclass(frozen=True)
class RewardConfig:
    w_serve: float = 1.0
    w_viol: float = 2.0
    deliberation_cost: float = 0.05
    w_latency: float = 0.05
    latency_s1_ms: float = 2.0
    latency_s2_ms: float = 18.0
    latency_budget_ms: float = 50.0  # real-time control budget (M4 latency test uses 50 ms)

    @classmethod
    def from_dict(cls, d: dict | None) -> RewardConfig:
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def step_reward(outcome: TwinOutcome, used_s2: bool, cfg: RewardConfig) -> tuple[float, dict]:
    """Return (reward, parts). `parts` is logged so every term can be inspected and plotted."""
    serve = cfg.w_serve * outcome.served_fraction
    viol = cfg.w_viol * (len(outcome.violating_buses) / len(NODES))
    delib = cfg.deliberation_cost if used_s2 else 0.0
    lat_ms = cfg.latency_s2_ms if used_s2 else cfg.latency_s1_ms
    lat = cfg.w_latency * lat_ms / cfg.latency_budget_ms
    reward = serve - viol - delib - lat
    return float(reward), {
        "serve": serve,
        "violation": viol,
        "deliberation": delib,
        "latency": lat,
        "latency_ms": lat_ms,
    }
