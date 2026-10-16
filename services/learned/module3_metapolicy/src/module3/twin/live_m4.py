"""LiveM4Twin — evaluates an action on the REAL M4 firewall through the gateway.

Used only for the final test of a trained policy (the 'hybrid' plan). It talks HTTP to
`POST {base_url}/api/module4/verify`, so M3 never imports OpenDSS (ADR 0003).

How the cyclone is applied without changing M4: the disturbance is translated into extra
commands that ride along with the policy's own action —
  * a tripped line   -> a breaker command  {edge_id, closed: False}
  * PV / diesel loss -> a dispatch command lowering that unit's setpoint
The policy's own load-shed commands are passed through untouched. This is the same
interface M4 already exposes, so no change to Hariswara's code is required. (A native
`scenario` field on the request would be cleaner — noted as a future change request.)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import Disturbance, TwinOutcome
from .scoring import DEFAULT_TIER_WEIGHTS, priority_weighted_served, shed_fractions
from .topology import GENERATORS, NODES

# A bus below this per-unit voltage is treated as not serving load.
SERVED_V_MIN_PU = 0.95


def overlay_commands(disturbance: Disturbance) -> tuple[list[dict], list[dict]]:
    """Translate a disturbance into (breaker_commands, dispatch_commands) for M4."""
    breakers = [{"edge_id": ln, "closed": False} for ln in sorted(disturbance.tripped_lines)]
    dispatch = []
    for node, (_name, kind, rated) in GENERATORS.items():
        factor = (
            disturbance.pv_factor
            if kind == "pv"
            else disturbance.diesel_factor
            if kind == "diesel"
            else 1.0
        )
        if factor < 1.0:
            dispatch.append({"node_id": node, "p_kw": rated * factor, "q_kvar": 0.0})
    return breakers, dispatch


def merge_with_overlay(action: dict, disturbance: Disturbance, action_id: str) -> dict:
    """Combine the policy's action with the physical overlay into one verify request.

    Overlay breakers win (a damaged line stays open whatever the policy says). The policy's
    own dispatch wins over the derate overlay for the same node.
    """
    ov_breakers, ov_dispatch = overlay_commands(disturbance)
    tripped = {b["edge_id"] for b in ov_breakers}
    breakers = ov_breakers + [
        {"edge_id": b["edge_id"], "closed": bool(b["closed"])}
        for b in (action.get("breakers") or [])
        if b["edge_id"] not in tripped
    ]
    policy_nodes = {d["node_id"] for d in (action.get("dispatch") or [])}
    dispatch = [d for d in ov_dispatch if d["node_id"] not in policy_nodes] + [
        {"node_id": d["node_id"], "p_kw": float(d["p_kw"]), "q_kvar": float(d.get("q_kvar", 0.0))}
        for d in (action.get("dispatch") or [])
    ]
    return {
        "action_id": action_id,
        "origin": str(action.get("origin", "SYSTEM1")),
        "rationale": str(action.get("rationale", "")),
        "breakers": breakers,
        "load_shed": [
            {
                "node_id": c["node_id"],
                "shed_fraction": float(c["shed_fraction"]),
                "priority_tier": int(c.get("priority_tier", 3)),
            }
            for c in (action.get("load_shed") or [])
        ],
        "dispatch": dispatch,
    }


class LiveM4Twin:
    """Implements the `Twin` protocol against a running gateway."""

    def __init__(
        self, base_url: str = "http://127.0.0.1:8000", timeout_s: float = 10.0, tier_weights=None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.tier_weights = dict(tier_weights or DEFAULT_TIER_WEIGHTS)
        self._n = 0

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/api/module4/verify",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # gateway down, wrong port, ...
            raise ConnectionError(
                f"live M4 not reachable at {self.base_url} ({exc}). Start the gateway: "
                "uv run uvicorn gateway.main:app --app-dir services/gateway --port 8000"
            ) from exc

    def evaluate(self, action: dict, disturbance: Disturbance) -> TwinOutcome:
        if disturbance.grid_lost:
            raise NotImplementedError(
                "The real M4 cannot represent a lost main supply (stiff slack source, finding F3): "
                "tripping Line_Source_N1 collapses every bus. Ask M4 for a grid-lost scenario "
                "(see the M4 change request in docs/module3.md) or evaluate with p_grid_loss=0."
            )
        self._n += 1
        payload = merge_with_overlay(action, disturbance, f"m3-live-{self._n:07d}")
        resp = self._post(payload)
        approved = resp["decision"] == "DECISION_APPROVE"
        violating = sorted({v["element_id"] for v in resp.get("violations", [])})
        voltages = {b["bus_name"].upper(): float(b["voltage_pu"]) for b in resp.get("buses", [])}
        energized = frozenset(n for n in NODES if voltages.get(n, 0.0) >= SERVED_V_MIN_PU)
        served = priority_weighted_served(shed_fractions(action), energized, self.tier_weights)
        return TwinOutcome(
            approved=approved,
            violating_buses=violating,
            served_fraction=served,
            energized=energized,
            latency_ms=float(resp.get("solve_latency_ms", 0.0)),
            source="live_m4",
            detail={"violations": resp.get("violations", [])},
        )
