"""System 1 — cheap reactive control-path stand-in.

Emits a dict with ProposedControlAction field names from module3.proto so
run_demo.py can write contract-shaped JSON directly. No real OpenDSS /
optimizer — a visibly different output from System 2 so the sample stream
shows a real S1 vs S2 distinction.
"""
from __future__ import annotations

from priority import PRIORITY


def system1_action(obs_raw: dict) -> dict:
    """Cheap heuristic: shed load proportional to vulnerability, lowest tiers first.

    Under a sensing drop (comms blackout), shed more conservatively — Duwaragie's
    guidance: missing data → conservative System-1 fallback, not deliberation.
    Leaves dispatch untouched. origin = SYSTEM1.

    INPUT  obs_raw (dict) -- reads: max_node_vulnerability (0..1), trigger_reason
           (none|value|sensing|both), top_at_risk_nodes (list of node ids).
    OUTPUT dict with origin="SYSTEM1", breakers=[], load_shed=[{node_id, shed_fraction,
           priority_tier}], dispatch=[], rationale (str).
    USES   priority.PRIORITY (node -> tier). Never sheds tier 1.
    """
    vuln = float(obs_raw.get("max_node_vulnerability", 0.0))
    reason = str(obs_raw.get("trigger_reason", "none"))
    sensing = reason in ("sensing", "both")
    top = list(obs_raw.get("top_at_risk_nodes") or ["N12", "N11", "N10"])
    candidates = sorted(top, key=lambda n: -PRIORITY.get(n, 3))
    load_shed = []
    # Sensing-loss: cut less aggressively (can't see the grid; don't over-shed).
    shed_scale = 0.45 if sensing else 0.8
    remaining = min(1.0, vuln * shed_scale)
    for node in candidates:
        if remaining <= 0:
            break
        tier = PRIORITY.get(node, 3)
        if tier == 1:
            continue
        frac = min(remaining, 0.25 if sensing else (0.35 if tier == 3 else 0.2))
        load_shed.append(
            {"node_id": node, "shed_fraction": float(frac), "priority_tier": tier}
        )
        remaining -= frac
    tag = "sensing-fallback" if sensing else "reactive"
    return {
        "origin": "SYSTEM1",
        "breakers": [],
        "load_shed": load_shed,
        "dispatch": [],
        "rationale": f"S1 {tag} shed vuln={vuln:.2f}",
    }
