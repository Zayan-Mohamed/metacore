"""System 1 for the Meta-RL pipeline - the cheap reactive rule.

It sheds load in proportion to node vulnerability, lowest-priority tiers first, never tier 1,
and sheds less when telemetry is lost (missing data -> conservative fallback, not deliberation).
It does not look at which lines are tripped - that is the whole point of the contrast with
System 2 (planner.py). Output uses the ProposedControlAction field names (M3_TO_M4_CONTRACT.md).
"""

from __future__ import annotations

from module3.control.priority import PRIORITY


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


def reactive_plan(context: dict, trigger_reason: str) -> dict:
    """Input: M1-style context dict + M2 trigger_reason. Output: ProposedControlAction dict."""
    return system1_action({**context, "trigger_reason": trigger_reason})
