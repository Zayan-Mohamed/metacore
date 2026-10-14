"""System 2 — deliberative survival-optimisation control-path stand-in.

Emits a dict with ProposedControlAction field names from module3.proto so
run_demo.py can write contract-shaped JSON directly. No real OpenDSS /
optimizer — a visibly different output from System 1 so the sample stream
shows a real S1 vs S2 distinction.
"""
from __future__ import annotations

from priority import PRIORITY


def system2_action(obs_raw: dict, rng) -> dict:
    """Deliberative path: protect tier-1 nodes; also adjust dispatch setpoints.

    Sheds less on critical nodes and adds a small diesel/BESS dispatch bump.

    INPUT  obs_raw (dict) -- reads: max_node_vulnerability (0..1), top_at_risk_nodes.
           rng -- numpy Generator, only used for +-5 kW jitter on the dispatch setpoint.
    OUTPUT dict with origin="SYSTEM2", breakers=[E_crit_1 closed], load_shed=[...],
           dispatch=[N4 setpoint], rationale (str).
    USES   priority.PRIORITY. Never sheds tier 1.
    NOTE   This is the WEEK-7 stand-in used by the dashboard demo. The Meta-RL pipeline uses
           module3.control.planner.deliberative_plan instead.
    """
    vuln = float(obs_raw.get("max_node_vulnerability", 0.0))
    top = list(obs_raw.get("top_at_risk_nodes") or ["N12", "N11"])
    load_shed = []
    remaining = min(1.0, vuln * 0.55)  # more conservative total shed
    # Shed only tier 3 first, then tier 2; never tier 1.
    ordered = sorted(top, key=lambda n: -PRIORITY.get(n, 3))
    for node in ordered:
        if remaining <= 0:
            break
        tier = PRIORITY.get(node, 3)
        if tier == 1:
            continue
        frac = min(remaining, 0.25 if tier == 3 else 0.15)
        load_shed.append(
            {"node_id": node, "shed_fraction": float(frac), "priority_tier": tier}
        )
        remaining -= frac

    # Dispatch bump on a generation node (stand-in optimizer output).
    gen_node = "N4"
    p_kw = 50.0 + 150.0 * vuln + float(rng.uniform(-5, 5))
    dispatch = [{"node_id": gen_node, "p_kw": float(p_kw), "q_kvar": 10.0}]
    # Keep a breaker closed on a critical feeder as a protective action.
    breakers = [{"edge_id": "E_crit_1", "closed": True}]
    return {
        "origin": "SYSTEM2",
        "breakers": breakers,
        "load_shed": load_shed,
        "dispatch": dispatch,
        "rationale": f"S2 survival opt vuln={vuln:.2f} protect-tier1",
    }
