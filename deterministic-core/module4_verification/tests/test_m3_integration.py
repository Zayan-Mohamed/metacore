"""Integration Tests for Module 4 with M3 Sample Action Stream."""

import json
from pathlib import Path

from verification.firewall.verifier import PhysicsVerifier
from verification.types import Decision, ProposedControlAction

M3_SAMPLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "services"
    / "learned"
    / "module3_metapolicy"
    / "sample_m3_to_m4.jsonl"
)

FALLBACK_M3_ACTIONS = [
    {
        "action_id": "eb9ad02e-d30a-44a6-8f50-175c6cec2f29",
        "origin": "SYSTEM1",
        "breakers": [],
        "load_shed": [{"node_id": "N8", "shed_fraction": 0.1117, "priority_tier": 3}],
        "dispatch": [],
        "rationale": "S1 reactive shed vuln=0.14",
    },
    {
        "action_id": "58fd3b5e-08eb-4e99-add5-344afa1565c1",
        "origin": "SYSTEM2",
        "breakers": [{"edge_id": "E_crit_1", "closed": True}],
        "load_shed": [
            {"node_id": "N11", "shed_fraction": 0.25, "priority_tier": 3},
            {"node_id": "N8", "shed_fraction": 0.0983, "priority_tier": 3},
        ],
        "dispatch": [{"node_id": "N4", "p_kw": 147.37, "q_kvar": 10.0}],
        "rationale": "S2 survival opt vuln=0.63 protect-tier1",
    },
]


def test_m3_sample_actions_execution() -> None:
    """Verifies all actions from M3 stream against OpenDSS physics."""
    verifier = PhysicsVerifier()
    actions: list[ProposedControlAction] = []

    if M3_SAMPLE_PATH.exists():
        with open(M3_SAMPLE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("message_type") == "ProposedControlAction" or "breakers" in data:
                    actions.append(ProposedControlAction.model_validate(data))
    else:
        actions = [ProposedControlAction.model_validate(raw) for raw in FALLBACK_M3_ACTIONS]

    assert len(actions) > 0, "No actions loaded from M3 dataset!"

    for action in actions:
        verdict = verifier.verify(action)
        assert verdict.action_id == action.action_id
        assert verdict.decision in (Decision.DECISION_APPROVE, Decision.DECISION_REJECT)
        assert verdict.solve_latency_ms >= 0.0

        trace = verifier.build_rejection_trace(verdict.action_id, verdict.violations)
        assert 0.0 <= trace.severity <= 1.0
