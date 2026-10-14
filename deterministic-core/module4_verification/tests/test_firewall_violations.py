"""Tests for Physics Firewall Violations, Bounds Enforcement, and Severity Scoring."""

from verification.firewall.verifier import PhysicsVerifier
from verification.types import (
    BreakerCommand,
    Decision,
    DispatchSetpoint,
    LoadShedCommand,
    ProposedControlAction,
    ViolationType,
)


def test_nominal_action_approved() -> None:
    verifier = PhysicsVerifier()
    # A safe, nominal load shed action
    action = ProposedControlAction(
        action_id="act-safe-001",
        origin="SYSTEM1",
        breakers=[],
        load_shed=[LoadShedCommand(node_id="N8", shed_fraction=0.10, priority_tier=3)],
        dispatch=[],
        rationale="Nominal shedding",
    )

    verdict = verifier.verify(action)
    assert verdict.decision == Decision.DECISION_APPROVE
    assert len(verdict.violations) == 0
    assert verdict.solve_latency_ms >= 0.0


def test_undervoltage_violation_rejected() -> None:
    verifier = PhysicsVerifier()
    # Opening tie lines and zeroing Island 3 gen leads to undervoltage on Island 3
    action = ProposedControlAction(
        action_id="act-undervolt-001",
        origin="SYSTEM2",
        breakers=[
            BreakerCommand(edge_id="Line_2_3", closed=False),
            BreakerCommand(edge_id="E_crit_1", closed=False),
        ],
        load_shed=[],
        dispatch=[
            DispatchSetpoint(node_id="N8", p_kw=0.0, q_kvar=0.0),
            DispatchSetpoint(node_id="N9", p_kw=0.0, q_kvar=0.0),
        ],
        rationale="Unsafe island isolation",
    )

    verdict = verifier.verify(action)
    assert verdict.decision == Decision.DECISION_REJECT
    assert len(verdict.violations) > 0

    undervolt_violations = [
        v for v in verdict.violations if v.type == ViolationType.VIOLATION_TYPE_UNDERVOLTAGE
    ]
    assert len(undervolt_violations) > 0

    trace = verifier.build_rejection_trace(verdict.action_id, verdict.violations)
    assert trace.severity > 0.0
    assert trace.severity <= 1.0


def test_overvoltage_violation_rejected() -> None:
    verifier = PhysicsVerifier()
    # Injecting massive reactive power creates severe overvoltage
    action = ProposedControlAction(
        action_id="act-overvolt-001",
        origin="SYSTEM2",
        breakers=[],
        load_shed=[],
        dispatch=[DispatchSetpoint(node_id="N1", p_kw=500.0, q_kvar=6000.0)],
        rationale="Extreme over-injection",
    )

    verdict = verifier.verify(action)
    assert verdict.decision == Decision.DECISION_REJECT
    overvolt_violations = [
        v for v in verdict.violations if v.type == ViolationType.VIOLATION_TYPE_OVERVOLTAGE
    ]
    assert len(overvolt_violations) > 0


def test_thermal_overload_violation_rejected() -> None:
    verifier = PhysicsVerifier()
    # Over-dispatching generator beyond line ratings (exceeds ampacity)
    action = ProposedControlAction(
        action_id="act-thermal-001",
        origin="SYSTEM2",
        breakers=[],
        load_shed=[],
        dispatch=[DispatchSetpoint(node_id="N8", p_kw=6000.0, q_kvar=500.0)],
        rationale="Over-dispatch causing thermal exceedance",
    )

    verdict = verifier.verify(action)
    assert verdict.decision == Decision.DECISION_REJECT
    thermal_violations = [
        v for v in verdict.violations if v.type == ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD
    ]
    assert len(thermal_violations) > 0


def test_malformed_action_rejected() -> None:
    verifier = PhysicsVerifier()
    # Providing non-existent node and line identifiers
    action = ProposedControlAction(
        action_id="act-malformed-001",
        origin="SYSTEM1",
        breakers=[BreakerCommand(edge_id="NON_EXISTENT_LINE", closed=False)],
        load_shed=[],
        dispatch=[],
        rationale="Invalid breaker target",
    )

    verdict = verifier.verify(action)
    assert verdict.decision == Decision.DECISION_REJECT
    assert len(verdict.violations) > 0
    assert verdict.violations[0].type == ViolationType.VIOLATION_TYPE_MALFORMED_ACTION
