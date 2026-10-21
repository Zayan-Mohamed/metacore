"""Tests for Physics Firewall Violations, Bounds Enforcement, and Severity Scoring.

Runs on the default model, northern-islands-v1. Every island is a stiff diesel source with two
short feeders, so the violations an action can cause come from the Eluvaitivu hybrid plant: it
sits behind 0.7 km of 400 V cable, where a battery charging or exporting moves the voltage.
"""

import pytest
from verification.firewall.verifier import PhysicsVerifier
from verification.opendss.circuit import CircuitTwin
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
    # A safe, nominal load shed on a non-critical feeder group
    action = ProposedControlAction(
        action_id="act-safe-001",
        origin="SYSTEM1",
        breakers=[],
        load_shed=[LoadShedCommand(node_id="NAI_F2", shed_fraction=0.10, priority_tier=3)],
        dispatch=[],
        rationale="Nominal shedding",
    )

    verdict = verifier.verify(action)
    assert verdict.decision == Decision.DECISION_APPROVE
    assert len(verdict.violations) == 0
    assert verdict.solve_latency_ms >= 0.0


def test_undervoltage_violation_rejected() -> None:
    verifier = PhysicsVerifier()
    # Fast-charging the hybrid battery pulls 72 kW down the Eluvaitivu feeder
    action = ProposedControlAction(
        action_id="act-undervolt-001",
        origin="SYSTEM2",
        breakers=[],
        load_shed=[],
        dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=-72.0, q_kvar=0.0)],
        rationale="Battery fast charge at full rating",
    )

    verdict = verifier.verify(action)
    assert verdict.decision == Decision.DECISION_REJECT
    assert len(verdict.violations) > 0

    undervolt_violations = [
        v for v in verdict.violations if v.type == ViolationType.VIOLATION_TYPE_UNDERVOLTAGE
    ]
    assert {v.element_id for v in undervolt_violations} >= {"ELU_HYB"}

    trace = verifier.build_rejection_trace(verdict.action_id, verdict.violations)
    assert trace.severity > 0.0
    assert trace.severity <= 1.0


def test_overvoltage_violation_rejected() -> None:
    verifier = PhysicsVerifier()
    # Exporting the battery at full rating into a lightly loaded feeder lifts the hybrid bus
    action = ProposedControlAction(
        action_id="act-overvolt-001",
        origin="SYSTEM2",
        breakers=[],
        load_shed=[],
        dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=72.0, q_kvar=0.0)],
        rationale="Battery export at full rating",
    )

    verdict = verifier.verify(action)
    assert verdict.decision == Decision.DECISION_REJECT
    overvolt_violations = [
        v for v in verdict.violations if v.type == ViolationType.VIOLATION_TYPE_OVERVOLTAGE
    ]
    assert len(overvolt_violations) > 0


def test_thermal_overload_violation_rejected() -> None:
    verifier = PhysicsVerifier()
    # Every hybrid unit at once (140 kW split over the in-service units) overloads the 125 A link
    action = ProposedControlAction(
        action_id="act-thermal-001",
        origin="SYSTEM2",
        breakers=[],
        load_shed=[],
        dispatch=[DispatchSetpoint(node_id="ELU_HYB", p_kw=140.0, q_kvar=0.0)],
        rationale="Hybrid plant flat out",
    )

    verdict = verifier.verify(action)
    assert verdict.decision == Decision.DECISION_REJECT
    thermal = {
        v.element_id for v in verdict.violations
        if v.type == ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD
    }
    assert "ELU_HYB_F1" in thermal


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


def test_names_match_exactly_not_by_substring() -> None:
    """``ELU_F1`` must not also shed ``ELU_F1``-prefixed or ``*F1*`` loads on other islands."""
    twin = CircuitTwin()
    assert twin.set_load_shed("ELU_F1", 0.5)
    demand = twin.get_load_demand()
    assert demand["ELU_F1_LOAD"][0] < demand["ELU_F2_LOAD"][0]
    assert demand["NAI_F1_LOAD"][0] == demand["NAI_F2_LOAD"][0]
    assert not twin.set_load_shed("F1", 0.5)


def test_bus_dispatch_is_split_across_in_service_units() -> None:
    twin = CircuitTwin()
    assert twin.set_generator_dispatch("NAI_PS", 100.0, 20.0)
    gens = twin.get_generator_states()
    # OpenDSS stores kW as a float: compare with a tolerance (Linux reads back 50.00000000000001)
    assert gens["NAI_DG2"]["p_kw"] == pytest.approx(50.0)
    assert gens["NAI_DG3"]["p_kw"] == pytest.approx(50.0)
    # A unit name addresses just that unit
    assert twin.set_generator_dispatch("NAI_DG2", 80.0)
    assert twin.get_generator_states()["NAI_DG2"]["p_kw"] == pytest.approx(80.0)
