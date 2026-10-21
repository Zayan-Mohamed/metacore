"""The fail-safe: no exception, timeout, solver failure or bad number can produce APPROVE.

Approval has exactly one path in PhysicsVerifier._verify. Each test below breaks one thing the
verifier depends on and asserts the verdict is UNVERIFIED (M4 could not establish the physics)
or REJECT (the action itself is bad) -- never APPROVE.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
from verification.firewall.verifier import PhysicsVerifier
from verification.powerflow.solver import PowerFlowSolver
from verification.types import (
    Decision,
    DispatchSetpoint,
    ProposedControlAction,
    ViolationType,
)

SAMPLES = (Path(__file__).resolve().parents[3] / "packages" / "contracts" / "python"
           / "metacore_contracts" / "schema" / "samples")
NAN, INF = float("nan"), float("inf")


def calm() -> dict:
    return json.loads((SAMPLES / "module1_snapshot_v1_calm.json").read_text())


def noop() -> ProposedControlAction:
    return ProposedControlAction(action_id="noop")


def test_the_unbroken_case_still_approves() -> None:
    assert PhysicsVerifier().verify(noop(), snapshot=calm()).decision == Decision.DECISION_APPROVE


def test_non_convergence_is_unverified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PowerFlowSolver, "solve_snapshot", staticmethod(lambda: (False, 1.0)))
    verdict = PhysicsVerifier().verify(noop())
    assert verdict.decision == Decision.DECISION_UNVERIFIED
    assert verdict.violations[0].type == ViolationType.VIOLATION_TYPE_NON_CONVERGENCE
    assert "converge" in verdict.unverified_reasons[0]


def test_a_solver_exception_is_unverified(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> tuple[bool, float]:
        raise RuntimeError("OpenDSS engine fault")

    monkeypatch.setattr(PowerFlowSolver, "solve_snapshot", staticmethod(boom))
    verdict = PhysicsVerifier().verify(noop())
    assert verdict.decision == Decision.DECISION_UNVERIFIED
    assert "OpenDSS engine fault" in verdict.unverified_reasons[0]


def test_a_blown_time_budget_is_unverified() -> None:
    verdict = PhysicsVerifier(time_budget_ms=0.0).verify(noop())
    assert verdict.decision == Decision.DECISION_UNVERIFIED
    assert "time budget" in verdict.unverified_reasons[0]


def test_a_non_finite_solved_state_is_unverified(monkeypatch: pytest.MonkeyPatch) -> None:
    v = PhysicsVerifier()
    real = v.circuit.get_bus_voltages_pu
    monkeypatch.setattr(v.circuit, "get_bus_voltages_pu",
                        lambda: {**real(), "ELU_F1": NAN})
    verdict = v.verify(noop())
    assert verdict.decision == Decision.DECISION_UNVERIFIED


@pytest.mark.parametrize("bad", [NAN, INF, -INF, -5.0, "lots"])
def test_a_bad_snapshot_load_is_unverified(bad: object) -> None:
    snap = calm()
    snap["node_states"][0]["p_kw"] = bad
    verdict = PhysicsVerifier().verify(noop(), snapshot=snap)
    assert verdict.decision == Decision.DECISION_UNVERIFIED


@pytest.mark.parametrize(("p", "q"), list(itertools.product([NAN, INF, 10.0], repeat=2))[:-1])
def test_a_non_finite_setpoint_is_a_malformed_action(p: float, q: float) -> None:
    action = ProposedControlAction(
        action_id="nan", dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=p, q_kvar=q)])
    verdict = PhysicsVerifier().verify(action, snapshot=calm())
    assert verdict.decision == Decision.DECISION_REJECT
    assert verdict.violations[0].type == ViolationType.VIOLATION_TYPE_MALFORMED_ACTION


def test_fault_injection_no_nan_anywhere_can_approve() -> None:
    """Every combination of a NaN in the snapshot and/or the action: never APPROVE."""
    v = PhysicsVerifier()
    for in_snapshot, in_action in itertools.product([False, True], repeat=2):
        if not (in_snapshot or in_action):
            continue
        snap = calm()
        if in_snapshot:
            snap["node_states"][3]["q_kvar"] = NAN
        p_kw = NAN if in_action else 5.0
        action = ProposedControlAction(
            action_id="fi", dispatch=[DispatchSetpoint(node_id="NAI_DG2", p_kw=p_kw)])
        assert v.verify(action, snapshot=snap).decision != Decision.DECISION_APPROVE


def test_the_circuit_is_reset_after_a_failure() -> None:
    v = PhysicsVerifier()
    snap = calm()
    snap["node_states"][0]["p_kw"] = NAN
    v.verify(noop(), snapshot=snap)
    assert v.verify(noop()).decision == Decision.DECISION_APPROVE
