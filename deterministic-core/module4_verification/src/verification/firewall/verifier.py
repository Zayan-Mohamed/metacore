"""Physics Firewall Verifier: Gatekeeper for AI-proposed actions.

ZERO ML DEPENDENCIES.
"""

from ..opendss.circuit import CircuitTwin
from ..powerflow.action_applicator import ActionApplicator
from ..powerflow.solver import PowerFlowSolver
from ..types import (
    Decision,
    ProposedControlAction,
    RejectionTrace,
    VerificationVerdict,
    Violation,
)
from .limits import PhysicsLimitsChecker


class PhysicsVerifier:
    """Synchronous physical verification firewall for MetaCore."""

    def __init__(self, circuit_twin: CircuitTwin | None = None) -> None:
        self.circuit = circuit_twin or CircuitTwin()
        self.applicator = ActionApplicator(self.circuit)
        self.limits_checker = PhysicsLimitsChecker()

    def verify(self, action: ProposedControlAction) -> VerificationVerdict:
        """Evaluates a proposed action against OpenDSS physical constraints."""
        try:
            # 1. Reset circuit to pristine base state
            self.circuit.reset_to_base()

            # 2. Apply proposed control action
            malformed_violations = self.applicator.apply_action(action)
            if malformed_violations:
                return self._build_verdict(
                    action_id=action.action_id,
                    decision=Decision.DECISION_REJECT,
                    violations=malformed_violations,
                    latency_ms=0.0,
                )

            # 3. Execute AC power flow solve
            converged, latency_ms = PowerFlowSolver.solve_snapshot()

            # 4. Check physics bounds
            violations = self.limits_checker.check_limits(self.circuit, converged)

            # 5. Formulate verdict
            decision = (
                Decision.DECISION_APPROVE if len(violations) == 0 else Decision.DECISION_REJECT
            )

            return self._build_verdict(
                action_id=action.action_id,
                decision=decision,
                violations=violations,
                latency_ms=latency_ms,
            )
        finally:
            self.circuit.reset_to_base()

    def build_rejection_trace(self, action_id: str, violations: list[Violation]) -> RejectionTrace:
        """Constructs feedback RejectionTrace with normalized severity for Module 2."""
        if not violations:
            return RejectionTrace(action_id=action_id, violations=[], severity=0.0)

        # Normalize severity: worst absolute margin capped at 1.0
        max_margin = max(abs(v.margin_fraction) for v in violations)
        severity = min(1.0, max(0.0, max_margin))

        return RejectionTrace(
            action_id=action_id,
            violations=violations,
            severity=round(severity, 4),
        )

    def _build_verdict(
        self,
        action_id: str,
        decision: Decision,
        violations: list[Violation],
        latency_ms: float,
    ) -> VerificationVerdict:
        return VerificationVerdict(
            action_id=action_id,
            decision=decision,
            violations=violations,
            solve_latency_ms=latency_ms,
        )
