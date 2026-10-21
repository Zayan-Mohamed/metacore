"""Physics Firewall Verifier: Gatekeeper for AI-proposed actions.

ZERO ML DEPENDENCIES.

Check A of the risk-certified shield: the action on this hour's snapshot. Three outcomes:

* APPROVE    -- the solve converged, every number is finite, and no limit is broken. This is the
                only path that approves, and it sits at the end of ``verify``.
* REJECT     -- the action breaks a limit, or names grid elements that do not exist.
* UNVERIFIED -- M4 could not establish the physics: a malformed snapshot, non-convergence, a
                non-finite number, a solver exception or a blown time budget. Fail-safe: nothing
                that goes wrong inside the checker can turn into an approval.
"""

import math
import time

from ..opendss.circuit import CircuitTwin
from ..powerflow.action_applicator import ActionApplicator
from ..powerflow.solver import PowerFlowSolver
from ..types import (
    Decision,
    ProposedControlAction,
    RejectionTrace,
    VerificationVerdict,
    Violation,
    ViolationType,
)
from .limits import DEFAULT_PROFILE, PhysicsLimitsChecker, SafetyLimitsConfig


def _malformed(element_id: str, component: str) -> Violation:
    return Violation(
        type=ViolationType.VIOLATION_TYPE_MALFORMED_ACTION,
        element_id=element_id,
        limit=0.0,
        measured=0.0,
        margin_fraction=1.0,
        attributed_component=component,
    )


def action_number_problems(action: ProposedControlAction) -> list[Violation]:
    """A setpoint that is NaN or infinite is a malformed action, whatever element it names.

    OpenDSS happily accepts NaN kW and then reports NaN voltages, which no limit comparison
    catches (NaN < 0.95 is False). Refuse the number before it reaches the solver.
    """
    bad: list[Violation] = []
    for d in action.dispatch:
        if not (math.isfinite(d.p_kw) and math.isfinite(d.q_kvar)):
            bad.append(_malformed(d.node_id, f"dispatch.{d.node_id}: non-finite setpoint"))
    for ls in action.load_shed:
        if not math.isfinite(ls.shed_fraction):
            bad.append(_malformed(ls.node_id, f"load_shed.{ls.node_id}: non-finite fraction"))
    return bad


def snapshot_number_problems(snapshot: dict) -> list[str]:
    """Snapshot demand that is non-finite or negative cannot describe a real hour."""
    problems: list[str] = []
    for ns in snapshot.get("node_states", []) or []:
        for key in ("p_kw", "q_kvar"):
            value = ns.get(key)
            if value is None:
                continue
            try:
                x = float(value)
            except (TypeError, ValueError):
                problems.append(f"{ns.get('node_id')!r} {key} is not a number: {value!r}")
                continue
            if not math.isfinite(x):
                problems.append(f"{ns.get('node_id')!r} {key} is not finite: {value!r}")
            elif key == "p_kw" and x < 0.0:
                problems.append(f"{ns.get('node_id')!r} p_kw is negative: {x}")
    return problems


def solved_state_is_finite(circuit: CircuitTwin) -> bool:
    """Every bus voltage and line current the limits read is a finite number."""
    if not all(math.isfinite(v) for v in circuit.get_bus_voltages_pu().values()):
        return False
    return all(math.isfinite(d["max_amps"]) for d in circuit.get_line_loadings().values())


class PhysicsVerifier:
    """Synchronous physical verification firewall for MetaCore."""

    def __init__(
        self,
        circuit_twin: CircuitTwin | None = None,
        profile: str = DEFAULT_PROFILE,
        time_budget_ms: float | None = None,
    ) -> None:
        self.circuit = circuit_twin or CircuitTwin()
        self.applicator = ActionApplicator(self.circuit)
        self.limits_checker = PhysicsLimitsChecker(SafetyLimitsConfig.from_profile(profile))
        # Wall-clock budget for one verify call. OpenDSS cannot be interrupted mid-solve, so the
        # budget is checked after it returns: an over-budget answer is UNVERIFIED, never late
        # approval.
        self.time_budget_ms = time_budget_ms

    def verify(
        self, action: ProposedControlAction, snapshot: dict | None = None
    ) -> VerificationVerdict:
        """Evaluates a proposed action against OpenDSS physical constraints.

        ``snapshot`` is an optional M1 GridStateSnapshot. Without it the action is judged against
        the model's base case.
        """
        started = time.perf_counter()
        try:
            return self._verify(action, snapshot, started)
        except Exception as exc:  # noqa: BLE001 -- the fail-safe: any fault is "not verified"
            return self._unverified(action.action_id, [f"verifier error: {exc!r}"])
        finally:
            try:
                self.circuit.reset_to_base()
            except Exception:  # noqa: BLE001 -- the verdict above already stands
                pass

    def _verify(
        self, action: ProposedControlAction, snapshot: dict | None, started: float
    ) -> VerificationVerdict:
        # 1. Reset circuit to pristine base state
        self.circuit.reset_to_base()

        # 2. Load this hour's grid state, if M1 supplied one. A snapshot that does not fit the
        # twin means M4 does not know the state it is judging: UNVERIFIED, never approve on a
        # half-applied state.
        if snapshot is not None:
            problems = snapshot_number_problems(snapshot)
            if not problems:
                problems = self.circuit.apply_snapshot(snapshot)
            if problems:
                return self._unverified(action.action_id, [f"snapshot: {p}" for p in problems])

        # 3. Apply proposed control action. Unknown elements and non-finite setpoints are the
        # action's fault: REJECT with the evidence.
        malformed = action_number_problems(action)
        if not malformed:
            malformed = self.applicator.apply_action(action)
        if malformed:
            return self._build_verdict(
                action_id=action.action_id,
                decision=Decision.DECISION_REJECT,
                violations=malformed,
                latency_ms=0.0,
            )

        # 4. Execute AC power flow solve
        converged, latency_ms = PowerFlowSolver.solve_snapshot()

        # 5. Check physics bounds
        violations = self.limits_checker.check_limits(self.circuit, converged)

        if not converged:
            return self._unverified(
                action.action_id, ["power flow did not converge"], violations, latency_ms
            )
        if not solved_state_is_finite(self.circuit):
            return self._unverified(
                action.action_id, ["solved state has non-finite values"], [], latency_ms
            )
        if violations:
            return self._build_verdict(
                action_id=action.action_id,
                decision=Decision.DECISION_REJECT,
                violations=violations,
                latency_ms=latency_ms,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if self.time_budget_ms is not None and elapsed_ms > self.time_budget_ms:
            return self._unverified(
                action.action_id,
                [f"time budget exceeded: {elapsed_ms:.1f} ms > {self.time_budget_ms:.1f} ms"],
                [],
                latency_ms,
            )

        # 6. The single approval path: converged, finite, inside every limit, on time.
        return self._build_verdict(
            action_id=action.action_id,
            decision=Decision.DECISION_APPROVE,
            violations=[],
            latency_ms=latency_ms,
        )

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

    def _unverified(
        self,
        action_id: str,
        reasons: list[str],
        violations: list[Violation] | None = None,
        latency_ms: float = 0.0,
    ) -> VerificationVerdict:
        verdict = self._build_verdict(
            action_id=action_id,
            decision=Decision.DECISION_UNVERIFIED,
            violations=violations or [],
            latency_ms=latency_ms,
        )
        verdict.unverified_reasons = reasons
        return verdict

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
            limit_profile=self.limits_checker.config.profile,
        )
