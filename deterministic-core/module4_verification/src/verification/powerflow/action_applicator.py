"""Action Applicator: Translates high-level ProposedControlAction into OpenDSS state changes.

ZERO ML DEPENDENCIES.
"""

from ..opendss.circuit import CircuitTwin
from ..types import ProposedControlAction, Violation, ViolationType


class ActionApplicator:
    """Applies control actions to the OpenDSS circuit twin."""

    def __init__(self, circuit_twin: CircuitTwin) -> None:
        self.circuit = circuit_twin

    def apply_action(self, action: ProposedControlAction) -> list[Violation]:
        """Applies all commands in action. Returns violations if elements are malformed."""
        malformed_violations: list[Violation] = []

        # 1. Apply line / breaker commands
        for cmd in action.breakers:
            success = self.circuit.set_line_state(cmd.edge_id, cmd.closed)
            if not success:
                malformed_violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_MALFORMED_ACTION,
                        element_id=cmd.edge_id,
                        limit=0.0,
                        measured=0.0,
                        margin_fraction=1.0,
                        attributed_component=f"breaker.{cmd.edge_id}",
                    )
                )

        # 2. Apply load shedding commands
        for cmd in action.load_shed:
            success = self.circuit.set_load_shed(cmd.node_id, cmd.shed_fraction)
            if not success:
                malformed_violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_MALFORMED_ACTION,
                        element_id=cmd.node_id,
                        limit=0.0,
                        measured=0.0,
                        margin_fraction=1.0,
                        attributed_component=f"load_shed.{cmd.node_id}",
                    )
                )

        # 3. Apply generator / BESS dispatch setpoints
        for cmd in action.dispatch:
            success = self.circuit.set_generator_dispatch(cmd.node_id, cmd.p_kw, cmd.q_kvar)
            if not success:
                malformed_violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_MALFORMED_ACTION,
                        element_id=cmd.node_id,
                        limit=0.0,
                        measured=0.0,
                        margin_fraction=1.0,
                        attributed_component=f"dispatch.{cmd.node_id}",
                    )
                )

        return malformed_violations
