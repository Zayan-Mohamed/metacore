"""Physics Limits Checker: Evaluates voltage and thermal ampacity boundaries.

ZERO ML DEPENDENCIES.
"""

from dataclasses import dataclass

from ..opendss.circuit import CircuitTwin
from ..types import Violation, ViolationType


@dataclass
class SafetyLimitsConfig:
    """Configurable statutory bounds for island microgrids."""

    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    thermal_overload_margin: float = 0.0  # 0.0 means strictly <= NormAmps


class PhysicsLimitsChecker:
    """Evaluates circuit state against physics boundaries and constructs Violation objects."""

    def __init__(self, config: SafetyLimitsConfig | None = None) -> None:
        self.config = config or SafetyLimitsConfig()

    def check_limits(self, circuit: CircuitTwin, converged: bool) -> list[Violation]:
        """Runs comprehensive checks on voltage, line loading, and solver convergence."""
        violations: list[Violation] = []

        # 1. Check AC power-flow convergence
        if not converged:
            violations.append(
                Violation(
                    type=ViolationType.VIOLATION_TYPE_NON_CONVERGENCE,
                    element_id="GLOBAL_CIRCUIT",
                    limit=1.0,
                    measured=0.0,
                    margin_fraction=-1.0,
                    attributed_component="powerflow_solver",
                )
            )
            return violations

        # 2. Check nodal bus voltages (0.95 <= V_pu <= 1.05)
        bus_voltages = circuit.get_bus_voltages_pu()
        for bus, v_pu in bus_voltages.items():
            if v_pu < self.config.v_min_pu:
                margin = (v_pu - self.config.v_min_pu) / self.config.v_min_pu
                violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_UNDERVOLTAGE,
                        element_id=bus,
                        limit=self.config.v_min_pu,
                        measured=round(v_pu, 4),
                        margin_fraction=round(margin, 4),
                    )
                )
            elif v_pu > self.config.v_max_pu:
                margin = (v_pu - self.config.v_max_pu) / self.config.v_max_pu
                violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_OVERVOLTAGE,
                        element_id=bus,
                        limit=self.config.v_max_pu,
                        measured=round(v_pu, 4),
                        margin_fraction=round(margin, 4),
                    )
                )

        # 3. Check thermal line loading margins (I <= NormAmps)
        line_loadings = circuit.get_line_loadings()
        for line, data in line_loadings.items():
            if data["enabled"] < 0.5:
                continue

            max_amps = data["max_amps"]
            norm_amps = data["norm_amps"]
            margin = data["margin_fraction"]

            limit_threshold = norm_amps * (1.0 + self.config.thermal_overload_margin)
            if norm_amps > 0.0 and max_amps > limit_threshold:
                violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD,
                        element_id=line,
                        limit=norm_amps,
                        measured=round(max_amps, 2),
                        margin_fraction=round(margin, 4),
                    )
                )

        return violations
