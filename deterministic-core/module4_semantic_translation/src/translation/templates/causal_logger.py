"""Template Causal Logger: Grounded natural language explanations from violations.

ZERO ML DEPENDENCIES.
"""

from ..types import (
    CausalLog,
    Decision,
    Generator,
    VerificationVerdict,
    ViolationType,
)


class TemplateCausalLogger:
    """Generates grounded causal diagnostic text from verification results."""

    @staticmethod
    def generate_log(verdict: VerificationVerdict, include_latency: bool = True) -> CausalLog:
        """Constructs a grounded causal log from a VerificationVerdict."""
        lat_str = (
            f" (power flow solved in {verdict.solve_latency_ms:.1f} ms)" if include_latency else ""
        )
        if verdict.decision == Decision.DECISION_APPROVE or not verdict.violations:
            return CausalLog(
                action_id=verdict.action_id,
                text=(
                    f"Action verified safe: All bus voltages and line ampacities remain within "
                    f"statutory bounds{lat_str}."
                ),
                grounded_entities=[],
                generator=Generator.GENERATOR_TEMPLATE,
            )

        # Build explanation phrases for each violation
        phrases: list[str] = []
        grounded_entities: list[str] = []

        for v in verdict.violations:
            element = v.element_id
            if element not in grounded_entities:
                grounded_entities.append(element)

            cause_str = (
                f" [Attributed to: {v.attributed_component}]" if v.attributed_component else ""
            )

            sign_str = "+" if v.margin_fraction >= 0 else "-"
            margin_pct = abs(v.margin_fraction) * 100.0

            if v.type == ViolationType.VIOLATION_TYPE_UNDERVOLTAGE:
                phrases.append(
                    f"Undervoltage on Bus {element} ({v.measured:.4f} pu < "
                    f"limit {v.limit:.4f} pu, margin: {sign_str}{margin_pct:.1f}%){cause_str}"
                )
            elif v.type == ViolationType.VIOLATION_TYPE_OVERVOLTAGE:
                phrases.append(
                    f"Overvoltage on Bus {element} ({v.measured:.4f} pu > "
                    f"limit {v.limit:.4f} pu, margin: {sign_str}{margin_pct:.1f}%){cause_str}"
                )
            elif v.type == ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD:
                phrases.append(
                    f"Thermal overload on Line {element} ({v.measured:.1f} A > "
                    f"ampacity {v.limit:.1f} A, margin: {sign_str}{margin_pct:.1f}%){cause_str}"
                )
            elif v.type == ViolationType.VIOLATION_TYPE_NON_CONVERGENCE:
                phrases.append("AC power flow solver failed to converge under topological changes")
            elif v.type == ViolationType.VIOLATION_TYPE_MALFORMED_ACTION:
                phrases.append(
                    f"Malformed action: Element {element} does not exist in target grid topology"
                )
            else:
                phrases.append(f"Unspecified violation on element {element}")

        check_lat = (
            f" (power flow checked in {verdict.solve_latency_ms:.1f} ms)" if include_latency else ""
        )
        summary_text = "Action REJECTED: " + "; ".join(phrases) + f"{check_lat}."

        return CausalLog(
            action_id=verdict.action_id,
            text=summary_text,
            grounded_entities=grounded_entities,
            generator=Generator.GENERATOR_TEMPLATE,
        )
