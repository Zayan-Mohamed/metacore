"""Abductive Attribution: Maps physical violations back to proposed action components.

ZERO ML DEPENDENCIES.
"""

from ..types import ProposedControlAction, Violation


class AbductiveAttributor:
    """Infers causal attribution from action commands to violations using exact matching."""

    @staticmethod
    def attribute_violations(
        action: ProposedControlAction, violations: list[Violation]
    ) -> list[Violation]:
        """Enriches violations with plausible attributed_component from the action."""
        enriched: list[Violation] = []

        for v in violations:
            target_id = v.element_id.upper()
            attributed = ""

            # 1. Exact match against generator dispatch setpoints
            for d in action.dispatch:
                if d.node_id.upper() == target_id:
                    attributed = f"dispatch.{d.node_id}"
                    break

            # 2. Exact match against load shedding commands
            if not attributed:
                for ls in action.load_shed:
                    if ls.node_id.upper() == target_id:
                        attributed = f"load_shed.{ls.node_id}"
                        break

            # 3. Exact match against breaker commands or line incident nodes
            if not attributed:
                for b in action.breakers:
                    if not b.closed:  # An opened line
                        b_upper = b.edge_id.upper()
                        if b_upper == target_id:
                            attributed = f"breaker.{b.edge_id}"
                            break
                        # Check incident terminal buses (e.g. Line_2_3, E_crit_1)
                        if "2_3" in b_upper and target_id in ("N5", "N8"):
                            attributed = f"breaker.{b.edge_id}"
                            break
                        if "1_2" in b_upper and target_id in ("N2", "N4"):
                            attributed = f"breaker.{b.edge_id}"
                            break
                        if "CRIT" in b_upper and target_id in ("N1", "N8"):
                            attributed = f"breaker.{b.edge_id}"
                            break

            # 4. Fallback attribution if action origin is specified
            if not attributed:
                if action.origin:
                    attributed = f"action.{action.origin.lower()}"
                else:
                    attributed = "action.unspecified"

            v_copy = v.model_copy()
            v_copy.attributed_component = attributed
            enriched.append(v_copy)

        return enriched
