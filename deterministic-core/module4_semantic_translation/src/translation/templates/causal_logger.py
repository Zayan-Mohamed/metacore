"""Template Causal Logger: Grounded natural language explanations from violations.

ZERO ML DEPENDENCIES.

Two rules make a sentence trustworthy:

* it is built only from structured evidence -- the violations, the risk certificate and the
  Shapley attribution -- never free text;
* every network name it mentions must exist in the topology pin. The GroundingValidator checks
  the claimed entities *and* scans the finished text for anything that looks like an element id,
  so a name that slipped in from anywhere else is caught (EU AI Act Art. 13, transparency: the
  operator can trace every word back to the network and the evidence).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..abductive.attribution import TopologyEntities
from ..types import (
    CausalLog,
    Decision,
    Generator,
    VerificationVerdict,
    Violation,
    ViolationType,
)

# What an element id looks like in this network: upper-case words joined by underscores with at
# least one underscore (NAI_F1, ELU_HYB_BESS, DEL_PS_F2) -- or a legacy N-number (N8).
_ELEMENT_LIKE = re.compile(r"\b(?:[A-Z][A-Z0-9]*_[A-Z0-9_]*[A-Z0-9]|N\d{1,2})\b")
# Upper-case tokens that are vocabulary, not network names.
_VOCABULARY = {"REJECTED", "APPROVED", "NOT", "VERIFIED", "AC"}


@dataclass
class GroundingResult:
    grounded: bool
    ungrounded: list[str]


class GroundingValidator:
    """Rejects any explanation that names something the network does not contain."""

    def __init__(self, topology: TopologyEntities | None) -> None:
        self.topology = topology

    def check(self, log: CausalLog, verdict: VerificationVerdict) -> GroundingResult:
        evidence = {v.element_id.upper() for v in verdict.violations}
        allowed = evidence | ({n.upper() for n in self.topology.names()}
                              if self.topology else set())
        if self.topology is None:
            # No pin at hand: fall back to the original invariant (entities from the evidence).
            allowed |= {a.player.upper() for a in verdict.attribution}
        claimed = {e.upper() for e in log.grounded_entities}
        mentioned = {m for m in _ELEMENT_LIKE.findall(log.text) if m not in _VOCABULARY}
        bad = sorted((claimed | mentioned) - allowed - {"GLOBAL_CIRCUIT"})
        # A malformed action names an element that does not exist -- that is the point of the
        # sentence, and the element is quoted from the action, so it is evidence, not a claim.
        malformed = {v.element_id.upper() for v in verdict.violations
                     if v.type == ViolationType.VIOLATION_TYPE_MALFORMED_ACTION}
        bad = [b for b in bad if b not in malformed]
        return GroundingResult(not bad, bad)


class TemplateCausalLogger:
    """Generates grounded causal diagnostic text from verification results."""

    @staticmethod
    def generate_log(verdict: VerificationVerdict, include_latency: bool = True) -> CausalLog:
        """Constructs a grounded causal log from a VerificationVerdict."""
        if verdict.decision == Decision.DECISION_UNVERIFIED:
            return TemplateCausalLogger._unverified_log(verdict)
        lat_str = (
            f" (power flow solved in {verdict.solve_latency_ms:.1f} ms)" if include_latency else ""
        )
        if verdict.decision == Decision.DECISION_APPROVE or not verdict.violations:
            risk = TemplateCausalLogger.risk_sentence(verdict)
            return CausalLog(
                action_id=verdict.action_id,
                text=(
                    f"Action verified safe: All bus voltages and line ampacities remain within "
                    f"statutory bounds{lat_str}." + (f" {risk}" if risk else "")
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
            phrases.append(_phrase(v))

        check_lat = (
            f" (power flow checked in {verdict.solve_latency_ms:.1f} ms)" if include_latency else ""
        )
        risk = TemplateCausalLogger.risk_sentence(verdict)
        for a in verdict.attribution:
            if a.share > 0 and a.player not in grounded_entities:
                grounded_entities.append(a.player)
        summary_text = (
            "Action REJECTED: "
            + (f"{risk} Worst cases: " if risk else "")
            + "; ".join(phrases)
            + f"{check_lat}."
        )

        return CausalLog(
            action_id=verdict.action_id,
            text=summary_text,
            grounded_entities=grounded_entities,
            generator=Generator.GENERATOR_TEMPLATE,
        )

    @staticmethod
    def risk_sentence(verdict: VerificationVerdict) -> str:
        """One sentence from the certificate and the attribution, or "" without a certificate.

        e.g. "13.8 % of likely next-6-hour conditions break a limit (99.9 % bound 18.3 % vs
        target 1 %, 100 scenarios); 13.8 points come from Eluvaitivu load uncertainty."
        """
        c = verdict.certificate
        if c is None:
            return ""
        rate = c.n_failures / c.n_scenarios if c.n_scenarios else 0.0
        conf = f"{c.confidence:.1%}".replace(".0%", "%")
        if c.n_failures == 0:
            return (f"Risk certificate: 0 of {c.n_scenarios} likely next-{c.horizon_h}-hour "
                    f"conditions break a limit, so the risk is at most {c.upper_bound:.2%} at "
                    f"{conf} confidence (target {c.epsilon:.0%}).")
        text = (f"{rate:.1%} of likely next-{c.horizon_h}-hour conditions break a limit "
                f"({conf} bound {c.lower_bound:.2%} to {c.upper_bound:.2%} vs target "
                f"{c.epsilon:.0%}, {c.n_scenarios} scenarios)")
        top = [a for a in verdict.attribution if a.share > 0]
        if top:
            lead = top[0]
            text += (f"; {lead.share * 100:.1f} points come from {lead.player} load "
                     f"uncertainty")
        return text + "."

    @staticmethod
    def _unverified_log(verdict: VerificationVerdict) -> CausalLog:
        """UNVERIFIED is never phrased as safe: it says what M4 could not establish."""
        reasons = list(verdict.unverified_reasons)
        grounded: list[str] = []
        for v in verdict.violations:
            if v.element_id not in grounded:
                grounded.append(v.element_id)
            if v.type == ViolationType.VIOLATION_TYPE_NON_CONVERGENCE:
                reasons.append("AC power flow solver failed to converge")
        detail = "; ".join(dict.fromkeys(reasons)) or "the checker could not establish safety"
        risk = TemplateCausalLogger.risk_sentence(verdict)
        return CausalLog(
            action_id=verdict.action_id,
            text=f"Action NOT VERIFIED (treat as not approved): {detail}."
                 + (f" {risk}" if risk else ""),
            grounded_entities=grounded,
            generator=Generator.GENERATOR_TEMPLATE,
        )


def _phrase(v: Violation) -> str:
    element = v.element_id
    cause_str = f" [Attributed to: {v.attributed_component}]" if v.attributed_component else ""
    sign_str = "+" if v.margin_fraction >= 0 else "-"
    margin_pct = abs(v.margin_fraction) * 100.0
    if v.type == ViolationType.VIOLATION_TYPE_UNDERVOLTAGE:
        return (f"Undervoltage on Bus {element} ({v.measured:.4f} pu < "
                f"limit {v.limit:.4f} pu, margin: {sign_str}{margin_pct:.1f}%){cause_str}")
    if v.type == ViolationType.VIOLATION_TYPE_OVERVOLTAGE:
        return (f"Overvoltage on Bus {element} ({v.measured:.4f} pu > "
                f"limit {v.limit:.4f} pu, margin: {sign_str}{margin_pct:.1f}%){cause_str}")
    if v.type == ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD:
        return (f"Thermal overload on Line {element} ({v.measured:.1f} A > "
                f"ampacity {v.limit:.1f} A, margin: {sign_str}{margin_pct:.1f}%){cause_str}")
    if v.type == ViolationType.VIOLATION_TYPE_GENERATION_INADEQUATE:
        return (f"Generation inadequate at station {element} ({v.measured:.1f} kVA demand > "
                f"{v.limit:.1f} kVA in service, margin: {sign_str}{margin_pct:.1f}%){cause_str}")
    if v.type == ViolationType.VIOLATION_TYPE_NON_CONVERGENCE:
        return "AC power flow solver failed to converge under topological changes"
    if v.type == ViolationType.VIOLATION_TYPE_MALFORMED_ACTION:
        return f"Malformed action: Element {element} does not exist in target grid topology"
    return f"Unspecified violation on element {element}"
