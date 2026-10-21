"""Grounded translation on the northern-islands-v1 pin (C14)."""

from __future__ import annotations

from pathlib import Path

import pytest
from translation.abductive.attribution import AbductiveAttributor, TopologyEntities, find_pin
from translation.templates.causal_logger import GroundingValidator, TemplateCausalLogger
from translation.types import (
    BreakerCommand,
    CausalLog,
    Decision,
    DispatchSetpoint,
    LoadShedCommand,
    ProposedControlAction,
    RiskAttribution,
    RiskCertificate,
    VerificationVerdict,
    Violation,
    ViolationType,
)

PIN = (Path(__file__).resolve().parents[3] / "packages" / "contracts" / "python"
       / "metacore_contracts" / "schema" / "module1_topology_v1.json")


@pytest.fixture(scope="module")
def topology() -> TopologyEntities:
    return TopologyEntities.load(PIN)


def overvolt(element: str = "ELU_HYB") -> Violation:
    return Violation(type=ViolationType.VIOLATION_TYPE_OVERVOLTAGE, element_id=element,
                     limit=1.05, measured=1.083, margin_fraction=0.0314)


def test_the_pin_is_found_from_the_package() -> None:
    assert find_pin() == PIN.resolve()


def test_entities_come_from_the_pin(topology: TopologyEntities) -> None:
    names = topology.names()
    assert {"NAI_F1", "ELU_HYB_BESS", "DEL_PS_F2", "ELU_F1_LOAD", "Eluvaitivu"} <= names
    assert "N8" not in names
    assert topology.buses_of("ELU_HYB_F1") == {"ELU_HYB", "ELU_F1"}
    assert topology.buses_of("ELU_HYB_BESS") == {"ELU_HYB"}


def test_a_unit_on_the_violating_bus_is_the_cause(topology: TopologyEntities) -> None:
    action = ProposedControlAction(
        action_id="a", origin="SYSTEM2",
        dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=72.0)],
        load_shed=[LoadShedCommand(node_id="NAI_F2", shed_fraction=0.1)])
    [v] = AbductiveAttributor.attribute_violations(action, [overvolt()], topology)
    assert v.attributed_component == "dispatch.ELU_HYB_BESS"


def test_a_line_violation_is_traced_through_its_end_buses(topology: TopologyEntities) -> None:
    action = ProposedControlAction(
        action_id="a", dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=-72.0)])
    line = Violation(type=ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD,
                     element_id="ELU_HYB_F1", limit=125.0, measured=130.0, margin_fraction=0.04)
    [v] = AbductiveAttributor.attribute_violations(action, [line], topology)
    assert v.attributed_component == "dispatch.ELU_HYB_BESS"


def test_a_command_on_another_island_is_never_the_cause(topology: TopologyEntities) -> None:
    action = ProposedControlAction(
        action_id="a", origin="SYSTEM1",
        load_shed=[LoadShedCommand(node_id="NAI_F2", shed_fraction=0.1)],
        breakers=[BreakerCommand(edge_id="DEL_PS_F2", closed=False)])
    [v] = AbductiveAttributor.attribute_violations(action, [overvolt()], topology)
    assert v.attributed_component == "action.system1"


def _rejected_with_risk() -> VerificationVerdict:
    return VerificationVerdict(
        action_id="bess-export-40",
        decision=Decision.DECISION_REJECT,
        violations=[overvolt().model_copy(update={
            "attributed_component": "next-6h scenarios: 95/100"})],
        certificate=RiskCertificate(
            decision=Decision.DECISION_REJECT, n_scenarios=100, n_failures=95,
            lower_bound=0.8659, upper_bound=0.9925, epsilon=0.01, confidence=0.999,
            horizon_h=6, stopped_early=True),
        attribution=[RiskAttribution(player="Eluvaitivu", island="Eluvaitivu", share=0.95,
                                     fraction_of_risk=1.0),
                     RiskAttribution(player="NAI_F1", island="Nainativu", share=0.0)],
    )


def test_the_risk_sentence(topology: TopologyEntities) -> None:
    verdict = _rejected_with_risk()
    log = TemplateCausalLogger.generate_log(verdict, include_latency=False)
    assert log.text.startswith("Action REJECTED: 95.0% of likely next-6-hour conditions "
                               "break a limit")
    assert "95.0 points come from Eluvaitivu load uncertainty" in log.text
    assert "Overvoltage on Bus ELU_HYB" in log.text
    assert log.grounded_entities == ["ELU_HYB", "Eluvaitivu"]  # zero-share players left out
    assert GroundingValidator(topology).check(log, verdict).grounded


def test_an_approval_carries_its_certificate(topology: TopologyEntities) -> None:
    verdict = VerificationVerdict(
        action_id="shed", decision=Decision.DECISION_APPROVE,
        certificate=RiskCertificate(
            decision=Decision.DECISION_APPROVE, n_scenarios=688, n_failures=0,
            lower_bound=0.0, upper_bound=0.00999, epsilon=0.01, confidence=0.999))
    log = TemplateCausalLogger.generate_log(verdict, include_latency=False)
    assert "verified safe" in log.text
    assert "0 of 688 likely next-6-hour conditions" in log.text
    assert "at most 1.00% at 99.9% confidence" in log.text
    assert GroundingValidator(topology).check(log, verdict).grounded


@pytest.mark.parametrize("fake", ["DEL_F9", "N8", "NAI_SUBSTATION"])
def test_a_planted_fake_name_is_rejected(topology: TopologyEntities, fake: str) -> None:
    verdict = _rejected_with_risk()
    log = TemplateCausalLogger.generate_log(verdict, include_latency=False)
    planted = CausalLog(action_id=log.action_id, text=log.text + f" Also check {fake}.",
                        grounded_entities=log.grounded_entities)
    result = GroundingValidator(topology).check(planted, verdict)
    assert not result.grounded and result.ungrounded == [fake]
    claimed = CausalLog(action_id=log.action_id, text=log.text,
                        grounded_entities=[*log.grounded_entities, fake])
    assert not GroundingValidator(topology).check(claimed, verdict).grounded


def test_every_sentence_type_passes_the_grounding_check(topology: TopologyEntities) -> None:
    violations = [
        overvolt(),
        Violation(type=ViolationType.VIOLATION_TYPE_UNDERVOLTAGE, element_id="ELU_F1",
                  limit=0.95, measured=0.92, margin_fraction=-0.03),
        Violation(type=ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD, element_id="ELU_PS_F1",
                  limit=100.0, measured=130.0, margin_fraction=0.3),
        Violation(type=ViolationType.VIOLATION_TYPE_GENERATION_INADEQUATE, element_id="NAI_PS",
                  limit=380.0, measured=460.0, margin_fraction=0.21),
        Violation(type=ViolationType.VIOLATION_TYPE_MALFORMED_ACTION, element_id="Line_2_3",
                  limit=0.0, measured=0.0, margin_fraction=1.0),
    ]
    action = ProposedControlAction(
        action_id="a", dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=72.0)])
    attributed = AbductiveAttributor.attribute_violations(action, violations, topology)
    for decision in (Decision.DECISION_REJECT, Decision.DECISION_UNVERIFIED):
        verdict = VerificationVerdict(action_id="a", decision=decision, violations=attributed,
                                      unverified_reasons=["power flow did not converge"])
        log = TemplateCausalLogger.generate_log(verdict, include_latency=False)
        assert GroundingValidator(topology).check(log, verdict).grounded, log.text
