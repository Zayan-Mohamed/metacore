"""Generation adequacy: island demand against the in-service generating capacity.

Each island's lead genset is a stiff source in the model, so the power flow alone would let an
island draw more than its gensets can make. This check is what turns "genset outage at peak"
into a rejectable state (docs/contracts/m1-m4-topology-v1.md, item 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from verification.firewall.limits import PhysicsLimitsChecker
from verification.firewall.verifier import PhysicsVerifier
from verification.opendss.circuit import CircuitTwin
from verification.types import (
    Decision,
    DispatchSetpoint,
    LoadShedCommand,
    ProposedControlAction,
    ViolationType,
)

SAMPLES = (Path(__file__).resolve().parents[3] / "packages" / "contracts" / "python"
           / "metacore_contracts" / "schema" / "samples")
INADEQUATE = ViolationType.VIOLATION_TYPE_GENERATION_INADEQUATE


def sample(name: str) -> dict:
    return json.loads((SAMPLES / f"module1_snapshot_v1_{name}.json").read_text())


def scale_island(snap: dict, prefix: str, factor: float) -> dict:
    for ns in snap["node_states"]:
        if ns["node_id"].startswith(prefix) and ns["p_kw"] is not None:
            ns["p_kw"] *= factor
            ns["q_kvar"] *= factor
    return snap


def noop() -> ProposedControlAction:
    return ProposedControlAction(action_id="noop")


def test_capacity_counts_the_lead_genset_and_in_service_units() -> None:
    c = CircuitTwin()
    assert c.apply_snapshot(sample("ditwah_evening")) == []
    elu = PhysicsLimitsChecker.island_adequacy(c)["Eluvaitivu"]
    # Hybrid down: battery and PV out, wind out -> 100 kVA lead genset + 30 kVA backup diesel
    assert elu["capacity_kva"] == pytest.approx(130.0)
    assert elu["station"] == "ELU_PS"
    # 2 x 20.387 kW at the snapshot's power factor
    assert elu["demand_kva"] == pytest.approx(2 * 20.387 / 0.85, rel=1e-3)


def test_ditwah_eluvaitivu_on_diesel_alone_passes_at_its_evening_load() -> None:
    snap = sample("ditwah_evening")
    snap["assets_out_of_service"].append("ELU_HYB_DG")
    verdict = PhysicsVerifier().verify(noop(), snapshot=snap)
    assert verdict.decision == Decision.DECISION_APPROVE, verdict.violations


def test_ditwah_eluvaitivu_fails_when_load_is_pushed_above_capacity() -> None:
    snap = scale_island(sample("ditwah_evening"), "ELU_", 2.2)
    snap["assets_out_of_service"].append("ELU_HYB_DG")
    verdict = PhysicsVerifier().verify(noop(), snapshot=snap)
    assert verdict.decision == Decision.DECISION_REJECT
    inadequate = [v for v in verdict.violations if v.type == INADEQUATE]
    assert [v.element_id for v in inadequate] == ["ELU_PS"]
    assert inadequate[0].limit == pytest.approx(100.0)
    assert inadequate[0].margin_fraction > 0.0
    # The feeders themselves are still inside their ratings: adequacy is the only thing failing
    assert {v.type for v in verdict.violations} == {INADEQUATE}


def test_peak_with_standby_gensets_out_is_inadequate_though_voltages_hold() -> None:
    """The stiff-source blind spot: 0.955 pu looks fine, but 380 kVA cannot carry 460 kVA."""
    snap = sample("nainativu_peak")
    snap["assets_out_of_service"] += ["NAI_DG2", "NAI_DG3"]
    verdict = PhysicsVerifier().verify(noop(), snapshot=snap)
    assert verdict.decision == Decision.DECISION_REJECT
    assert {v.type for v in verdict.violations} == {INADEQUATE}
    assert verdict.violations[0].element_id == "NAI_PS"


def test_shedding_restores_adequacy() -> None:
    snap = sample("nainativu_peak")
    snap["assets_out_of_service"] += ["NAI_DG2", "NAI_DG3"]
    shed = ProposedControlAction(
        action_id="shed-nai-f2",
        load_shed=[LoadShedCommand(node_id="NAI_F2", shed_fraction=0.5, priority_tier=3)])
    assert PhysicsVerifier().verify(shed, snapshot=snap).decision == Decision.DECISION_APPROVE


def test_a_charging_battery_counts_as_demand_not_capacity() -> None:
    c = CircuitTwin()
    before = PhysicsLimitsChecker.island_adequacy(c)["Eluvaitivu"]
    c.set_generator_dispatch("ELU_HYB_BESS", -40.0, 0.0)
    after = PhysicsLimitsChecker.island_adequacy(c)["Eluvaitivu"]
    assert after["capacity_kva"] == pytest.approx(before["capacity_kva"] - 72.0)
    assert after["demand_kva"] > before["demand_kva"]


def test_dispatching_a_standby_genset_does_not_change_capacity() -> None:
    action = ProposedControlAction(
        action_id="dg2-on", dispatch=[DispatchSetpoint(node_id="NAI_DG2", p_kw=100.0)])
    assert PhysicsVerifier().verify(action).decision == Decision.DECISION_APPROVE


def test_dispatching_an_out_of_service_unit_is_malformed() -> None:
    """Ditwah: the hybrid is down, so a battery setpoint would be silently ignored otherwise."""
    action = ProposedControlAction(
        action_id="bess-on", dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=20.0)])
    verdict = PhysicsVerifier().verify(action, snapshot=sample("ditwah_evening"))
    assert verdict.decision == Decision.DECISION_REJECT
    assert verdict.violations[0].type == ViolationType.VIOLATION_TYPE_MALFORMED_ACTION
