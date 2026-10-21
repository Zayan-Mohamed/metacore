"""The shared M1 -> M4 topology (northern-islands-v1) under M4's own physics.

M1 generates the model and the snapshot fixtures but may not import OpenDSS (ADR 0003), so this
is where they are compiled and solved. The files are read by path from the contracts package:
this lane installs M4 alone and must not depend on Module 1's code.

northern-islands-v1 is M4's default model. M4 ships a vendored copy (so the module builds on its
own); the tests below hold it byte-identical to the contracts copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import opendssdirect as dss
import pytest
from verification.firewall.verifier import PhysicsVerifier
from verification.opendss.circuit import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PIN_PATH,
    DEFAULT_TOPOLOGY_ID,
    CircuitTwin,
)
from verification.types import (
    BreakerCommand,
    Decision,
    LoadShedCommand,
    ProposedControlAction,
    ViolationType,
)

CONTRACTS = (Path(__file__).resolve().parents[3] / "packages" / "contracts" / "python"
             / "metacore_contracts")
DSS = CONTRACTS / "topology" / "northern_islands_v1.dss"
PIN_PATH = CONTRACTS / "schema" / "module1_topology_v1.json"
PIN = json.loads(PIN_PATH.read_text())
GOLDEN = Path(__file__).parent / "golden" / "ditwah_evening_noop.json"
TOPOLOGY_ID = PIN["topology_id"]


def sample(name: str) -> dict:
    return json.loads((CONTRACTS / "schema" / "samples" / f"module1_snapshot_v1_{name}.json")
                      .read_text())


def twin() -> CircuitTwin:
    return CircuitTwin(str(DSS), topology_id=TOPOLOGY_ID)


def noop(action_id: str = "noop") -> ProposedControlAction:
    return ProposedControlAction(action_id=action_id, rationale="no control action")


def island_of(bus: str) -> str:
    return bus.upper().split("_", 1)[0]


# ------------------------------------------------------------ the model itself

def test_model_buses_match_the_pin_exactly() -> None:
    c = twin()
    assert set(c.get_all_buses()) == {b["node_id"].upper() for b in PIN["buses"]}


def test_no_line_joins_two_islands() -> None:
    twin()
    for name in dss.Lines.AllNames():
        dss.Circuit.SetActiveElement(f"Line.{name}")
        b1, b2 = (b.split(".")[0] for b in dss.CktElement.BusNames())
        assert island_of(b1) == island_of(b2), f"{name} joins {b1} and {b2}"


def test_every_island_has_its_own_source() -> None:
    twin()
    sources = set()
    for name in dss.Vsources.AllNames():
        dss.Circuit.SetActiveElement(f"Vsource.{name}")
        sources.add(island_of(dss.CktElement.BusNames()[0]))
    assert sources == {island_of(b["node_id"]) for b in PIN["buses"]}


def test_base_case_converges_inside_statutory_limits() -> None:
    verdict = PhysicsVerifier(twin()).verify(noop())
    assert verdict.decision == Decision.DECISION_APPROVE, verdict.violations


def test_out_of_service_wind_is_disabled_in_the_model() -> None:
    twin()
    dss.Circuit.SetActiveElement("Generator.ELU_HYB_WIND")
    assert not dss.CktElement.Enabled()


# ------------------------------------------------------------ snapshots

@pytest.mark.parametrize("name", ["calm", "nainativu_peak", "ditwah_evening"])
def test_every_fixture_snapshot_applies_and_is_physically_feasible(name: str) -> None:
    verdict = PhysicsVerifier(twin()).verify(noop(name), snapshot=sample(name))
    assert verdict.decision == Decision.DECISION_APPROVE, verdict.violations


def test_snapshot_sets_this_hours_demand_and_skips_missing_fields() -> None:
    c = twin()
    snap = sample("nainativu_peak")
    assert c.apply_snapshot(snap) == []
    dss.Loads.Name("NAI_F1_LOAD")
    want = next(n["p_kw"] for n in snap["node_states"] if n["node_id"] == "NAI_F1")
    assert dss.Loads.kW() == pytest.approx(want, rel=1e-6)
    hyb = next(n for n in snap["node_states"] if n["node_id"] == "ELU_HYB")
    assert hyb["p_kw"] is None  # missing in the snapshot, so the model value stays untouched


def test_hybrid_down_takes_battery_and_pv_out_of_service() -> None:
    c = twin()
    snap = sample("ditwah_evening")
    assert snap["hybrid_plant_down"] is True
    assert c.apply_snapshot(snap) == []
    for element in ("ELU_HYB_BESS", "ELU_HYB_PV"):
        dss.Circuit.SetActiveElement(f"Generator.{element}")
        assert not dss.CktElement.Enabled()


def test_load_shed_is_a_fraction_of_the_snapshot_demand() -> None:
    v = PhysicsVerifier(twin())
    snap = sample("nainativu_peak")
    shed = ProposedControlAction(
        action_id="shed-nai-f2",
        load_shed=[LoadShedCommand(node_id="NAI_F2", shed_fraction=0.5, priority_tier=3)])
    assert v.verify(shed, snapshot=snap).decision == Decision.DECISION_APPROVE


def test_a_snapshot_for_another_topology_is_unverified() -> None:
    """M4 does not know the state it would judge: not approved, and not the action's fault."""
    snap = dict(sample("calm"), topology_id="delft-3island-v0")
    verdict = PhysicsVerifier(twin()).verify(noop(), snapshot=snap)
    assert verdict.decision == Decision.DECISION_UNVERIFIED
    assert "topology" in verdict.unverified_reasons[0]


def test_a_snapshot_naming_an_unknown_bus_is_unverified() -> None:
    snap = sample("calm")
    snap["node_states"].append({"node_id": "N8", "p_kw": 10.0, "q_kvar": 1.0,
                                "quality": {"p_kw": "QUALITY_INTERPOLATED"}})
    verdict = PhysicsVerifier(twin()).verify(noop(), snapshot=snap)
    assert verdict.decision == Decision.DECISION_UNVERIFIED
    assert any("N8" in r for r in verdict.unverified_reasons)


def test_the_firewall_still_bites_on_this_model() -> None:
    """Three times the record peak on Nainativu's feeders must breach a limit."""
    snap = sample("nainativu_peak")
    for ns in snap["node_states"]:
        if ns["node_id"].startswith("NAI_F") and ns["p_kw"] is not None:
            ns["p_kw"] *= 3.0
            ns["q_kvar"] *= 3.0
    verdict = PhysicsVerifier(twin()).verify(noop(), snapshot=snap)
    assert verdict.decision == Decision.DECISION_REJECT
    assert {v.type for v in verdict.violations} & {
        ViolationType.VIOLATION_TYPE_UNDERVOLTAGE, ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD}


def test_tie_line_commands_are_malformed_because_no_tie_lines_exist() -> None:
    action = ProposedControlAction(
        action_id="tie", breakers=[BreakerCommand(edge_id="E_crit_1", closed=True)])
    verdict = PhysicsVerifier(twin()).verify(action)
    assert verdict.decision == Decision.DECISION_REJECT
    assert verdict.violations[0].type == ViolationType.VIOLATION_TYPE_MALFORMED_ACTION


# ------------------------------------------------------------ default model

def test_default_twin_is_northern_islands() -> None:
    c = CircuitTwin()
    assert c.topology_id == DEFAULT_TOPOLOGY_ID == TOPOLOGY_ID
    assert "NAI_F1" in c.get_all_buses()
    assert "N8" not in c.get_all_buses()


def test_vendored_model_and_pin_match_the_contracts_copy() -> None:
    assert DEFAULT_MODEL_PATH.read_bytes() == DSS.read_bytes()
    assert DEFAULT_PIN_PATH.read_bytes() == PIN_PATH.read_bytes()


def _solved_state(c: CircuitTwin) -> dict:
    return {
        "voltages_pu": {b: round(v, 5) for b, v in sorted(c.get_bus_voltages_pu().items())},
        "line_amps": {n: round(d["max_amps"], 3)
                      for n, d in sorted(c.get_line_loadings().items())},
    }


def test_ditwah_evening_matches_the_golden_solution() -> None:
    """The Ditwah evening snapshot solves to the recorded voltages and currents.

    Regenerate after a deliberate model change with ``M4_UPDATE_GOLDEN=1 pytest``.
    """
    import os

    from verification.powerflow.solver import PowerFlowSolver

    c = CircuitTwin()
    assert c.apply_snapshot(sample("ditwah_evening")) == []
    converged, _ = PowerFlowSolver.solve_snapshot()
    assert converged
    state = _solved_state(c)
    if os.environ.get("M4_UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    golden = json.loads(GOLDEN.read_text())
    assert state["voltages_pu"] == pytest.approx(golden["voltages_pu"], abs=1e-5)
    assert state["line_amps"] == pytest.approx(golden["line_amps"], abs=1e-3)
