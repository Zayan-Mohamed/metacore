"""Module 4 — Physical Verification & Causal Translation Gateway Router.

REST surface for the risk-certified safety shield:

* ``POST /module4/verify`` judges a proposed action on a recorded hour, either with the plain
  snapshot gate (Check A only, today's M4 and baseline B0) or with the risk certificate (Check A,
  then Check B on 688 seeded scenarios of the next six hours, conformal PID in storm mode and
  Shapley risk attribution). It returns the verdict, the certificate, the attribution, the
  grounded operator explanation and the solved grid state for the dashboard.
* ``GET /module4/presets``, ``/hours``, ``/profiles`` feed the dashboard's pickers.

Hours are replayed (ADR 0004): the three committed M1 fixture snapshots load from the contracts
package; any other hour in 2024-2025 is fetched from Module 1's snapshot producer.
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from translation.abductive.attribution import AbductiveAttributor, TopologyEntities
from translation.templates.causal_logger import GroundingValidator, TemplateCausalLogger
from translation.types import ProposedControlAction as TransProposedAction
from translation.types import VerificationVerdict as TransVerdict
from verification.firewall.limits import PhysicsLimitsChecker, available_profiles
from verification.opendss.circuit import load_pin
from verification.powerflow.solver import PowerFlowSolver
from verification.risk.calibration import Calibration
from verification.risk.certifier import CertifierConfig, RiskCertifiedVerifier
from verification.risk.history import default_paths
from verification.risk.pid import ReplayHistory
from verification.types import (
    BreakerCommand,
    Decision,
    DispatchSetpoint,
    LoadShedCommand,
    ProposedControlAction,
    ViolationType,
)

from . import module1

router = APIRouter(prefix="/module4", tags=["module4"])

FIXTURES = module1.CONTRACTS_PY / "metacore_contracts" / "schema" / "samples"

HOURS: dict[str, dict[str, str]] = {
    "calm": {
        "timestamp": "2025-10-05T12:00",
        "title": "Calm noon, 5 Oct 2025",
        "description": "An ordinary midday hour: demand is low and the evening ramp is ahead.",
    },
    "nainativu_peak": {
        "timestamp": "2025-07-20T20:00",
        "title": "Nainativu record peak, 20 Jul 2025",
        "description": "The highest evening demand on record; the tightest voltages (0.955 pu).",
    },
    "ditwah_evening": {
        "timestamp": "2025-11-28T19:00",
        "title": "Cyclone Ditwah evening, 28 Nov 2025",
        "description": "Peak storm hazard; the Eluvaitivu hybrid plant is down.",
    },
}
DEFAULT_HOUR = HOURS["calm"]["timestamp"]

# One OpenDSS engine per process: every solve goes through this lock.
_LOCK = threading.Lock()


class VerifyRequest(BaseModel):
    action_id: str = Field(default="act-web-001", description="Action unique identifier")
    origin: str = Field(default="SYSTEM1", description="SYSTEM1 or SYSTEM2")
    rationale: str = Field(default="", description="Operator or AI rationale")
    breakers: list[BreakerCommand] = Field(default_factory=list)
    load_shed: list[LoadShedCommand] = Field(default_factory=list)
    dispatch: list[DispatchSetpoint] = Field(default_factory=list)
    hour: str = Field(default=DEFAULT_HOUR, description="Recorded hour, ISO local time")
    outages: list[str] = Field(default_factory=list,
                               description="Extra generators out of service this hour")
    profile: str = Field(default="METACORE-STRICT", description="Limit profile id")
    mode: Literal["certified", "plain"] = Field(
        default="certified", description="certified = Check A + B; plain = Check A only (B0)")
    pid: Literal["auto", "on", "off"] = Field(default="auto", description="Conformal PID mode")


class ViolationDTO(BaseModel):
    type: ViolationType
    element_id: str
    limit: float
    measured: float
    margin_fraction: float
    attributed_component: str = ""


class CausalLogDTO(BaseModel):
    action_id: str
    text: str
    grounded_entities: list[str]
    generator: str
    grounded: bool = True
    ungrounded: list[str] = Field(default_factory=list)


class GridBusState(BaseModel):
    bus_name: str
    island: str
    voltage_pu: float
    status: str  # "SAFE", "UNDERVOLTAGE", "OVERVOLTAGE"


class GridLineState(BaseModel):
    line_name: str
    current_amps: float
    norm_amps: float
    margin_fraction: float
    is_closed: bool
    status: str  # "NORMAL", "OVERLOAD", "TRIPPED"


class IslandAdequacy(BaseModel):
    island: str
    station: str
    demand_kva: float
    capacity_kva: float
    status: str  # "ADEQUATE", "INADEQUATE"


class VerifyResponse(BaseModel):
    action_id: str
    decision: Decision
    check_a_decision: Decision
    mode: str
    hour: str
    profile: str
    v_min_pu: float
    v_max_pu: float
    snapshot_source: str
    solve_latency_ms: float
    violations: list[ViolationDTO]
    unverified_reasons: list[str] = Field(default_factory=list)
    rejection_severity: float
    certificate: dict[str, Any] | None = None
    attribution: list[dict[str, Any]] = Field(default_factory=list)
    causal_log: CausalLogDTO
    buses: list[GridBusState]
    lines: list[GridLineState]
    adequacy: list[IslandAdequacy]


def _payload(action_id: str, origin: str, rationale: str, hour: str, *,
             load_shed: list | None = None, dispatch: list | None = None,
             breakers: list | None = None, outages: list | None = None) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "origin": origin,
        "rationale": rationale,
        "breakers": breakers or [],
        "load_shed": load_shed or [],
        "dispatch": dispatch or [],
        "hour": HOURS[hour]["timestamp"],
        "outages": outages or [],
    }


PRESETS: dict[str, dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # APPROVED under both gates
    # -------------------------------------------------------------------------
    "nominal_safe": {
        "title": "1. Nominal S1 Feeder Shed",
        "description": "Routine minor load shed on Nainativu's non-critical feeder group F2.",
        "expected": {"plain": "DECISION_APPROVE", "certified": "DECISION_APPROVE"},
        "payload": _payload(
            "act-001-nominal-safe", "SYSTEM1", "S1 reactive shed vuln=0.14", "calm",
            load_shed=[{"node_id": "NAI_F2", "shed_fraction": 0.1117, "priority_tier": 3}]),
    },
    "cyclone_survival": {
        "title": "2. Cyclone Ditwah Survival Dispatch",
        "description": (
            "Ditwah evening, hybrid down: start Eluvaitivu's backup genset and shed "
            "non-critical feeders."
        ),
        "expected": {"plain": "DECISION_APPROVE", "certified": "DECISION_APPROVE"},
        "payload": _payload(
            "act-002-cyclone-survival", "SYSTEM2", "S2 survival opt vuln=0.63 protect-tier1",
            "ditwah_evening",
            load_shed=[{"node_id": "ELU_F2", "shed_fraction": 0.25, "priority_tier": 3},
                       {"node_id": "DEL_F2", "shed_fraction": 0.0983, "priority_tier": 3}],
            dispatch=[{"node_id": "ELU_HYB_DG", "p_kw": 20.0, "q_kvar": 10.0}]),
    },
    "peak_power_sharing": {
        "title": "3. Peak Standby Genset Dispatch",
        "description": (
            "Record-peak evening: bring Nainativu's and Delft's standby gensets on."
        ),
        "expected": {"plain": "DECISION_APPROVE", "certified": "DECISION_APPROVE"},
        "payload": _payload(
            "act-003-peak-sharing", "SYSTEM2", "S2 survival opt vuln=0.88 protect-tier1",
            "nainativu_peak",
            load_shed=[{"node_id": "DEL_F2", "shed_fraction": 0.2325, "priority_tier": 3}],
            dispatch=[{"node_id": "NAI_DG2", "p_kw": 120.0, "q_kvar": 40.0},
                      {"node_id": "DEL_DG2", "p_kw": 100.0, "q_kvar": 30.0}]),
    },
    # -------------------------------------------------------------------------
    # The case the risk certificate exists for: safe now, unsafe in the next hours
    # -------------------------------------------------------------------------
    "fragile_export": {
        "title": "4. Battery Export, Safe Now Only",
        "description": (
            "40 kW battery export at a calm noon passes the snapshot check, but as demand "
            "moves over the next hours the hybrid bus rises above 1.05 pu."
        ),
        "expected": {"plain": "DECISION_APPROVE", "certified": "DECISION_REJECT"},
        "payload": _payload(
            "act-004-fragile-export", "SYSTEM1", "S1 battery export to relieve the diesel",
            "calm", dispatch=[{"node_id": "ELU_HYB_BESS", "p_kw": 40.0, "q_kvar": 20.0}]),
    },
    # -------------------------------------------------------------------------
    # REJECTED under both gates
    # -------------------------------------------------------------------------
    "adequacy_breach": {
        "title": "5. Peak With Standby Gensets Down",
        "description": (
            "Record peak with both Nainativu standby gensets out: voltages look fine on the "
            "stiff-source model, but 380 kVA cannot carry 460 kVA."
        ),
        "expected": {"plain": "DECISION_REJECT", "certified": "DECISION_REJECT"},
        "payload": _payload(
            "act-005-adequacy-breach", "SYSTEM1", "No action during a genset outage",
            "nainativu_peak", outages=["NAI_DG2", "NAI_DG3"]),
    },
    "unsafe_undervolt": {
        "title": "6. Battery Fast-Charge Voltage Collapse",
        "description": (
            "Charging Eluvaitivu's battery at full rating pulls 72 kW down 0.7 km of 400 V "
            "cable, dragging the hybrid bus below 0.95 pu."
        ),
        "expected": {"plain": "DECISION_REJECT", "certified": "DECISION_REJECT"},
        "payload": _payload(
            "act-006-unsafe-undervolt", "SYSTEM2", "Aggressive battery recharge before the storm",
            "calm", dispatch=[{"node_id": "ELU_HYB_BESS", "p_kw": -72.0, "q_kvar": 0.0}]),
    },
    "unsafe_overvolt": {
        "title": "7. Battery Export Over-Voltage",
        "description": "Exporting the battery at full rating lifts the hybrid bus above 1.05 pu.",
        "expected": {"plain": "DECISION_REJECT", "certified": "DECISION_REJECT"},
        "payload": _payload(
            "act-007-unsafe-overvolt", "SYSTEM2", "Uncompensated battery export", "calm",
            dispatch=[{"node_id": "ELU_HYB_BESS", "p_kw": 72.0, "q_kvar": 0.0}]),
    },
    "unsafe_overload": {
        "title": "8. Hybrid Link Thermal Overload",
        "description": "Running every hybrid unit at once exceeds the 125 A hybrid link rating.",
        "expected": {"plain": "DECISION_REJECT", "certified": "DECISION_REJECT"},
        "payload": _payload(
            "act-008-unsafe-overload", "SYSTEM2", "Hybrid plant flat out", "calm",
            dispatch=[{"node_id": "ELU_HYB", "p_kw": 140.0, "q_kvar": 0.0}]),
    },
}

_BUS_ISLAND: dict[str, str] = {b["node_id"].upper(): b["island"] for b in load_pin()["buses"]}


def _classify_bus_island(bus_name: str) -> str:
    """The island a bus belongs to, from the northern-islands-v1 topology pin."""
    return _BUS_ISLAND.get(bus_name.upper(), "Unknown")


# ------------------------------------------------------------------------------- shared state

@lru_cache(maxsize=1)
def _history() -> ReplayHistory | None:
    try:
        return ReplayHistory.load(Calibration.load())
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=8)
def _verifier(profile: str) -> RiskCertifiedVerifier:
    paths = None
    try:
        paths = default_paths()
    except FileNotFoundError:
        pass
    return RiskCertifiedVerifier(profile=profile, history=_history(), input_paths=paths)


@lru_cache(maxsize=1)
def _topology() -> TopologyEntities | None:
    return TopologyEntities.from_pin(load_pin())


@lru_cache(maxsize=256)
def _snapshot_json(hour: str) -> tuple[str, str]:
    """(snapshot JSON, source) for a recorded hour; fixtures first, else Module 1."""
    for key, meta in HOURS.items():
        if meta["timestamp"] == hour:
            path = FIXTURES / f"module1_snapshot_v1_{key}.json"
            return path.read_text(), f"fixture:{key}"
    snap = module1.grid_state_snapshot(module1.Module1SnapshotRequest(timestamp=hour))
    return json.dumps(snap), "module1"


def _snapshot(hour: str, outages: list[str]) -> tuple[dict, str]:
    text, source = _snapshot_json(hour)
    snap = json.loads(text)
    if outages:
        snap["assets_out_of_service"] = sorted(set(snap.get("assets_out_of_service", []))
                                               | set(outages))
    return snap, source


# ------------------------------------------------------------------------------- endpoints

@router.get("/presets")
async def get_presets() -> dict[str, Any]:
    """Curated control actions, each on a recorded hour, with the verdict each gate gives."""
    return PRESETS


@router.get("/hours")
async def get_hours() -> dict[str, Any]:
    """The replay hours with committed fixture snapshots (any 2024-2025 hour also works)."""
    return HOURS


@router.get("/profiles")
async def get_profiles() -> dict[str, Any]:
    """The versioned limit profiles a verdict can be judged against."""
    return available_profiles()


@router.post("/verify", response_model=VerifyResponse)
async def verify_action(req: VerifyRequest) -> VerifyResponse:
    """Judges a proposed action on a recorded hour and explains the verdict."""
    try:
        if req.profile.upper() not in available_profiles():
            raise HTTPException(status_code=422, detail=f"unknown profile {req.profile!r}")
        snapshot, source = _snapshot(req.hour, req.outages)
        action = ProposedControlAction(
            action_id=req.action_id,
            origin=req.origin,
            breakers=req.breakers,
            load_shed=req.load_shed,
            dispatch=req.dispatch,
            rationale=req.rationale,
        )
        with _LOCK:
            return _verify_locked(req, action, snapshot, source)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc


def _verify_locked(req: VerifyRequest, action: ProposedControlAction, snapshot: dict,
                   source: str) -> VerifyResponse:
    rcv = _verifier(req.profile.upper())
    check_a = rcv.gate.verify(action, snapshot)
    if req.mode == "plain":
        verdict = check_a
    else:
        rcv.config = CertifierConfig(pid=req.pid)
        verdict = rcv.verify(action, snapshot=snapshot)

    # 1. Abductive attribution on the network's own entities
    action_trans = TransProposedAction.model_validate(action.model_dump())
    if verdict.violations:
        verdict.violations = AbductiveAttributor.attribute_violations(
            action_trans, verdict.violations, _topology())
    # 2. Severity feedback trace (M2)
    trace = rcv.gate.build_rejection_trace(verdict.action_id, verdict.violations)
    # 3. Grounded causal log, checked against the pin
    trans_verdict = TransVerdict.model_validate(verdict.model_dump())
    causal_log = TemplateCausalLogger.generate_log(trans_verdict, include_latency=True)
    grounding = GroundingValidator(_topology()).check(causal_log, trans_verdict)
    # 4. The solved snapshot with the action applied, for the grid panels
    buses, lines, adequacy = _solved_state(rcv, action, snapshot)

    limits = rcv.limits.config
    return VerifyResponse(
        action_id=verdict.action_id,
        decision=verdict.decision,
        check_a_decision=check_a.decision,
        mode=req.mode,
        hour=req.hour,
        profile=verdict.limit_profile,
        v_min_pu=limits.v_min_pu,
        v_max_pu=limits.v_max_pu,
        snapshot_source=source,
        solve_latency_ms=round(verdict.solve_latency_ms, 3),
        violations=[ViolationDTO(**v.model_dump()) for v in verdict.violations],
        unverified_reasons=verdict.unverified_reasons,
        rejection_severity=round(trace.severity, 4),
        certificate=verdict.certificate.model_dump(mode="json") if verdict.certificate else None,
        attribution=[a.model_dump(mode="json") for a in verdict.attribution],
        causal_log=CausalLogDTO(
            action_id=causal_log.action_id,
            text=causal_log.text,
            grounded_entities=causal_log.grounded_entities,
            generator=causal_log.generator.value,
            grounded=grounding.grounded,
            ungrounded=grounding.ungrounded,
        ),
        buses=buses,
        lines=lines,
        adequacy=adequacy,
    )


def _solved_state(rcv: RiskCertifiedVerifier, action: ProposedControlAction, snapshot: dict,
                  ) -> tuple[list[GridBusState], list[GridLineState], list[IslandAdequacy]]:
    circuit = rcv.circuit
    limits = rcv.limits.config
    try:
        circuit.reset_to_base()
        circuit.apply_snapshot(snapshot)
        rcv.gate.applicator.apply_action(action)
        PowerFlowSolver.solve_snapshot()
        bus_voltages = circuit.get_bus_voltages_pu()
        line_loadings = circuit.get_line_loadings()
        adequacy = PhysicsLimitsChecker.island_adequacy(circuit)
    finally:
        circuit.reset_to_base()

    buses: list[GridBusState] = []
    for bus, v_pu in sorted(bus_voltages.items()):
        if v_pu < limits.v_min_pu:
            status = "UNDERVOLTAGE"
        elif v_pu > limits.v_max_pu:
            status = "OVERVOLTAGE"
        else:
            status = "SAFE"
        buses.append(GridBusState(bus_name=bus, island=_classify_bus_island(bus),
                                  voltage_pu=round(v_pu, 4), status=status))

    lines: list[GridLineState] = []
    for line, data in sorted(line_loadings.items()):
        is_closed = data["enabled"] > 0.5
        margin = data["margin_fraction"]
        st = "TRIPPED" if not is_closed else "OVERLOAD" if margin > 0.0 else "NORMAL"
        lines.append(GridLineState(
            line_name=line, current_amps=round(data["max_amps"], 2),
            norm_amps=round(data["norm_amps"], 2), margin_fraction=round(margin, 4),
            is_closed=is_closed, status=st))

    islands = [
        IslandAdequacy(
            island=island, station=str(a["station"]),
            demand_kva=round(float(a["demand_kva"]), 1),
            capacity_kva=round(float(a["capacity_kva"]), 1),
            status="INADEQUATE" if a["demand_kva"] > a["capacity_kva"] else "ADEQUATE")
        for island, a in adequacy.items()
    ]
    return buses, lines, islands
