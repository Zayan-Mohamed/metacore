"""Module 4 — Physical Verification & Causal Translation Gateway Router.

Provides real-time REST API endpoints for OpenDSS AC power-flow verification,
physics limits evaluation (voltage and thermal ampacity), abductive attribution,
and grounded operator explanation logs.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from translation.abductive.attribution import AbductiveAttributor
from translation.templates.causal_logger import TemplateCausalLogger
from translation.types import (
    BreakerCommand as TransBreakerCommand,
)
from translation.types import (
    DispatchSetpoint as TransDispatchSetpoint,
)
from translation.types import (
    LoadShedCommand as TransLoadShedCommand,
)
from translation.types import (
    ProposedControlAction as TransProposedAction,
)
from verification.firewall.verifier import PhysicsVerifier
from verification.powerflow.solver import PowerFlowSolver
from verification.types import (
    BreakerCommand,
    Decision,
    DispatchSetpoint,
    LoadShedCommand,
    ProposedControlAction,
    ViolationType,
)

router = APIRouter(prefix="/module4", tags=["module4"])

_verifier: PhysicsVerifier | None = None


def _get_verifier() -> PhysicsVerifier:
    global _verifier
    if _verifier is None:
        _verifier = PhysicsVerifier()
    return _verifier


class VerifyRequest(BaseModel):
    action_id: str = Field(default="act-web-001", description="Action unique identifier")
    origin: str = Field(default="SYSTEM1", description="SYSTEM1 or SYSTEM2")
    rationale: str = Field(default="", description="Operator or AI rationale")
    breakers: list[BreakerCommand] = Field(default_factory=list)
    load_shed: list[LoadShedCommand] = Field(default_factory=list)
    dispatch: list[DispatchSetpoint] = Field(default_factory=list)


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


class VerifyResponse(BaseModel):
    action_id: str
    decision: Decision
    solve_latency_ms: float
    violations: list[ViolationDTO]
    rejection_severity: float
    causal_log: CausalLogDTO
    buses: list[GridBusState]
    lines: list[GridLineState]


PRESETS: dict[str, dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # 3 APPROVED (SAFE) ACTIONS
    # -------------------------------------------------------------------------
    "nominal_safe": {
        "title": "1. Nominal S1 Reactive Shed (Approved)",
        "description": "Routine minor load shed on Island 3 to stabilize small voltage drift.",
        "payload": {
            "action_id": "act-001-nominal-safe",
            "origin": "SYSTEM1",
            "rationale": "S1 reactive shed vuln=0.14",
            "breakers": [],
            "load_shed": [{"node_id": "N8", "shed_fraction": 0.1117, "priority_tier": 3}],
            "dispatch": [],
        },
    },
    "cyclone_survival": {
        "title": "2. Cyclone Ditwah Survival Dispatch (Approved)",
        "description": (
            "System 2 closes critical tie-line, dispatches generator, sheds non-essential load."
        ),
        "payload": {
            "action_id": "act-002-cyclone-survival",
            "origin": "SYSTEM2",
            "rationale": "S2 survival opt vuln=0.63 protect-tier1",
            "breakers": [{"edge_id": "E_crit_1", "closed": True}],
            "load_shed": [
                {"node_id": "N11", "shed_fraction": 0.25, "priority_tier": 3},
                {"node_id": "N8", "shed_fraction": 0.0983, "priority_tier": 3},
            ],
            "dispatch": [{"node_id": "N4", "p_kw": 147.37, "q_kvar": 10.0}],
        },
    },
    "peak_power_sharing": {
        "title": "3. Peak Emergency Power-Sharing (Approved)",
        "description": (
            "System 2 coordinates multi-island power-sharing during peak cyclone storm intensity."
        ),
        "payload": {
            "action_id": "act-003-peak-sharing",
            "origin": "SYSTEM2",
            "rationale": "S2 survival opt vuln=0.88 protect-tier1",
            "breakers": [{"edge_id": "E_crit_1", "closed": True}],
            "load_shed": [
                {"node_id": "N8", "shed_fraction": 0.25, "priority_tier": 3},
                {"node_id": "N12", "shed_fraction": 0.2325, "priority_tier": 3},
            ],
            "dispatch": [{"node_id": "N4", "p_kw": 184.21, "q_kvar": 10.0}],
        },
    },
    # -------------------------------------------------------------------------
    # 3 REJECTED (UNSAFE) ACTIONS
    # -------------------------------------------------------------------------
    "unsafe_undervolt": {
        "title": "4. Tie-Line Trip Island Collapse (Rejected)",
        "description": (
            "Trips inter-island connection while local generation is zeroed, "
            "triggering undervoltage."
        ),
        "payload": {
            "action_id": "act-004-unsafe-undervolt",
            "origin": "SYSTEM2",
            "rationale": "Aggressive island isolation without backup generation",
            "breakers": [
                {"edge_id": "Line_2_3", "closed": False},
                {"edge_id": "E_crit_1", "closed": False},
            ],
            "load_shed": [],
            "dispatch": [
                {"node_id": "N8", "p_kw": 0.0, "q_kvar": 0.0},
                {"node_id": "N9", "p_kw": 0.0, "q_kvar": 0.0},
            ],
        },
    },
    "unsafe_overvolt": {
        "title": "5. Reactive Over-Injection (Rejected)",
        "description": (
            "Excessive reactive power injection drives bus voltages beyond 1.05 pu limit."
        ),
        "payload": {
            "action_id": "act-005-unsafe-overvolt",
            "origin": "SYSTEM2",
            "rationale": "Uncompensated voltage support attempt",
            "breakers": [],
            "load_shed": [],
            "dispatch": [{"node_id": "N1", "p_kw": 500.0, "q_kvar": 6000.0}],
        },
    },
    "unsafe_overload": {
        "title": "6. Cable Thermal Overload (Rejected)",
        "description": "High active power dispatch exceeds subsea cable thermal ampacity limit.",
        "payload": {
            "action_id": "act-006-unsafe-overload",
            "origin": "SYSTEM2",
            "rationale": "Excessive export beyond subsea cable rating",
            "breakers": [],
            "load_shed": [],
            "dispatch": [{"node_id": "N8", "p_kw": 6000.0, "q_kvar": 500.0}],
        },
    },
}


def _classify_bus_island(bus_name: str) -> str:
    """Classifies bus according to delft_3island.dss topological node clusters."""
    name = bus_name.upper()
    if name in ("SOURCEBUS", "N1", "N2", "N3"):
        return "Nainativu Island (Grid 1)"
    if name in ("N4", "N5", "N6"):
        return "Analaitivu Island (Grid 2)"
    if name in ("N8", "N9", "N11", "N12"):
        return "Delft Island (Grid 3)"
    return "Inter-Island Tie"


@router.get("/presets")
async def get_presets() -> dict[str, Any]:
    """Returns the 6 curated representative control actions (3 safe, 3 unsafe)."""
    return PRESETS


@router.post("/verify", response_model=VerifyResponse)
async def verify_action(req: VerifyRequest) -> VerifyResponse:
    """Evaluates a proposed action against OpenDSS physics and emits grounded causal log."""
    try:
        verifier = _get_verifier()
        action_verif = ProposedControlAction(
            action_id=req.action_id,
            origin=req.origin,
            breakers=req.breakers,
            load_shed=req.load_shed,
            dispatch=req.dispatch,
            rationale=req.rationale,
        )

        action_trans = TransProposedAction(
            action_id=req.action_id,
            origin=req.origin,
            breakers=[TransBreakerCommand(**b.model_dump()) for b in req.breakers],
            load_shed=[TransLoadShedCommand(**ls.model_dump()) for ls in req.load_shed],
            dispatch=[TransDispatchSetpoint(**d.model_dump()) for d in req.dispatch],
            rationale=req.rationale,
        )

        # 1. Physical Simulation & Limit Evaluation
        verifier.circuit.reset_to_base()
        malformed = verifier.applicator.apply_action(action_verif)

        if malformed:
            violations = malformed
            decision = Decision.DECISION_REJECT
            solve_latency_ms = 0.0
            bus_voltages = verifier.circuit.get_bus_voltages_pu()
            line_loadings = verifier.circuit.get_line_loadings()
        else:
            converged, solve_latency_ms = PowerFlowSolver.solve_snapshot()
            violations = verifier.limits_checker.check_limits(verifier.circuit, converged)
            decision = (
                Decision.DECISION_APPROVE
                if len(violations) == 0
                else Decision.DECISION_REJECT
            )
            # Capture snapshot electrical state WHILE solver solution is live in memory
            bus_voltages = verifier.circuit.get_bus_voltages_pu()
            line_loadings = verifier.circuit.get_line_loadings()

        # Reset circuit to clean base state after capturing solved state
        verifier.circuit.reset_to_base()

        # 2. Abductive Attribution
        if violations:
            violations = AbductiveAttributor.attribute_violations(action_trans, violations)

        verdict = verifier._build_verdict(
            action_id=action_verif.action_id,
            decision=decision,
            violations=violations,
            latency_ms=solve_latency_ms,
        )

        # 3. Severity feedback trace
        trace = verifier.build_rejection_trace(verdict.action_id, verdict.violations)

        # 4. Grounded Causal Log
        causal_log = TemplateCausalLogger.generate_log(verdict, include_latency=True)

        # 5. Build DTOs for snapshot grid state
        buses_dto: list[GridBusState] = []
        for bus, v_pu in sorted(bus_voltages.items()):
            if v_pu < 0.95:
                status = "UNDERVOLTAGE"
            elif v_pu > 1.05:
                status = "OVERVOLTAGE"
            else:
                status = "SAFE"
            buses_dto.append(
                GridBusState(
                    bus_name=bus,
                    island=_classify_bus_island(bus),
                    voltage_pu=round(v_pu, 4),
                    status=status,
                )
            )

        lines_dto: list[GridLineState] = []
        for line, data in sorted(line_loadings.items()):
            max_amps = data["max_amps"]
            norm_amps = data["norm_amps"]
            margin = data["margin_fraction"]
            is_closed = data["enabled"] > 0.5
            if not is_closed:
                st = "TRIPPED"
            elif margin > 0.0:
                st = "OVERLOAD"
            else:
                st = "NORMAL"
            lines_dto.append(
                GridLineState(
                    line_name=line,
                    current_amps=round(max_amps, 2),
                    norm_amps=round(norm_amps, 2),
                    margin_fraction=round(margin, 4),
                    is_closed=is_closed,
                    status=st,
                )
            )

        return VerifyResponse(
            action_id=verdict.action_id,
            decision=verdict.decision,
            solve_latency_ms=round(verdict.solve_latency_ms, 3),
            violations=[ViolationDTO(**v.model_dump()) for v in verdict.violations],
            rejection_severity=round(trace.severity, 4),
            causal_log=CausalLogDTO(
                action_id=causal_log.action_id,
                text=causal_log.text,
                grounded_entities=causal_log.grounded_entities,
                generator=causal_log.generator.value,
            ),
            buses=buses_dto,
            lines=lines_dto,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc
