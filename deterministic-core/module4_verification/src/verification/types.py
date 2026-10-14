"""Data types and schemas for Module 4 Physics Verification.

ZERO ML DEPENDENCIES. Pydantic dataclasses mirroring verification.proto and module3.proto.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class Decision(StrEnum):
    """Firewall verdict decision."""

    DECISION_UNSPECIFIED = "DECISION_UNSPECIFIED"
    DECISION_APPROVE = "DECISION_APPROVE"
    DECISION_REJECT = "DECISION_REJECT"


class ViolationType(StrEnum):
    """Specific physical violation category."""

    VIOLATION_TYPE_UNSPECIFIED = "VIOLATION_TYPE_UNSPECIFIED"
    VIOLATION_TYPE_UNDERVOLTAGE = "VIOLATION_TYPE_UNDERVOLTAGE"
    VIOLATION_TYPE_OVERVOLTAGE = "VIOLATION_TYPE_OVERVOLTAGE"
    VIOLATION_TYPE_THERMAL_OVERLOAD = "VIOLATION_TYPE_THERMAL_OVERLOAD"
    VIOLATION_TYPE_NON_CONVERGENCE = "VIOLATION_TYPE_NON_CONVERGENCE"
    VIOLATION_TYPE_MALFORMED_ACTION = "VIOLATION_TYPE_MALFORMED_ACTION"


class Violation(BaseModel):
    """Structured physical boundary violation record."""

    type: ViolationType
    element_id: str
    limit: float = Field(description="Statutory or thermal boundary limit")
    measured: float = Field(description="Simulated physical value (pu voltage or Amperes)")
    margin_fraction: float = Field(
        description="Signed margin past limit (e.g. +0.15 = 15% exceedance)"
    )
    attributed_component: str = Field(
        default="", description="Plausible causing component from ProposedControlAction"
    )


class BreakerCommand(BaseModel):
    """Command to open or close a line/switch breaker."""

    edge_id: str
    closed: bool


class LoadShedCommand(BaseModel):
    """Command to shed load on a given bus node."""

    node_id: str
    shed_fraction: float = Field(ge=0.0, le=1.0, description="Fraction of nominal load to shed")
    priority_tier: int = Field(default=3, description="1=Critical/Hospital, 2=Commercial, 3=Res")


class DispatchSetpoint(BaseModel):
    """Active and reactive power setpoint for generation/BESS assets."""

    node_id: str
    p_kw: float
    q_kvar: float = 0.0


class ProcessPath(StrEnum):
    """Origin path of the proposed action."""

    PROCESS_PATH_UNSPECIFIED = "PROCESS_PATH_UNSPECIFIED"
    SYSTEM1 = "SYSTEM1"
    SYSTEM2 = "SYSTEM2"


class ProposedControlAction(BaseModel):
    """Incoming action payload from Module 3 (Saabir)."""

    action_id: str
    origin: str = "SYSTEM1"
    breakers: list[BreakerCommand] = Field(default_factory=list)
    load_shed: list[LoadShedCommand] = Field(default_factory=list)
    dispatch: list[DispatchSetpoint] = Field(default_factory=list)
    rationale: str = ""


class VerificationVerdict(BaseModel):
    """Synchronous physical verification decision emitted to the orchestrator."""

    action_id: str
    decision: Decision
    violations: list[Violation] = Field(default_factory=list)
    solve_latency_ms: float = Field(ge=0.0, description="AC power flow solve latency in ms")
    spec_version: str = "safety-spec-v1.0"


class RejectionTrace(BaseModel):
    """Feedback payload for Module 2 uncertainty calibration."""

    action_id: str
    violations: list[Violation] = Field(default_factory=list)
    severity: float = Field(
        ge=0.0, le=1.0, description="Normalized 0..1 severity of worst violation"
    )
