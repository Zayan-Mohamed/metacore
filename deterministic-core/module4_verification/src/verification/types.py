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
    # The checker could not establish safety: solver error, non-convergence, a malformed
    # snapshot, or (with the risk certificate) evidence that straddles the risk target.
    # Never an approval -- M3 treats it as not approved.
    DECISION_UNVERIFIED = "DECISION_UNVERIFIED"


class ViolationType(StrEnum):
    """Specific physical violation category."""

    VIOLATION_TYPE_UNSPECIFIED = "VIOLATION_TYPE_UNSPECIFIED"
    VIOLATION_TYPE_UNDERVOLTAGE = "VIOLATION_TYPE_UNDERVOLTAGE"
    VIOLATION_TYPE_OVERVOLTAGE = "VIOLATION_TYPE_OVERVOLTAGE"
    VIOLATION_TYPE_THERMAL_OVERLOAD = "VIOLATION_TYPE_THERMAL_OVERLOAD"
    VIOLATION_TYPE_NON_CONVERGENCE = "VIOLATION_TYPE_NON_CONVERGENCE"
    VIOLATION_TYPE_MALFORMED_ACTION = "VIOLATION_TYPE_MALFORMED_ACTION"
    # Island demand above the in-service generating capacity. The model's lead genset is a stiff
    # source that never runs out, so without this check an overloaded station would pass.
    VIOLATION_TYPE_GENERATION_INADEQUATE = "VIOLATION_TYPE_GENERATION_INADEQUATE"


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


class RiskCertificate(BaseModel):
    """Check B's evidence: k of N likely next-hour conditions broke a limit, with exact bounds.

    Everything needed to replay the verdict is here (calibration hash, seed, band, PID state);
    nothing time-dependent is, so the same input gives a byte-identical certificate.
    """

    decision: Decision
    n_scenarios: int = Field(ge=0, description="Scenarios evaluated (N, or fewer if stopped)")
    n_max: int = Field(ge=0, description="Planned scenarios (688 for 1 % at 99.9 %)")
    n_failures: int = Field(ge=0, description="Scenarios that broke at least one limit (k)")
    lower_bound: float = Field(ge=0.0, le=1.0, description="Clopper-Pearson lower bound on p")
    upper_bound: float = Field(ge=0.0, le=1.0, description="Clopper-Pearson upper bound on p")
    epsilon: float = Field(description="Risk target")
    confidence: float
    stopped_early: bool = False
    horizon_h: int
    limit_profile: str
    calibration_version: str
    calibration_sha256: str
    seed: int
    timestamp: str = Field(description="The recorded hour the snapshot describes")
    situation: dict[str, str | int] = Field(default_factory=dict)
    band: dict[str, float | int | str] = Field(default_factory=dict)
    pid: dict[str, float | int | bool] | None = None
    corner_top_safe: bool | None = None
    corner_bottom_safe: bool | None = None
    failures_by_island: dict[str, int] = Field(default_factory=dict)
    failing_elements: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RiskAttribution(BaseModel):
    """One player's Shapley share of the certified risk (shares sum to k / N)."""

    player: str = Field(description="A feeder-group bus or an island name from the pin")
    island: str
    buses: list[str] = Field(default_factory=list)
    share: float = Field(description="Shapley value: points of failure rate this player owns")
    fraction_of_risk: float = Field(description="share / total failure rate")
    alone: float = Field(description="Failure rate when only this player moves")


class VerificationVerdict(BaseModel):
    """Synchronous physical verification decision emitted to the orchestrator."""

    action_id: str
    decision: Decision
    violations: list[Violation] = Field(default_factory=list)
    solve_latency_ms: float = Field(ge=0.0, description="AC power flow solve latency in ms")
    spec_version: str = "safety-spec-v1.0"
    limit_profile: str = Field(
        default="METACORE-STRICT-v1", description="Versioned limit profile the verdict used"
    )
    unverified_reasons: list[str] = Field(
        default_factory=list, description="Why the verdict is UNVERIFIED (empty otherwise)"
    )
    certificate: RiskCertificate | None = Field(
        default=None, description="Check B evidence; None for the plain snapshot gate"
    )
    attribution: list[RiskAttribution] = Field(
        default_factory=list, description="Shapley risk shares; empty when nothing failed"
    )


class RejectionTrace(BaseModel):
    """Feedback payload for Module 2 uncertainty calibration."""

    action_id: str
    violations: list[Violation] = Field(default_factory=list)
    severity: float = Field(
        ge=0.0, le=1.0, description="Normalized 0..1 severity of worst violation"
    )
