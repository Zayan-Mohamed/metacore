"""Data types for Module 4 Semantic Translation.

ZERO ML DEPENDENCIES. Mirrors verification and action types without requiring OpenDSS.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Decision(StrEnum):
    DECISION_UNSPECIFIED = "DECISION_UNSPECIFIED"
    DECISION_APPROVE = "DECISION_APPROVE"
    DECISION_REJECT = "DECISION_REJECT"
    # The checker could not establish safety: solver error, non-convergence, a malformed
    # snapshot, or (with the risk certificate) evidence that straddles the risk target.
    # Never an approval -- M3 treats it as not approved.
    DECISION_UNVERIFIED = "DECISION_UNVERIFIED"


class ViolationType(StrEnum):
    VIOLATION_TYPE_UNSPECIFIED = "VIOLATION_TYPE_UNSPECIFIED"
    VIOLATION_TYPE_UNDERVOLTAGE = "VIOLATION_TYPE_UNDERVOLTAGE"
    VIOLATION_TYPE_OVERVOLTAGE = "VIOLATION_TYPE_OVERVOLTAGE"
    VIOLATION_TYPE_THERMAL_OVERLOAD = "VIOLATION_TYPE_THERMAL_OVERLOAD"
    VIOLATION_TYPE_NON_CONVERGENCE = "VIOLATION_TYPE_NON_CONVERGENCE"
    VIOLATION_TYPE_MALFORMED_ACTION = "VIOLATION_TYPE_MALFORMED_ACTION"
    VIOLATION_TYPE_GENERATION_INADEQUATE = "VIOLATION_TYPE_GENERATION_INADEQUATE"


class Violation(BaseModel):
    """Structured physical boundary violation record."""

    type: ViolationType
    element_id: str
    limit: float
    measured: float
    margin_fraction: float
    attributed_component: str = ""


class BreakerCommand(BaseModel):
    edge_id: str
    closed: bool


class LoadShedCommand(BaseModel):
    node_id: str
    shed_fraction: float
    priority_tier: int = 3


class DispatchSetpoint(BaseModel):
    node_id: str
    p_kw: float
    q_kvar: float = 0.0


class ProposedControlAction(BaseModel):
    action_id: str
    origin: str = "SYSTEM1"
    breakers: list[BreakerCommand] = Field(default_factory=list)
    load_shed: list[LoadShedCommand] = Field(default_factory=list)
    dispatch: list[DispatchSetpoint] = Field(default_factory=list)
    rationale: str = ""


class RiskCertificate(BaseModel):
    """The fields of verification's RiskCertificate the explanation uses (the rest ignored)."""

    model_config = ConfigDict(extra="ignore")

    decision: Decision
    n_scenarios: int
    n_failures: int
    lower_bound: float
    upper_bound: float
    epsilon: float
    confidence: float
    horizon_h: int = 6
    stopped_early: bool = False
    limit_profile: str = ""


class RiskAttribution(BaseModel):
    model_config = ConfigDict(extra="ignore")

    player: str
    island: str = ""
    share: float
    fraction_of_risk: float = 0.0


class VerificationVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action_id: str
    decision: Decision
    violations: list[Violation] = Field(default_factory=list)
    solve_latency_ms: float = 0.0
    unverified_reasons: list[str] = Field(default_factory=list)
    certificate: RiskCertificate | None = None
    attribution: list[RiskAttribution] = Field(default_factory=list)


class RejectionTrace(BaseModel):
    action_id: str
    violations: list[Violation] = Field(default_factory=list)
    severity: float = 0.0


class Generator(StrEnum):
    """Generator type used for causal log synthesis."""

    GENERATOR_UNSPECIFIED = "GENERATOR_UNSPECIFIED"
    GENERATOR_TEMPLATE = "GENERATOR_TEMPLATE"
    GENERATOR_CONSTRAINED = "GENERATOR_CONSTRAINED"


class CausalLog(BaseModel):
    """Operator-facing causal explanation, grounded in physical violation evidence."""

    action_id: str
    text: str
    grounded_entities: list[str] = Field(
        default_factory=list,
        description="Elements cited in the text (must be a subset of violation element_ids)",
    )
    generator: Generator = Generator.GENERATOR_TEMPLATE
