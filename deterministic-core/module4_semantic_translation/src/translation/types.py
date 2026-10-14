"""Data types for Module 4 Semantic Translation.

ZERO ML DEPENDENCIES. Mirrors verification and action types without requiring OpenDSS.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class Decision(StrEnum):
    DECISION_UNSPECIFIED = "DECISION_UNSPECIFIED"
    DECISION_APPROVE = "DECISION_APPROVE"
    DECISION_REJECT = "DECISION_REJECT"


class ViolationType(StrEnum):
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


class VerificationVerdict(BaseModel):
    action_id: str
    decision: Decision
    violations: list[Violation] = Field(default_factory=list)
    solve_latency_ms: float = 0.0


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
