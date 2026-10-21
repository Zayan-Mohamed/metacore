"""Module 4 Verification Package — OpenDSS Physics Firewall.

ZERO ML DEPENDENCIES.
"""

from .firewall.limits import PhysicsLimitsChecker, SafetyLimitsConfig
from .firewall.verifier import PhysicsVerifier
from .opendss.circuit import CircuitTwin
from .powerflow.action_applicator import ActionApplicator
from .powerflow.solver import PowerFlowSolver
from .risk.certifier import CertifierConfig, RiskCertifiedVerifier
from .types import (
    BreakerCommand,
    Decision,
    DispatchSetpoint,
    LoadShedCommand,
    ProcessPath,
    ProposedControlAction,
    RejectionTrace,
    RiskAttribution,
    RiskCertificate,
    VerificationVerdict,
    Violation,
    ViolationType,
)

__all__ = [
    "CircuitTwin",
    "ActionApplicator",
    "PowerFlowSolver",
    "PhysicsLimitsChecker",
    "SafetyLimitsConfig",
    "PhysicsVerifier",
    "RiskCertifiedVerifier",
    "CertifierConfig",
    "RiskCertificate",
    "RiskAttribution",
    "Decision",
    "ViolationType",
    "Violation",
    "BreakerCommand",
    "LoadShedCommand",
    "DispatchSetpoint",
    "ProcessPath",
    "ProposedControlAction",
    "VerificationVerdict",
    "RejectionTrace",
]
