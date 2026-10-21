"""The five gates compared on the same cases (the plan's B0 to B4).

    B0  plain gate: Check A on the snapshot only (M4 before this work)
    B1  fixed buffer: Check A with the voltage band tightened to 0.97-1.03 pu
    B2  Monte Carlo without conformal situations: scenarios from every calibration row, one band
    B3  risk-certified with static situation bands (conformal PID off)
    B4  risk-certified, full: situation bands + conformal PID in storms + Clopper-Pearson
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from verification.firewall.limits import PhysicsLimitsChecker, SafetyLimitsConfig
from verification.firewall.verifier import PhysicsVerifier
from verification.opendss.circuit import CircuitTwin
from verification.risk.certifier import CertifierConfig, RiskCertifiedVerifier
from verification.risk.pid import ReplayHistory
from verification.types import ProposedControlAction, VerificationVerdict

BUFFER_PU = 0.02


@dataclass
class Baseline:
    key: str
    name: str
    verify: Callable[[ProposedControlAction, dict], VerificationVerdict]


def baselines(history: ReplayHistory, circuit: CircuitTwin | None = None,
              band_scale: float = 1.0, attribution: bool = True) -> list[Baseline]:
    circuit = circuit or CircuitTwin()
    plain = PhysicsVerifier(circuit)
    buffered = PhysicsVerifier(circuit)
    buffered.limits_checker = PhysicsLimitsChecker(SafetyLimitsConfig(
        v_min_pu=0.95 + BUFFER_PU, v_max_pu=1.05 - BUFFER_PU, profile="B1-BUFFER-0.97"))

    def rcv(**cfg) -> RiskCertifiedVerifier:
        return RiskCertifiedVerifier(circuit, history=history,
                                     config=CertifierConfig(band_scale=band_scale, **cfg))

    b2, b3 = rcv(situation_blind=True, attribution=False), rcv(pid="off", attribution=False)
    b4 = rcv(pid="auto", attribution=attribution)
    return [
        Baseline("B0", "Plain gate (snapshot only)", plain.verify),
        Baseline("B1", "Fixed buffer 0.97-1.03 pu", buffered.verify),
        Baseline("B2", "Monte Carlo, no situation bands", lambda a, s: b2.verify(a, s)),
        Baseline("B3", "RCV, static bands", lambda a, s: b3.verify(a, s)),
        Baseline("B4", "RCV, bands + conformal PID", lambda a, s: b4.verify(a, s)),
    ]
