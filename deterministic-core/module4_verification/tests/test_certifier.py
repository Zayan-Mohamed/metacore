"""RiskCertifiedVerifier: Check A, then Check B, then the certificate (C11)."""

from __future__ import annotations

import json
import statistics
import time
from functools import cache
from pathlib import Path

import pytest
from verification.firewall.verifier import PhysicsVerifier
from verification.risk import certifier as certifier_module
from verification.risk.calibration import DEFAULT_ARTIFACT, Calibration
from verification.risk.certifier import CertifierConfig, RiskCertifiedVerifier, ScenarioOutcome
from verification.risk.pid import ReplayHistory
from verification.types import (
    Decision,
    DispatchSetpoint,
    LoadShedCommand,
    ProposedControlAction,
    ViolationType,
)

SAMPLES = (Path(__file__).resolve().parents[3] / "packages" / "contracts" / "python"
           / "metacore_contracts" / "schema" / "samples")


def sample(name: str) -> dict:
    return json.loads((SAMPLES / f"module1_snapshot_v1_{name}.json").read_text())


@cache
def history() -> ReplayHistory:
    return ReplayHistory.load(Calibration.load())


def rcv(**config) -> RiskCertifiedVerifier:
    return RiskCertifiedVerifier(history=history(), config=CertifierConfig(**config))


SAFE = ProposedControlAction(
    action_id="shed-nai-f2",
    load_shed=[LoadShedCommand(node_id="NAI_F2", shed_fraction=0.3, priority_tier=3)])
# Battery export that the calm noon snapshot can absorb, but the next hours cannot
FRAGILE = ProposedControlAction(
    action_id="bess-export-40",
    dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=40.0, q_kvar=20.0)])


def test_a_known_safe_action_is_approved_with_a_full_certificate() -> None:
    verdict = rcv().verify(SAFE, snapshot=sample("calm"))
    assert verdict.decision == Decision.DECISION_APPROVE
    c = verdict.certificate
    assert c is not None and c.decision == Decision.DECISION_APPROVE
    assert (c.n_scenarios, c.n_max, c.n_failures) == (688, 688, 0)
    assert c.upper_bound <= c.epsilon == 0.01
    assert c.limit_profile == "METACORE-STRICT-v1"
    assert c.calibration_sha256 == Calibration.load().sha256
    assert c.seed == Calibration.load().seed
    assert c.timestamp == "2025-10-05T12:00"
    assert c.corner_top_safe and c.corner_bottom_safe


def test_the_plain_gate_approves_what_the_next_hours_break() -> None:
    """The motivating case: Check A passes on the snapshot, Check B rejects."""
    assert PhysicsVerifier().verify(FRAGILE, sample("calm")).decision == \
        Decision.DECISION_APPROVE
    verdict = rcv().verify(FRAGILE, snapshot=sample("calm"))
    assert verdict.decision == Decision.DECISION_REJECT
    c = verdict.certificate
    assert c.lower_bound > c.epsilon
    assert c.stopped_early and c.n_scenarios == 100
    assert c.failures_by_island["Eluvaitivu"] == c.n_failures
    assert {v.type for v in verdict.violations} == {ViolationType.VIOLATION_TYPE_OVERVOLTAGE}
    assert "scenarios" in verdict.violations[0].attributed_component


def test_a_check_a_rejection_is_final() -> None:
    bad = ProposedControlAction(
        action_id="bess-charge",
        dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=-72.0)])
    verdict = rcv().verify(bad, snapshot=sample("calm"))
    assert verdict.decision == Decision.DECISION_REJECT
    assert verdict.certificate is None


def test_no_hour_to_certify_is_unverified() -> None:
    verdict = rcv().verify(SAFE)
    assert verdict.decision == Decision.DECISION_UNVERIFIED
    assert "no recorded hour" in verdict.unverified_reasons[0]


def test_a_stale_calibration_is_unverified(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_ARTIFACT.read_text())
    raw["seed"] = 1
    edited = tmp_path / "calibration_v1.json"
    edited.write_text(json.dumps(raw))
    verdict = RiskCertifiedVerifier(calibration=edited).verify(SAFE, snapshot=sample("calm"))
    assert verdict.decision == Decision.DECISION_UNVERIFIED
    assert "calibration unavailable" in verdict.unverified_reasons[0]


def test_a_straddling_count_is_unverified(monkeypatch: pytest.MonkeyPatch) -> None:
    """3 failures in 688 gives p in [0.03 %, 1.9 %]: neither side of 1 % is certain."""
    real = RiskCertifiedVerifier._scenario
    calls = {"n": 0}

    def three_failures(self, *args):
        outcome, ms = real(self, *args)
        calls["n"] += 1
        if calls["n"] in (50, 300, 600):
            return ScenarioOutcome(True, frozenset({"Delft-Neduntivu"})), ms
        return outcome, ms

    monkeypatch.setattr(certifier_module.RiskCertifiedVerifier, "_scenario", three_failures)
    verdict = rcv().verify(SAFE, snapshot=sample("calm"))
    assert verdict.decision == Decision.DECISION_UNVERIFIED
    assert verdict.certificate.n_failures == 3 and verdict.certificate.n_scenarios == 688
    assert "straddles" in verdict.unverified_reasons[0]


def test_storm_mode_turns_conformal_pid_on() -> None:
    c = rcv().verify(SAFE, snapshot=sample("ditwah_evening")).certificate
    assert c.situation["hazard"] in ("SEVERE", "EXTREME")
    assert c.pid["enabled"] and c.pid["storm"]
    assert c.band["key"].endswith("+pid")
    calm = rcv().verify(SAFE, snapshot=sample("calm")).certificate
    assert calm.pid == {"enabled": False, "storm": False}
    off = rcv(pid="off").verify(SAFE, snapshot=sample("ditwah_evening")).certificate
    assert off.pid is None and not off.band["key"].endswith("+pid")


def test_without_history_the_static_band_is_used_and_said_so() -> None:
    c = RiskCertifiedVerifier().verify(SAFE, snapshot=sample("calm")).certificate
    assert any("hazard tag missing" in n for n in c.notes)
    assert any("PID needs the replayed history" in n for n in c.notes)


def test_latency_of_688_scenarios() -> None:
    """E3: p50 / p95 wall-clock per verify call (printed; asserted only against a loose cap)."""
    v = rcv()
    snap = sample("nainativu_peak")
    v.verify(SAFE, snapshot=snap)  # warm caches
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        verdict = v.verify(SAFE, snapshot=snap)
        times.append((time.perf_counter() - t0) * 1000.0)
        assert verdict.certificate.n_scenarios == 688
    times.sort()
    p95 = times[int(0.95 * (len(times) - 1))]
    print(f"\n[M4 RCV] 688 scenarios: p50 {statistics.median(times):.0f} ms | p95 {p95:.0f} ms")
    assert p95 < 5000.0


def test_a_wider_band_spreads_the_scenarios_not_only_the_corners() -> None:
    """E5's lever: band_scale must change what the scenarios contain."""
    snap = sample("calm")
    action = ProposedControlAction(
        action_id="bess-export-25",
        dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=25.0, q_kvar=10.0)])
    base = rcv().verify(action, snapshot=snap).certificate
    wide = rcv(band_scale=2.0).verify(action, snapshot=snap).certificate
    assert wide.band["key"].endswith("x2")
    assert wide.n_failures > base.n_failures
