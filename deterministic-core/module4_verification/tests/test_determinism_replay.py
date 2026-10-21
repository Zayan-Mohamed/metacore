"""Same input, same certificate, byte for byte (C11).

The certificate holds everything a replay needs (calibration hash, seed, band, PID state) and
nothing time-dependent, so an auditor re-running a logged verdict gets the identical bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

from verification.risk.calibration import Calibration
from verification.risk.certifier import RiskCertifiedVerifier
from verification.risk.pid import ReplayHistory
from verification.types import DispatchSetpoint, LoadShedCommand, ProposedControlAction

SAMPLES = (Path(__file__).resolve().parents[3] / "packages" / "contracts" / "python"
           / "metacore_contracts" / "schema" / "samples")

ACTIONS = [
    ProposedControlAction(
        action_id="shed", load_shed=[LoadShedCommand(node_id="DEL_F2", shed_fraction=0.2)]),
    ProposedControlAction(
        action_id="bess", dispatch=[DispatchSetpoint(node_id="ELU_HYB_BESS", p_kw=40.0,
                                                     q_kvar=20.0)]),
]


def test_two_fresh_verifiers_give_byte_identical_certificates() -> None:
    cal = Calibration.load()
    hist = ReplayHistory.load(cal)
    for name in ("calm", "nainativu_peak", "ditwah_evening"):
        snap = json.loads((SAMPLES / f"module1_snapshot_v1_{name}.json").read_text())
        for action in ACTIONS:
            a = RiskCertifiedVerifier(history=hist).verify(action, snapshot=snap)
            b = RiskCertifiedVerifier(history=ReplayHistory.load(Calibration.load())).verify(
                action, snapshot=json.loads(json.dumps(snap)))
            assert a.decision == b.decision
            if a.certificate is not None:
                assert a.certificate.model_dump_json() == b.certificate.model_dump_json()
            assert [v.model_dump() for v in a.violations] == [v.model_dump()
                                                              for v in b.violations]


def test_the_same_verifier_twice_is_identical_too() -> None:
    v = RiskCertifiedVerifier(history=ReplayHistory.load(Calibration.load()))
    snap = json.loads((SAMPLES / "module1_snapshot_v1_ditwah_evening.json").read_text())
    first = v.verify(ACTIONS[0], snapshot=snap).certificate.model_dump_json()
    second = v.verify(ACTIONS[0], snapshot=snap).certificate.model_dump_json()
    assert first == second
