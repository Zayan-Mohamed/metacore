"""Module 4 on one live run: can the grid carry each forecast hour, with no control action?

Run as its own process so OpenDSS stays inside deterministic-core's code (ADR 0003): the
orchestrator pipes Module 1's snapshots in on stdin. Each hour is judged by Module 4's own
``RiskCertifiedVerifier`` with an empty ProposedControlAction ("do nothing"), and the solved
state is read back the same way the gateway's verify endpoint reads it. Nothing of Module 4's
logic is reimplemented; its hazard level comes from its own calibration.

Live hours are outside the replay history, so Module 4's online band adaptation (PID) does not
run and the static calibrated band is used; Module 4 records that in the certificate notes.

    python app/m4_runner.py < snapshots.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

CORE = Path(__file__).resolve().parents[3] / "deterministic-core"


def main() -> int:
    for sub in ("module4_verification", "module4_semantic_translation"):
        sys.path.insert(0, str(CORE / sub / "src"))
    from verification.firewall.limits import PhysicsLimitsChecker
    from verification.powerflow.solver import PowerFlowSolver
    from verification.risk.calibration import Calibration
    from verification.risk.certifier import RiskCertifiedVerifier
    from verification.types import ProposedControlAction

    payload = json.load(sys.stdin)
    t0 = time.perf_counter()
    rcv = RiskCertifiedVerifier()
    cal = Calibration.load()
    limits = rcv.limits.config
    hours = []
    for lead, (snap, met) in enumerate(zip(payload["snapshots"], payload["meteorology"],
                                           strict=True)):
        hazard = cal.hazard_level_at([met["wind_50m_ms"][i] for i in cal.islands],
                                     [met["precip_mm_hr"][i] for i in cal.islands])
        action = ProposedControlAction(action_id=f"live-noop-{snap['timestamp_lst']}",
                                       rationale="no control action: can the grid carry this hour?")
        t1 = time.perf_counter()
        verdict = rcv.verify(action, snapshot=snap, hazard_level=hazard)
        verify_ms = (time.perf_counter() - t1) * 1000

        circuit = rcv.circuit
        try:
            circuit.reset_to_base()
            circuit.apply_snapshot(snap)
            PowerFlowSolver.solve_snapshot()
            buses = circuit.get_bus_voltages_pu()
            lines = circuit.get_line_loadings()
            adequacy = PhysicsLimitsChecker.island_adequacy(circuit)
        finally:
            circuit.reset_to_base()
        hours.append({
            "hour": snap["timestamp_lst"], "lead_h": lead, "hazard_level": hazard,
            "decision": verdict.decision.name if hasattr(verdict.decision, "name")
            else str(verdict.decision),
            "violations": [v.model_dump(mode="json") for v in verdict.violations],
            "unverified_reasons": verdict.unverified_reasons,
            "certificate": (verdict.certificate.model_dump(mode="json")
                            if verdict.certificate else None),
            "verify_ms": round(verify_ms, 1),
            "buses": [{"bus": b, "voltage_pu": round(v, 4),
                       "status": "UNDERVOLTAGE" if v < limits.v_min_pu else
                       "OVERVOLTAGE" if v > limits.v_max_pu else "SAFE"}
                      for b, v in sorted(buses.items())],
            "lines": [{"line": ln, "amps": round(d["max_amps"], 1),
                       "rating_amps": round(d["norm_amps"], 1),
                       "loading_pct": round(100 * d["max_amps"] / d["norm_amps"], 1)
                       if d["norm_amps"] else None,
                       "in_service": d["enabled"] > 0.5}
                      for ln, d in sorted(lines.items())],
            "adequacy": [{"island": isl, "station": str(a["station"]),
                          "demand_kva": round(float(a["demand_kva"]), 1),
                          "capacity_kva": round(float(a["capacity_kva"]), 1)}
                         for isl, a in adequacy.items()],
            "assets_out_of_service": snap.get("assets_out_of_service", []),
        })
    print(json.dumps({"profile": rcv.limits.config.name if hasattr(rcv.limits.config, "name")
                      else None, "v_min_pu": limits.v_min_pu, "v_max_pu": limits.v_max_pu,
                      "hours": hours, "total_ms": round((time.perf_counter() - t0) * 1000, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
