"""Module 4 Standalone End-to-End Demo & Verification Runner.

Simulates sample actions from Module 3 against the OpenDSS physical twin,
evaluates limits, computes verdicts, attribution, and grounded causal logs.
Writes sample_m4_output.jsonl for dashboard and downstream integration.
"""

import json
from pathlib import Path

from translation.abductive.attribution import AbductiveAttributor
from translation.templates.causal_logger import TemplateCausalLogger
from translation.types import (
    BreakerCommand,
    DispatchSetpoint,
    LoadShedCommand,
    ProposedControlAction,
)
from verification.firewall.verifier import PhysicsVerifier

CORE_DIR = Path(__file__).resolve().parent

SAMPLE_EPISODES = [
    {
        "description": "1. Nominal System 1 reactive load shed (Safe)",
        "action": ProposedControlAction(
            action_id="act-demo-001-nominal",
            origin="SYSTEM1",
            breakers=[],
            load_shed=[LoadShedCommand(node_id="N8", shed_fraction=0.1117, priority_tier=3)],
            dispatch=[],
            rationale="S1 reactive shed vuln=0.14",
        ),
    },
    {
        "description": "2. System 2 survival dispatch under cyclone stress (Safe)",
        "action": ProposedControlAction(
            action_id="act-demo-002-survival",
            origin="SYSTEM2",
            breakers=[BreakerCommand(edge_id="E_crit_1", closed=True)],
            load_shed=[
                LoadShedCommand(node_id="N11", shed_fraction=0.25, priority_tier=3),
                LoadShedCommand(node_id="N8", shed_fraction=0.0983, priority_tier=3),
            ],
            dispatch=[DispatchSetpoint(node_id="N4", p_kw=147.37, q_kvar=10.0)],
            rationale="S2 survival opt vuln=0.63 protect-tier1",
        ),
    },
    {
        "description": "3. Unsafe action: Tripping tie lines causing island undervoltage (Reject)",
        "action": ProposedControlAction(
            action_id="act-demo-003-undervolt",
            origin="SYSTEM2",
            breakers=[
                BreakerCommand(edge_id="Line_2_3", closed=False),
                BreakerCommand(edge_id="E_crit_1", closed=False),
            ],
            load_shed=[],
            dispatch=[
                DispatchSetpoint(node_id="N8", p_kw=0.0, q_kvar=0.0),
                DispatchSetpoint(node_id="N9", p_kw=0.0, q_kvar=0.0),
            ],
            rationale="Aggressive island isolation without backup generation",
        ),
    },
    {
        "description": "4. Unsafe action: Reactive power over-injection (Reject)",
        "action": ProposedControlAction(
            action_id="act-demo-004-overvolt",
            origin="SYSTEM2",
            breakers=[],
            load_shed=[],
            dispatch=[DispatchSetpoint(node_id="N1", p_kw=500.0, q_kvar=6000.0)],
            rationale="Uncompensated voltage support attempt",
        ),
    },
    {
        "description": "5. Unsafe action: Excessive generator export (Reject)",
        "action": ProposedControlAction(
            action_id="act-demo-005-overload",
            origin="SYSTEM2",
            breakers=[],
            load_shed=[],
            dispatch=[DispatchSetpoint(node_id="N8", p_kw=6000.0, q_kvar=500.0)],
            rationale="Excessive export beyond subsea cable rating",
        ),
    },
    {
        "description": "6. Malformed action: Referencing non-existent grid components (Reject)",
        "action": ProposedControlAction(
            action_id="act-demo-006-malformed",
            origin="SYSTEM1",
            breakers=[BreakerCommand(edge_id="Line_Invalid_99", closed=False)],
            load_shed=[],
            dispatch=[],
            rationale="Faulty topology mapping",
        ),
    },
]


def run_demo() -> None:
    print("=" * 80)
    print("MetaCore Module 4 — OpenDSS Physics Firewall & Semantic Translation Demo")
    print("ZERO ML DEPENDENCIES · AUDITABLE PHYSICAL VERIFICATION")
    print("=" * 80)

    verifier = PhysicsVerifier()
    golden_records = []

    for ep in SAMPLE_EPISODES:
        print(f"\n--- Episode: {ep['description']} ---")
        action: ProposedControlAction = ep["action"]
        print(
            f"Action ID: {action.action_id} | Origin: {action.origin} | "
            f"Rationale: {action.rationale}"
        )

        # 1. Physics Verification in OpenDSS
        verdict = verifier.verify(action)
        print(
            f"Verdict: [{verdict.decision.value}] "
            f"(Power flow solve latency: {verdict.solve_latency_ms:.2f} ms)"
        )

        # 2. Abductive Attribution
        if verdict.violations:
            attributed_violations = AbductiveAttributor.attribute_violations(
                action, verdict.violations
            )
            verdict.violations = attributed_violations

        # 3. Feedback Rejection Trace (to M2)
        rejection_trace = verifier.build_rejection_trace(verdict.action_id, verdict.violations)
        if verdict.violations:
            print(
                f"Violations Detected: {len(verdict.violations)} | "
                f"Rejection Severity: {rejection_trace.severity:.4f}"
            )
            for v in verdict.violations:
                print(
                    f"  • [{v.type.value}] Element: {v.element_id} | "
                    f"Measured: {v.measured} | Limit: {v.limit} | "
                    f"Cause: {v.attributed_component}"
                )

        # 4. Grounded Causal Log (to Operator Dashboard)
        causal_log_live = TemplateCausalLogger.generate_log(verdict, include_latency=True)
        print(f'Causal Log: "{causal_log_live.text}"')
        if causal_log_live.grounded_entities:
            print(f"Grounded Entities: {causal_log_live.grounded_entities}")

        # Deterministic golden output (reproducible across all test environments)
        causal_log_golden = TemplateCausalLogger.generate_log(verdict, include_latency=False)
        golden_records.append(
            {
                "action_id": action.action_id,
                "decision": verdict.decision.value,
                "violations": [v.model_dump() for v in verdict.violations],
                "rejection_severity": rejection_trace.severity,
                "causal_log": causal_log_golden.model_dump(),
            }
        )

    # Write deterministic golden output artifact
    out_path = CORE_DIR / "sample_m4_output.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in golden_records:
            f.write(json.dumps(rec) + "\n")

    print("\n" + "=" * 80)
    print(f"Demo complete. Published {len(golden_records)} deterministic records to:")
    print(f"  {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
