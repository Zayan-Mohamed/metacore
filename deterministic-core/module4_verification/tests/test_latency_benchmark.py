"""Benchmark Tests: Asserts solve and limit check latency targets (< 50 ms)."""

import statistics

from verification.firewall.verifier import PhysicsVerifier
from verification.types import LoadShedCommand, ProposedControlAction


def test_verification_latency_benchmark() -> None:
    verifier = PhysicsVerifier()
    action = ProposedControlAction(
        action_id="bench-action",
        origin="SYSTEM1",
        breakers=[],
        load_shed=[LoadShedCommand(node_id="N8", shed_fraction=0.10, priority_tier=3)],
        dispatch=[],
        rationale="Latency benchmark check",
    )

    latencies_ms = []
    # Run 25 consecutive verification loops
    for _ in range(25):
        verdict = verifier.verify(action)
        latencies_ms.append(verdict.solve_latency_ms)

    mean_latency = statistics.mean(latencies_ms)
    p95_latency = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]

    print(f"\n[M4 Verification Benchmark] Mean: {mean_latency:.2f} ms | P95: {p95_latency:.2f} ms")

    # Assert that mean latency is well under the 50 ms real-time control budget
    assert mean_latency < 50.0, f"Mean latency {mean_latency:.2f} ms exceeded 50 ms budget"
