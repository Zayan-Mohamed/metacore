"""Power Flow Solver: Executes OpenDSS AC power flow and records execution latency.

ZERO ML DEPENDENCIES.
"""

import time

import opendssdirect as dss


class PowerFlowSolver:
    """Solves AC snapshot power flow using OpenDSSDirect and measures solve latency."""

    @staticmethod
    def solve_snapshot() -> tuple[bool, float]:
        """Executes a static AC power flow snapshot.

        Returns:
            (converged: bool, solve_latency_ms: float)
        """
        start_ns = time.perf_counter_ns()
        dss.Solution.Solve()
        end_ns = time.perf_counter_ns()

        converged = bool(dss.Solution.Converged())
        solve_latency_ms = (end_ns - start_ns) / 1_000_000.0

        return converged, solve_latency_ms
