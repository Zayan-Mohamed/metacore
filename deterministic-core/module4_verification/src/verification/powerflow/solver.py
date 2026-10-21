"""Power Flow Solver: Executes OpenDSS AC power flow and records execution latency.

ZERO ML DEPENDENCIES.
"""

import time

import opendssdirect as dss

from ..opendss.circuit import CircuitTwin


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

    @staticmethod
    def solve_with_demand(
        circuit: CircuitTwin, demand: dict[str, tuple[float, float]]
    ) -> tuple[bool, float]:
        """Sets (kW, kvar) on the named loads in place, then solves.

        The scenario fast path: a recompile costs ~5 ms, setting eight loads costs microseconds,
        and everything else the action changed (dispatch, breakers) stays as it was applied.
        """
        circuit.set_load_kw(demand)
        return PowerFlowSolver.solve_snapshot()
