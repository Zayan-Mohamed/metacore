"""Tests for OpenDSS Circuit Twin Loading and Base Power Flow."""

from verification.opendss.circuit import CircuitTwin
from verification.powerflow.solver import PowerFlowSolver


def test_circuit_twin_initialization() -> None:
    twin = CircuitTwin()
    buses = twin.get_all_buses()
    lines = twin.get_all_lines()
    generators = twin.get_all_generators()
    loads = twin.get_all_loads()

    assert len(buses) >= 7, f"Expected >= 7 buses, found {len(buses)}: {buses}"
    assert len(lines) >= 6, f"Expected >= 6 lines, found {len(lines)}: {lines}"
    assert len(generators) >= 3, f"Expected >= 3 generators, found {len(generators)}"
    assert len(loads) >= 4, f"Expected >= 4 loads, found {len(loads)}"


def test_base_power_flow_convergence() -> None:
    twin = CircuitTwin()
    converged, latency_ms = PowerFlowSolver.solve_snapshot()

    assert converged is True, "Baseline circuit should converge cleanly"
    assert latency_ms >= 0.0

    voltages = twin.get_bus_voltages_pu()
    assert len(voltages) > 0

    # In baseline state, all voltages should be within normal operating range
    for bus, v_pu in voltages.items():
        if bus != "SOURCEBUS":
            assert 0.90 <= v_pu <= 1.10, f"Bus {bus} voltage {v_pu} out of expected range"


def test_circuit_reset_to_base() -> None:
    twin = CircuitTwin()
    twin.set_line_state("Line_1_2", False)
    loadings_tripped = twin.get_line_loadings()
    assert loadings_tripped["LINE_1_2"]["enabled"] == 0.0

    twin.reset_to_base()
    loadings_reset = twin.get_line_loadings()
    assert loadings_reset["LINE_1_2"]["enabled"] == 1.0
