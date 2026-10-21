"""OpenDSS Circuit Twin Wrapper.

ZERO ML DEPENDENCIES. Sole interface to OpenDSSDirect.py.
"""

import json
from pathlib import Path

import opendssdirect as dss

MODELS_DIR = Path(__file__).parent / "models"

# The shared M1 -> M4 model (PR #13). Vendored byte-for-byte from packages/contracts so the module
# still builds and runs on its own (Dockerfile, wheel); tests/test_northern_islands_v1.py asserts
# both copies stay identical.
DEFAULT_MODEL_PATH = MODELS_DIR / "northern_islands_v1.dss"
DEFAULT_PIN_PATH = MODELS_DIR / "northern_islands_v1.pin.json"
DEFAULT_TOPOLOGY_ID = "northern-islands-v1"

_MISSING = "QUALITY_MISSING"


def load_pin(path: Path | str = DEFAULT_PIN_PATH) -> dict:
    """The topology pin: buses, lines, assets and loads with their provenance."""
    return json.loads(Path(path).read_text())


class CircuitTwin:
    """Manages an OpenDSS microgrid simulation instance and state extraction."""

    def __init__(
        self,
        dss_file_path: str | None = None,
        topology_id: str | None = None,
        pin: dict | None = None,
    ) -> None:
        self.dss_file_path = Path(dss_file_path) if dss_file_path else DEFAULT_MODEL_PATH
        # Which GridTopology this twin instantiates. A GridStateSnapshot for any other topology
        # is refused rather than half-applied (see apply_snapshot).
        self.topology_id = topology_id or (
            DEFAULT_TOPOLOGY_ID if self.dss_file_path == DEFAULT_MODEL_PATH
            else self.dss_file_path.stem
        )
        if pin is None and self.topology_id == DEFAULT_TOPOLOGY_ID:
            pin = load_pin()
        self.pin = pin
        self._initial_load_cache: dict[str, tuple[float, float]] = {}
        self._initial_gen_cache: dict[str, tuple[float, float]] = {}
        self._load_bus: dict[str, str] = {}
        self._gen_bus: dict[str, str] = {}
        self.compile_base_circuit()

    def compile_base_circuit(self) -> bool:
        """Compiles or resets the circuit to its baseline definition."""
        dss.Command(f'compile "{self.dss_file_path}"')
        self._index_elements()
        self._cache_base_equipment()
        return True

    def _index_elements(self) -> None:
        """Maps every load and generator to the bus it hangs off (names lower-cased)."""
        self._load_bus.clear()
        for name in dss.Loads.AllNames():
            dss.Circuit.SetActiveElement(f"Load.{name}")
            self._load_bus[name.lower()] = dss.CktElement.BusNames()[0].split(".")[0].lower()
        self._gen_bus.clear()
        for name in dss.Generators.AllNames():
            dss.Circuit.SetActiveElement(f"Generator.{name}")
            self._gen_bus[name.lower()] = dss.CktElement.BusNames()[0].split(".")[0].lower()

    def _cache_base_equipment(self) -> None:
        """Caches nominal generation and load ratings for relative adjustments."""
        self._initial_load_cache.clear()
        for load_name in dss.Loads.AllNames():
            dss.Loads.Name(load_name)
            p = float(dss.Loads.kW())
            q = float(dss.Loads.kvar())
            self._initial_load_cache[load_name.lower()] = (p, q)

        self._initial_gen_cache.clear()
        for gen_name in dss.Generators.AllNames():
            dss.Generators.Name(gen_name)
            p = float(dss.Generators.kW())
            q = float(dss.Generators.kvar())
            self._initial_gen_cache[gen_name.lower()] = (p, q)

    def reset_to_base(self) -> None:
        """Restores circuit to nominal baseline state."""
        self.compile_base_circuit()

    def apply_snapshot(self, snapshot: dict) -> list[str]:
        """Loads an M1 GridStateSnapshot (module1.proto) onto the twin.

        Sets feeder-bus demand from every node state whose p_kw is not QUALITY_MISSING, and
        disables the assets the snapshot lists as out of service. Fields the snapshot marks
        missing are left at the model's base value -- never zeroed. The applied state becomes
        the new reference for load shedding, so a shed fraction is a fraction of *this hour's*
        demand. Returns a list of problems; any problem means the snapshot was not applied
        consistently and the caller must not approve.
        """
        problems: list[str] = []
        if snapshot.get("topology_id") != self.topology_id:
            return [f"snapshot topology {snapshot.get('topology_id')!r} != twin "
                    f"{self.topology_id!r}"]

        loads_by_bus = self.loads_by_bus()
        for ns in snapshot.get("node_states", []):
            quality = ns.get("quality", {})
            if ns.get("p_kw") is None or quality.get("p_kw") == _MISSING:
                continue
            names = loads_by_bus.get(str(ns["node_id"]).lower())
            if not names:
                problems.append(f"no load at bus {ns['node_id']!r}")
                continue
            p_total = float(ns["p_kw"])
            q_total = float(ns["q_kvar"]) if ns.get("q_kvar") is not None else None
            base = [self._initial_load_cache.get(n, (0.0, 0.0)) for n in names]
            base_p = sum(p for p, _ in base)
            for name, (p0, _q0) in zip(names, base, strict=True):
                # Split across the bus's loads in proportion to their base demand.
                w = (p0 / base_p) if base_p else 1.0 / len(names)
                dss.Loads.Name(name)
                dss.Loads.kW(p_total * w)
                if q_total is not None:
                    dss.Loads.kvar(q_total * w)

        for element in snapshot.get("assets_out_of_service", []):
            if element.lower() not in self._gen_bus:
                problems.append(f"no generator {element!r} to take out of service")
                continue
            dss.Circuit.SetActiveElement(f"Generator.{element}")
            dss.CktElement.Enabled(False)

        self._cache_base_equipment()
        return problems

    def load_bus(self, load_name: str) -> str:
        """Upper-case bus a load hangs off."""
        return self._load_bus[load_name.lower()].upper()

    def generator_bus(self, gen_name: str) -> str:
        """Upper-case bus a generator hangs off."""
        return self._gen_bus[gen_name.lower()].upper()

    def loads_by_bus(self) -> dict[str, list[str]]:
        """Load names (lower-case) grouped by the bus they hang off."""
        out: dict[str, list[str]] = {}
        for name, bus in self._load_bus.items():
            out.setdefault(bus, []).append(name)
        return out

    def get_all_buses(self) -> list[str]:
        """Returns all bus names in the active circuit."""
        return [b.upper() for b in dss.Circuit.AllBusNames()]

    def get_all_lines(self) -> list[str]:
        """Returns all line names in the active circuit."""
        return [line.upper() for line in dss.Lines.AllNames()]

    def get_all_generators(self) -> list[str]:
        """Returns all generator names in the active circuit."""
        return [gen.upper() for gen in dss.Generators.AllNames()]

    def get_all_loads(self) -> list[str]:
        """Returns all load names in the active circuit."""
        return [load.upper() for load in dss.Loads.AllNames()]

    def get_bus_voltages_pu(self) -> dict[str, float]:
        """Returns minimum per-unit voltage magnitude for each bus across all active phases."""
        # One call for every node instead of one SetActiveBus per bus: the scenario loop reads
        # voltages hundreds of times per verify call.
        names = dss.Circuit.AllNodeNames()
        mags = dss.Circuit.AllBusMagPu()
        bus_voltages: dict[str, float] = {}
        for node, v in zip(names, mags, strict=True):
            bus = node.split(".")[0].upper()
            v = float(v)
            if bus not in bus_voltages or v < bus_voltages[bus]:
                bus_voltages[bus] = v
        return bus_voltages

    def get_line_loadings(self) -> dict[str, dict[str, float]]:
        """Returns line current magnitudes and rating margins."""
        line_data: dict[str, dict[str, float]] = {}
        for line_name in dss.Lines.AllNames():
            dss.Lines.Name(line_name)
            dss.Circuit.SetActiveElement(f"Line.{line_name}")
            norm_amps = float(dss.Lines.NormAmps())
            is_enabled = bool(dss.CktElement.Enabled())
            currents = dss.CktElement.CurrentsMagAng()
            max_current = 0.0
            if currents and len(currents) >= 2 and is_enabled:
                mags = [float(currents[i]) for i in range(0, min(6, len(currents)), 2)]
                if mags:
                    max_current = max(mags)

            margin_fraction = (max_current - norm_amps) / norm_amps if norm_amps > 0 else 0.0
            line_data[line_name.upper()] = {
                "max_amps": max_current,
                "norm_amps": norm_amps,
                "margin_fraction": margin_fraction,
                "enabled": 1.0 if is_enabled else 0.0,
            }
        return line_data

    def get_load_demand(self) -> dict[str, tuple[float, float]]:
        """Current (kW, kvar) setting of every load, keyed by upper-case load name."""
        out: dict[str, tuple[float, float]] = {}
        for name in dss.Loads.AllNames():
            dss.Loads.Name(name)
            out[name.upper()] = (float(dss.Loads.kW()), float(dss.Loads.kvar()))
        return out

    def get_generator_states(self) -> dict[str, dict[str, float]]:
        """Setpoint, rating and in-service flag of every Generator element."""
        out: dict[str, dict[str, float]] = {}
        for name in dss.Generators.AllNames():
            dss.Generators.Name(name)
            dss.Circuit.SetActiveElement(f"Generator.{name}")
            out[name.upper()] = {
                "p_kw": float(dss.Generators.kW()),
                "q_kvar": float(dss.Generators.kvar()),
                "kva": float(dss.Generators.kVARated()),
                "enabled": 1.0 if dss.CktElement.Enabled() else 0.0,
            }
        return out

    def set_load_kw(self, demand: dict[str, tuple[float, float]]) -> None:
        """Sets (kW, kvar) on loads in place, without recompiling (the scenario fast path)."""
        for name, (p, q) in demand.items():
            dss.Loads.Name(name)
            dss.Loads.kW(p)
            dss.Loads.kvar(q)

    def set_line_state(self, edge_id: str, closed: bool) -> bool:
        """Toggles a line or tie-breaker switch."""
        for line_name in dss.Lines.AllNames():
            if line_name.lower() == edge_id.lower():
                dss.Circuit.SetActiveElement(f"Line.{line_name}")
                dss.CktElement.Enabled(closed)
                return True
        return False

    def set_load_shed(self, node_id: str, shed_fraction: float) -> bool:
        """Sheds a fraction of this hour's demand at a feeder-group bus or on one named load.

        Matching is exact. A feeder-group bus (e.g. ``NAI_F2``) sheds every load on it; a load
        name (e.g. ``NAI_F2_LOAD``) sheds just that load.
        """
        shed_fraction = max(0.0, min(1.0, shed_fraction))
        key = node_id.lower()
        targets = [n for n, bus in self._load_bus.items() if bus == key or n == key]
        for name in targets:
            base_p, base_q = self._initial_load_cache[name]
            dss.Loads.Name(name)
            dss.Loads.kW(base_p * (1.0 - shed_fraction))
            dss.Loads.kvar(base_q * (1.0 - shed_fraction))
        return bool(targets)

    def set_generator_dispatch(self, node_id: str, p_kw: float, q_kvar: float = 0.0) -> bool:
        """Sets active (kW) and reactive (kvar) output on a generator or at a bus.

        A generator name (e.g. ``ELU_HYB_BESS``) sets that unit, if it is in service. A bus name
        (e.g. ``NAI_PS``) splits the setpoint evenly across the in-service generators on that
        bus, so asking a station for 100 kW never produces 100 kW per unit.
        """
        key = node_id.lower()
        if key in self._gen_bus:
            # A unit the snapshot took out of service cannot be dispatched: refuse the command
            # (MALFORMED) instead of silently setting a setpoint nothing will follow.
            dss.Circuit.SetActiveElement(f"Generator.{key}")
            targets = [key] if dss.CktElement.Enabled() else []
        else:
            targets = []
            for name, bus in self._gen_bus.items():
                if bus != key:
                    continue
                dss.Circuit.SetActiveElement(f"Generator.{name}")
                if dss.CktElement.Enabled():
                    targets.append(name)
        for name in targets:
            dss.Generators.Name(name)
            dss.Generators.kW(p_kw / len(targets))
            dss.Generators.kvar(q_kvar / len(targets))
        return bool(targets)
