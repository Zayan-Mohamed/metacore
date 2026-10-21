"""Physics Limits Checker: Evaluates voltage, thermal ampacity and generation adequacy.

ZERO ML DEPENDENCIES.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from ..opendss.circuit import CircuitTwin
from ..types import Violation, ViolationType

PROFILES_DIR = Path(__file__).parent / "profiles"
DEFAULT_PROFILE = "METACORE-STRICT"


@dataclass
class SafetyLimitsConfig:
    """Configurable statutory bounds for island microgrids."""

    v_min_pu: float = 0.95
    v_max_pu: float = 1.05
    thermal_overload_margin: float = 0.0  # 0.0 means strictly <= NormAmps
    # Which versioned limit profile these numbers came from; recorded on every verdict.
    profile: str = f"{DEFAULT_PROFILE}-v1"

    @classmethod
    def from_profile(cls, profile_id: str = DEFAULT_PROFILE) -> "SafetyLimitsConfig":
        """Loads a versioned limit profile from firewall/profiles/ by its id (case-insensitive)."""
        spec = available_profiles().get(profile_id.upper())
        if spec is None:
            raise KeyError(f"unknown limit profile {profile_id!r}; "
                           f"known: {sorted(available_profiles())}")
        return cls(
            v_min_pu=float(spec["v_min_pu"]),
            v_max_pu=float(spec["v_max_pu"]),
            thermal_overload_margin=float(spec["thermal_overload_margin"]),
            profile=f"{spec['profile_id']}-v{spec['version']}",
        )


def available_profiles() -> dict[str, dict]:
    """Every profile file, keyed by upper-case profile_id."""
    out: dict[str, dict] = {}
    for f in sorted(PROFILES_DIR.glob("*.json")):
        spec = json.loads(f.read_text())
        out[spec["profile_id"].upper()] = spec
    return out


class PhysicsLimitsChecker:
    """Evaluates circuit state against physics boundaries and constructs Violation objects."""

    def __init__(self, config: SafetyLimitsConfig | None = None) -> None:
        self.config = config or SafetyLimitsConfig.from_profile(DEFAULT_PROFILE)

    def check_limits(
        self, circuit: CircuitTwin, converged: bool, adequacy: bool = True
    ) -> list[Violation]:
        """Runs comprehensive checks on voltage, line loading, adequacy and convergence.

        ``adequacy=False`` skips step 4 for callers that judge adequacy from demand they already
        know (the scenario loop, risk/certifier.py).
        """
        violations: list[Violation] = []

        # 1. Check AC power-flow convergence
        if not converged:
            violations.append(
                Violation(
                    type=ViolationType.VIOLATION_TYPE_NON_CONVERGENCE,
                    element_id="GLOBAL_CIRCUIT",
                    limit=1.0,
                    measured=0.0,
                    margin_fraction=-1.0,
                    attributed_component="powerflow_solver",
                )
            )
            return violations

        # 2. Check nodal bus voltages (v_min <= V_pu <= v_max)
        bus_voltages = circuit.get_bus_voltages_pu()
        for bus, v_pu in bus_voltages.items():
            if v_pu < self.config.v_min_pu:
                margin = (v_pu - self.config.v_min_pu) / self.config.v_min_pu
                violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_UNDERVOLTAGE,
                        element_id=bus,
                        limit=self.config.v_min_pu,
                        measured=round(v_pu, 4),
                        margin_fraction=round(margin, 4),
                    )
                )
            elif v_pu > self.config.v_max_pu:
                margin = (v_pu - self.config.v_max_pu) / self.config.v_max_pu
                violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_OVERVOLTAGE,
                        element_id=bus,
                        limit=self.config.v_max_pu,
                        measured=round(v_pu, 4),
                        margin_fraction=round(margin, 4),
                    )
                )

        # 3. Check thermal line loading margins (I <= NormAmps)
        line_loadings = circuit.get_line_loadings()
        for line, data in line_loadings.items():
            if data["enabled"] < 0.5:
                continue

            max_amps = data["max_amps"]
            norm_amps = data["norm_amps"]
            margin = data["margin_fraction"]

            limit_threshold = norm_amps * (1.0 + self.config.thermal_overload_margin)
            if norm_amps > 0.0 and max_amps > limit_threshold:
                violations.append(
                    Violation(
                        type=ViolationType.VIOLATION_TYPE_THERMAL_OVERLOAD,
                        element_id=line,
                        limit=norm_amps,
                        measured=round(max_amps, 2),
                        margin_fraction=round(margin, 4),
                    )
                )

        # 4. Generation adequacy per island (needs the topology pin for the lead-genset ratings)
        if adequacy:
            violations.extend(self.check_generation_adequacy(circuit))

        return violations

    @staticmethod
    def adequacy_parts(circuit: CircuitTwin) -> dict[str, dict[str, float | str]]:
        """Per island: station bus, capacity (kVA), load P/Q, and charging P/Q from generators.

        Capacity is the grid-forming lead genset's rating (from the pin; the model holds it as a
        stiff source) plus the kVA of every in-service Generator that is not absorbing power.
        Anything charging (a battery with negative kW) counts as demand instead. Returns {} for
        a model without a pin.
        """
        if not circuit.pin:
            return {}
        bus_island = {b["node_id"].upper(): b["island"] for b in circuit.pin["buses"]}
        parts: dict[str, dict[str, float | str]] = {}

        def part(island: str) -> dict[str, float | str]:
            return parts.setdefault(island, {"station": island, "capacity_kva": 0.0,
                                             "load_p": 0.0, "load_q": 0.0,
                                             "extra_p": 0.0, "extra_q": 0.0})

        for a in circuit.pin["assets"]:
            if a.get("grid_forming") and a.get("in_service", True):
                p = part(a["island"])
                p["capacity_kva"] += float(a["rating_kva"])
                p["station"] = a["bus"].upper()
        for name, (pk, qk) in circuit.get_load_demand().items():
            p = part(bus_island[circuit.load_bus(name)])
            p["load_p"] += pk
            p["load_q"] += qk
        for name, g in circuit.get_generator_states().items():
            if g["enabled"] < 0.5:
                continue
            p = part(bus_island[circuit.generator_bus(name)])
            if g["p_kw"] < 0.0:
                p["extra_p"] -= g["p_kw"]
                p["extra_q"] += max(0.0, -g["q_kvar"])
            else:
                p["capacity_kva"] += g["kva"]
        return dict(sorted(parts.items()))

    @staticmethod
    def island_adequacy(circuit: CircuitTwin) -> dict[str, dict[str, float | str]]:
        """Apparent-power demand against in-service generating capacity, per island."""
        return {
            island: {
                "station": p["station"],
                "demand_kva": math.hypot(float(p["load_p"]) + float(p["extra_p"]),
                                         float(p["load_q"]) + float(p["extra_q"])),
                "capacity_kva": float(p["capacity_kva"]),
            }
            for island, p in PhysicsLimitsChecker.adequacy_parts(circuit).items()
        }

    @staticmethod
    def inadequacy(island: str, station: str, demand: float, capacity: float) -> Violation:
        return Violation(
            type=ViolationType.VIOLATION_TYPE_GENERATION_INADEQUATE,
            element_id=station,
            limit=round(capacity, 2),
            measured=round(demand, 2),
            margin_fraction=round((demand - capacity) / capacity, 4),
            attributed_component=f"island.{island}",
        )

    def check_generation_adequacy(self, circuit: CircuitTwin) -> list[Violation]:
        """One GENERATION_INADEQUATE violation per island whose demand exceeds its capacity."""
        return [
            self.inadequacy(island, str(a["station"]), float(a["demand_kva"]),
                            float(a["capacity_kva"]))
            for island, a in self.island_adequacy(circuit).items()
            if float(a["demand_kva"]) > float(a["capacity_kva"])
        ]
