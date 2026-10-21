"""RiskCertifiedVerifier: Check A, then Check B, then the certificate (Steps 4-5, and 6 in storms).

    Check A   the action on this hour's snapshot (firewall/verifier.py). A REJECT or UNVERIFIED
              here is final: no statistics can rescue an action that already breaks a limit.
    Check B   the action on N seeded scenarios of the next six hours, drawn from the calibrated
              band for this hour's situation (adapted by conformal PID in storm mode). Each
              scenario is solved at its peak and its trough; k of N fail.
    Verdict   Clopper-Pearson on k of N against eps: APPROVE / REJECT / UNVERIFIED.

The runtime-assurance pattern (Sha 2001, the Simplex architecture; Alshiekh et al. 2018, safety
shields): the untrusted planner (M3) proposes, this trusted checker decides, and anything it
cannot establish is "not approved". Approval needs a clean Check A *and* a Check B upper bound at
or below eps.

Deterministic: same snapshot, action, calibration and seed give a byte-identical certificate.
Scenarios set loads in place on the already-applied action; nothing is recompiled per scenario.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..firewall.limits import DEFAULT_PROFILE, PhysicsLimitsChecker
from ..firewall.verifier import PhysicsVerifier, action_number_problems
from ..opendss.circuit import CircuitTwin
from ..powerflow.solver import PowerFlowSolver
from ..types import (
    Decision,
    ProposedControlAction,
    RiskAttribution,
    RiskCertificate,
    VerificationVerdict,
    Violation,
)
from .attribution import coalition_values, island_subsets, players_from_pin, shapley_values
from .bands import Band
from .calibration import Calibration, StaleCalibrationError
from .pid import ReplayHistory, in_storm, replay_pid
from .scenarios import ScenarioSet, draw_scenarios
from .situation import Situation
from .statistics import (
    APPROVE,
    BATCH,
    CONFIDENCE,
    EPSILON,
    REJECT,
    SequentialTest,
    required_scenarios,
)

_DECISION = {APPROVE: Decision.DECISION_APPROVE, REJECT: Decision.DECISION_REJECT}


@dataclass
class CertifierConfig:
    """The stated design choices of Check B (the defaults are the plan's)."""

    epsilon: float = EPSILON
    confidence: float = CONFIDENCE
    batch: int = BATCH
    n_max: int | None = None  # None -> the zero-failure sample size for (epsilon, confidence)
    pid: str = "auto"  # "auto" (storm mode only), "on", "off"
    seed: int | None = None  # None -> the calibration artifact's seed
    band_scale: float = 1.0  # sensitivity analysis (E5): scale both band edges
    situation_blind: bool = False  # baseline B2: one band and one row pool for every hour
    early_stopping: bool = True
    attribution: bool = True  # Shapley shares over the feeder players when k > 0

    def planned(self) -> int:
        return self.n_max or required_scenarios(self.epsilon, self.confidence)


@dataclass
class ScenarioOutcome:
    """One scenario, solved at its peak and its trough."""

    failed: bool
    islands: frozenset[str]
    violations: list[Violation] = field(default_factory=list)


class RiskCertifiedVerifier:
    """The risk-certified safety shield for M3's proposed actions."""

    def __init__(
        self,
        circuit_twin: CircuitTwin | None = None,
        profile: str = DEFAULT_PROFILE,
        calibration: Calibration | Path | str | None = None,
        history: ReplayHistory | None = None,
        config: CertifierConfig | None = None,
        input_paths: dict[str, Path] | None = None,
    ) -> None:
        self.gate = PhysicsVerifier(circuit_twin, profile)
        self.circuit = self.gate.circuit
        self.limits = self.gate.limits_checker
        self.config = config or CertifierConfig()
        self.history = history
        self._calibration_error: str | None = None
        try:
            self.calibration = (calibration if isinstance(calibration, Calibration)
                                else Calibration.load(calibration) if calibration
                                else Calibration.load())
            if input_paths:
                self.calibration.check_inputs(input_paths["hourly_load"],
                                              input_paths["meteorology"])
        except (StaleCalibrationError, OSError, ValueError) as exc:
            self.calibration = None
            self._calibration_error = str(exc)
        self._island_of = _element_islands(self.circuit)

    # ------------------------------------------------------------------ public

    def verify(
        self,
        action: ProposedControlAction,
        snapshot: dict | None = None,
        timestamp: str | None = None,
        hazard_level: int | None = None,
    ) -> VerificationVerdict:
        """Check A, then Check B. ``timestamp`` defaults to the snapshot's ``timestamp_lst``."""
        try:
            return self._verify(action, snapshot, timestamp, hazard_level)
        except Exception as exc:  # noqa: BLE001 -- the fail-safe, as in Check A
            return self.gate._unverified(action.action_id, [f"certifier error: {exc!r}"])
        finally:
            try:
                self.circuit.reset_to_base()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ internals

    def _verify(self, action, snapshot, timestamp, hazard_level) -> VerificationVerdict:
        check_a = self.gate.verify(action, snapshot)
        if check_a.decision != Decision.DECISION_APPROVE:
            return check_a

        timestamp = timestamp or (snapshot or {}).get("timestamp_lst")
        if self.calibration is None:
            return self.gate._unverified(
                action.action_id, [f"calibration unavailable: {self._calibration_error}"])
        if not timestamp:
            return self.gate._unverified(
                action.action_id, ["no recorded hour to certify: pass a snapshot or timestamp"])

        cal = self.calibration
        notes: list[str] = []
        situation, band, calibrated, pid_json = self._band(timestamp, hazard_level, notes)
        scenarios = draw_scenarios(cal, band, self.config.planned(), self.config.seed,
                                   static_band=calibrated if band != calibrated else None)

        # Re-apply the snapshot and the action once; scenarios then only move the loads.
        self._prepare(action, snapshot)
        base = self.circuit.get_load_demand()
        load_island = {name: self._island_of[self.circuit.load_bus(name)] for name in base}
        adequacy = PhysicsLimitsChecker.adequacy_parts(self.circuit)

        test = SequentialTest(n_max=len(scenarios), eps=self.config.epsilon,
                              confidence=self.config.confidence, batch=self.config.batch,
                              early=self.config.early_stopping)
        outcomes: list[ScenarioOutcome] = []
        solve_ms = 0.0
        for boundary in test.boundaries():
            batch = range(test.n, boundary)
            failures = 0
            for s in batch:
                outcome, ms = self._scenario(base, load_island, adequacy,
                                             scenarios.hi[s], scenarios.lo[s], scenarios)
                solve_ms += ms
                outcomes.append(outcome)
                failures += int(outcome.failed)
            if test.record(failures, len(batch)):
                break

        top, bottom = scenarios.corners()
        corner_top, ms_t = self._scenario(base, load_island, adequacy, top, top, scenarios)
        corner_bottom, ms_b = self._scenario(base, load_island, adequacy, bottom, bottom,
                                             scenarios)
        solve_ms += ms_t + ms_b

        attribution: list[RiskAttribution] = []
        if self.config.attribution and test.k > 0 and self.circuit.pin:
            attribution, ms_a = self._attribute(base, load_island, adequacy, scenarios,
                                                outcomes)
            solve_ms += ms_a

        verdict_name = test.verdict()
        decision = _DECISION.get(verdict_name, Decision.DECISION_UNVERIFIED)
        lower, upper = test.bounds()
        certificate = RiskCertificate(
            decision=decision,
            n_scenarios=test.n,
            n_max=test.n_max,
            n_failures=test.k,
            lower_bound=round(lower, 6),
            upper_bound=round(upper, 6),
            epsilon=self.config.epsilon,
            confidence=self.config.confidence,
            stopped_early=test.stopped_early,
            horizon_h=cal.horizon_h,
            limit_profile=self.limits.config.profile,
            calibration_version=cal.version,
            calibration_sha256=cal.sha256,
            seed=scenarios.seed,
            timestamp=timestamp,
            situation={"hour": situation.hour, "season": situation.season,
                       "hazard": situation.hazard_name},
            band={"key": band.key, "n": band.n, "q_up": _r(band.q_up),
                  "q_down": _r(band.q_down)},
            pid=pid_json,
            corner_top_safe=not corner_top.failed,
            corner_bottom_safe=not corner_bottom.failed,
            failures_by_island=_count_islands(outcomes, cal.islands),
            failing_elements=_count_elements(outcomes),
            notes=notes,
        )
        verdict = self.gate._build_verdict(
            action_id=action.action_id,
            decision=decision,
            violations=_summarise(outcomes, test.n, cal.horizon_h)
            if decision != Decision.DECISION_APPROVE
            else [],
            latency_ms=round(solve_ms, 3),
        )
        verdict.certificate = certificate
        verdict.attribution = attribution
        if decision == Decision.DECISION_UNVERIFIED:
            verdict.unverified_reasons = [
                f"risk bound straddles the target: {test.k} of {test.n} scenarios failed, "
                f"p in [{lower:.3%}, {upper:.3%}] vs eps {self.config.epsilon:.0%}"]
        return verdict

    def _band(self, timestamp: str, hazard_level: int | None,
              notes: list[str]) -> tuple[Situation, Band, Band, dict | None]:
        cal = self.calibration
        hist = self.history
        if hazard_level is None and hist is not None and timestamp in hist.timestamps:
            hazard_level = hist.hazard_at(timestamp)
        if hazard_level is None:
            notes.append("hazard tag missing for this hour: band looked up without hazard")
        situation = Situation.at(timestamp, hazard_level)
        # ``calibrated`` is the frozen band the rows were drawn under; ``static`` is the band this
        # call uses before PID (they differ only in the band-width sensitivity analysis, E5).
        # The scenario draws are rescaled from one to the other, so a wider band really spreads
        # the scenarios instead of only moving the corner rows.
        calibrated = (cal.bands["all"] if self.config.situation_blind
                      else cal.band_for(situation))
        static = calibrated
        if self.config.band_scale != 1.0:
            static = Band(f"{calibrated.key}x{self.config.band_scale:g}", calibrated.n,
                          calibrated.q_up * self.config.band_scale,
                          min(1.0, calibrated.q_down * self.config.band_scale), calibrated.alpha)
        if not self.config.situation_blind and calibrated.key != situation.keys()[0]:
            notes.append(f"situation group {situation.keys()[0]!r} too thin: "
                         f"fell back to {calibrated.key!r}")

        mode = "off" if self.config.situation_blind else self.config.pid
        if mode == "off":
            return situation, static, calibrated, None
        if hist is None or timestamp not in hist.timestamps:
            notes.append("conformal PID needs the replayed history; static band used")
            return situation, static, calibrated, None
        storm = in_storm(hist, hist.index(timestamp))
        if mode == "auto" and not storm:
            return situation, static, calibrated, {"enabled": False, "storm": False}
        adapted, state = replay_pid(cal, hist, timestamp)
        if self.config.band_scale != 1.0:
            adapted = static.widened(state.extra, "+pid")
        notes.append("storm mode: band adapted by conformal PID" if storm
                     else "conformal PID forced on")
        return situation, adapted, calibrated, {"enabled": True, "storm": storm,
                                                **state.to_json()}

    def _prepare(self, action: ProposedControlAction, snapshot: dict | None) -> None:
        self.circuit.reset_to_base()
        if snapshot is not None:
            problems = self.circuit.apply_snapshot(snapshot)
            if problems:  # Check A passed on the same snapshot, so this cannot happen
                raise RuntimeError(f"snapshot no longer applies: {problems}")
        if action_number_problems(action) or self.gate.applicator.apply_action(action):
            raise RuntimeError("action no longer applies")

    def _scenario(self, base: dict[str, tuple[float, float]], load_island: dict[str, str],
                  adequacy: dict[str, dict], hi: np.ndarray, lo: np.ndarray,
                  scenarios: ScenarioSet, moving: frozenset[str] | None = None,
                  ) -> tuple[ScenarioOutcome, float]:
        """Solves one scenario at its peak and its trough.

        ``moving`` restricts which loads take the scenario's multipliers (attribution); the
        rest stay at this hour's snapshot. None means every load moves.
        """
        index = {isl: j for j, isl in enumerate(scenarios.islands)}
        islands: set[str] = set()
        violations: list[Violation] = []
        ms = 0.0
        for factors in (hi, lo) if not np.array_equal(hi, lo) else (hi,):
            demand = {}
            for name, (p, q) in base.items():
                f = (factors[index[load_island[name]]]
                     if moving is None or name in moving else 1.0)
                demand[name] = (p * f, q * f)
            converged, solve_ms = PowerFlowSolver.solve_with_demand(self.circuit, demand)
            ms += solve_ms
            found = self.limits.check_limits(self.circuit, converged, adequacy=False)
            found += self._adequacy(adequacy, demand, load_island)
            for v in found:
                isl = self._island_of.get(v.element_id.upper())
                # A failure with no island (non-convergence) counts against every island.
                islands.update([isl] if isl else scenarios.islands)
            violations.extend(found)
        return ScenarioOutcome(bool(violations), frozenset(islands), violations), ms

    @staticmethod
    def _adequacy(parts: dict[str, dict], demand: dict[str, tuple[float, float]],
                  load_island: dict[str, str]) -> list[Violation]:
        """Adequacy from demand already known in Python (no OpenDSS reads per scenario)."""
        p_by: dict[str, float] = {}
        q_by: dict[str, float] = {}
        for name, (p, q) in demand.items():
            isl = load_island[name]
            p_by[isl] = p_by.get(isl, 0.0) + p
            q_by[isl] = q_by.get(isl, 0.0) + q
        out: list[Violation] = []
        for island, part in parts.items():
            kva = math.hypot(p_by.get(island, 0.0) + part["extra_p"],
                             q_by.get(island, 0.0) + part["extra_q"])
            if kva > part["capacity_kva"]:
                out.append(PhysicsLimitsChecker.inadequacy(island, str(part["station"]), kva,
                                                           float(part["capacity_kva"])))
        return out

    def _attribute(self, base: dict[str, tuple[float, float]], load_island: dict[str, str],
                   adequacy: dict[str, dict], scenarios: ScenarioSet,
                   outcomes: list[ScenarioOutcome]) -> tuple[list[RiskAttribution], float]:
        """Shapley shares of the failure rate over the feeder players (risk/attribution.py)."""
        players = players_from_pin(self.circuit.pin)
        load_player = {}
        for name in base:
            bus = self.circuit.load_bus(name)
            load_player[name] = next(i for i, p in enumerate(players) if bus in p.buses)
        members: dict[str, frozenset[int]] = {}
        for i, p in enumerate(players):
            members[p.island] = members.get(p.island, frozenset()) | {i}

        table: list[dict[str, dict]] = [
            {isl: {frozenset(): False, m: isl in outcome.islands} for isl, m in members.items()}
            for outcome in outcomes
        ]
        ms = 0.0
        for config in island_subsets(players):
            chosen = frozenset().union(*config.values())
            moving = frozenset(n for n, i in load_player.items() if i in chosen)
            for s in range(len(outcomes)):
                outcome, solve_ms = self._scenario(base, load_island, adequacy,
                                                   scenarios.hi[s], scenarios.lo[s], scenarios,
                                                   moving)
                ms += solve_ms
                for isl, subset in config.items():
                    table[s][isl][subset] = isl in outcome.islands

        values = coalition_values(players, table)
        phi = shapley_values(len(players), values)
        total = values[frozenset(range(len(players)))]
        result = [
            RiskAttribution(
                player=p.name,
                island=p.island,
                buses=sorted(p.buses),
                share=round(phi[i], 6),
                fraction_of_risk=round(phi[i] / total, 6) if total > 0 else 0.0,
                alone=round(values[frozenset({i})], 6),
            )
            for i, p in enumerate(players)
        ]
        result.sort(key=lambda a: (-a.share, a.player))
        return result, ms


# ---------------------------------------------------------------------- helpers

def _r(x: float) -> float:
    return round(float(x), 4) if math.isfinite(x) else -1.0


def _element_islands(circuit: CircuitTwin) -> dict[str, str]:
    """Bus, line and station names (upper-case) -> island, from the topology pin."""
    if not circuit.pin:
        return {}
    out = {b["node_id"].upper(): b["island"] for b in circuit.pin["buses"]}
    for line in circuit.pin["lines"]:
        out[line["edge_id"].upper()] = out[line["from_node"].upper()]
    return out


def _count_islands(outcomes: list[ScenarioOutcome], islands: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(i for o in outcomes for i in o.islands)
    return {i: counts.get(i, 0) for i in islands}


def _count_elements(outcomes: list[ScenarioOutcome]) -> dict[str, int]:
    counts = Counter(e for o in outcomes for e in {v.element_id for v in o.violations})
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10])


def _summarise(outcomes: list[ScenarioOutcome], n: int, horizon: int) -> list[Violation]:
    """Check B's evidence as Violations: per element, its worst case and how often it failed."""
    worst: dict[tuple[str, str], Violation] = {}
    count: Counter = Counter()
    for o in outcomes:
        seen = set()
        for v in o.violations:
            key = (v.type, v.element_id)
            if key not in seen:
                count[key] += 1
                seen.add(key)
            if key not in worst or abs(v.margin_fraction) > abs(worst[key].margin_fraction):
                worst[key] = v
    ranked = sorted(worst, key=lambda k: (-count[k], k))[:10]
    return [worst[k].model_copy(update={
        "attributed_component": f"next-{horizon}h scenarios: {count[k]}/{n}"}) for k in ranked]
