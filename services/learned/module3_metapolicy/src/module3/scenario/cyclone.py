"""Simulated cyclone episodes — the 'weather' the Meta-RL gate is trained against.

One `CycloneEpisode` is a pre-computed storm: for every control step it holds the wind, the
per-node vulnerability M1 would report, and the physical damage the storm does to the grid
(the four disturbances agreed for M3):

    1. PV derate            wind above the cut-in speed progressively shuts PV down
    2. Overhead-feeder trips  probability tied to the vulnerability of the feeder's end node
    3. Diesel availability   a diesel unit can be lost once the storm is severe
    4. Communications cut    telemetry drops out; M2 then reports `sensing`

The storm does NOT depend on what the controller does (a cyclone does not care how the
grid responds), so the whole episode is generated up front from a seed. That makes runs
reproducible and lets every policy and baseline be scored on the *identical* storm.

Everything here is a modelling assumption, not measured data. The named profiles
(`burevi_like`, `ditwah_like`) are PLACEHOLDER parameter sets with the right qualitative
shape; they must be replaced with real best-track / wind data before any survivability
claim is made about those historical storms.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from module3.twin.base import Disturbance
from module3.twin.topology import (
    NODE_EXPOSURE,
    NODES,
    OVERHEAD_END_NODE,
    OVERHEAD_LINES,
)

HAZARD_STAGES = ("normal", "elevated", "severe", "extreme")
_TOP_COUNT = {"normal": 1, "elevated": 3, "severe": 5, "extreme": 8}
_MINUTES_PER_STEP_REF = 0.68  # same time scale as the older synthetic scenario (30 min / 44 steps)


@dataclass(frozen=True)
class CycloneParams:
    """The knobs that define one storm. A 'task' in the meta-RL sense."""

    name: str = "random"
    peak_wind_kmh: float = 120.0  # maximum sustained wind at the peak
    peak_frac: float = 0.55  # where in the episode the peak falls (0..1)
    width_frac: float = 0.12  # how long the storm lasts (std of the wind pulse, as a fraction)
    ood: bool = False  # 'unseen conditions' — M2 reports value-axis uncertainty
    ood_trip_boost: float = 1.5  # extra trip risk S1's assumptions do not anticipate
    trip_scale: float = 0.12  # per-step trip probability at vulnerability 1.0
    comms_rate: float = 0.10  # per-step chance telemetry drops, at peak wind
    comms_len: tuple[int, int] = (3, 8)
    diesel_loss_prob: float = 0.15  # per severe step, chance a diesel unit is lost (permanent)


# Placeholder parameter sets for the benchmark task — see module docstring.
PROFILES: dict[str, CycloneParams] = {
    "burevi_like": CycloneParams(
        name="burevi_like", peak_wind_kmh=95.0, peak_frac=0.5, width_frac=0.16, ood=False
    ),
    "ditwah_like": CycloneParams(
        name="ditwah_like", peak_wind_kmh=125.0, peak_frac=0.55, width_frac=0.10, ood=True
    ),
}


def sample_params(rng: np.random.Generator, cfg: dict | None = None) -> CycloneParams:
    """Draw a random storm from the training distribution (the 'task distribution')."""
    c = {
        "peak_wind_kmh": (45.0, 190.0),
        "peak_frac": (0.35, 0.70),
        "width_frac": (0.07, 0.20),
        "p_ood": 0.30,
        "trip_scale": 0.12,
        "comms_rate": 0.10,
        "diesel_loss_prob": 0.15,
        **(cfg or {}),
    }
    lo, hi = c["peak_wind_kmh"]
    return CycloneParams(
        name="random",
        peak_wind_kmh=float(rng.uniform(lo, hi)),
        peak_frac=float(rng.uniform(*c["peak_frac"])),
        width_frac=float(rng.uniform(*c["width_frac"])),
        ood=bool(rng.random() < c["p_ood"]),
        trip_scale=float(c["trip_scale"]),
        comms_rate=float(c["comms_rate"]),
        diesel_loss_prob=float(c["diesel_loss_prob"]),
    )


def resolve_params(spec, rng: np.random.Generator, cfg: dict | None = None) -> CycloneParams:
    """Turn None / a profile name / a CycloneParams into a CycloneParams."""
    if spec is None:
        return sample_params(rng, cfg)
    if isinstance(spec, CycloneParams):
        return spec
    if isinstance(spec, str):
        if spec not in PROFILES:
            raise KeyError(f"unknown cyclone profile {spec!r}; known: {sorted(PROFILES)}")
        return PROFILES[spec]
    raise TypeError(f"cannot resolve cyclone spec of type {type(spec).__name__}")


@dataclass
class CycloneEpisode:
    """A fully pre-computed storm. Index it by step with `state(t)`."""

    params: CycloneParams
    n_steps: int
    wind: np.ndarray  # (T,) km/h
    stage: list[str]  # (T,) hazard stage per step
    node_vuln: np.ndarray  # (T, len(NODES)) vulnerability per node
    tripped: list[frozenset[str]]  # (T,) lines physically out at each step
    pv_factor: np.ndarray  # (T,)
    diesel_factor: np.ndarray  # (T,)
    comms_up: np.ndarray  # (T,) bool
    onset_step: float
    top_at_risk: list[list[str]] = field(default_factory=list)

    def disturbance(self, t: int) -> Disturbance:
        """Ground truth: what is physically wrong with the grid at step t."""
        return Disturbance(
            tripped_lines=self.tripped[t],
            pv_factor=float(self.pv_factor[t]),
            diesel_factor=float(self.diesel_factor[t]),
            comms_up=bool(self.comms_up[t]),
        )

    def known(self, t: int) -> tuple[frozenset[str], float, float]:
        """What the controller can *see* at step t: (tripped lines, pv factor, diesel factor).

        With comms up it is the truth. With comms down it is the last value seen while comms
        were up — stale, and stale data is exactly what makes deliberation unable to help.
        """
        s = t
        while s > 0 and not self.comms_up[s]:
            s -= 1
        if not self.comms_up[s]:  # comms down from the very first step: assume healthy grid
            return frozenset(), 1.0, 1.0
        return self.tripped[s], float(self.pv_factor[s]), float(self.diesel_factor[s])

    def context(self, t: int) -> dict:
        """The M1-style `SystemContext` for step t (same keys the older flat code uses)."""
        v = self.node_vuln[t]
        mps = _MINUTES_PER_STEP_REF
        return {
            "max_node_vulnerability": float(v.max()),
            "mean_node_vulnerability": float(v.mean()),
            "top_at_risk_nodes": list(self.top_at_risk[t]),
            "time_to_hazard_onset_min": float((self.onset_step - t) * mps),
            "severity": self.stage[t],
            "wind_kmh": float(self.wind[t]),
        }


def _stage_for(wind: float, thresholds: tuple[float, float, float]) -> str:
    e, s, x = thresholds
    return (
        "extreme" if wind >= x else "severe" if wind >= s else "elevated" if wind >= e else "normal"
    )


def generate_episode(
    params: CycloneParams,
    n_steps: int,
    rng: np.random.Generator,
    stage_wind_kmh: tuple[float, float, float] = (40.0, 80.0, 120.0),
    pv_cut_in_out_kmh: tuple[float, float] = (50.0, 100.0),
    wind_ref_kmh: float = 140.0,
) -> CycloneEpisode:
    """Build one storm. All randomness comes from `rng`, so a seed reproduces it exactly."""
    t_idx = np.arange(n_steps, dtype=float)
    centre = params.peak_frac * n_steps
    sigma = max(params.width_frac * n_steps, 1.0)
    wind = params.peak_wind_kmh * np.exp(-0.5 * ((t_idx - centre) / sigma) ** 2)
    wind = np.clip(wind + rng.normal(0.0, 2.0, n_steps), 0.0, None)

    stage = [_stage_for(float(w), stage_wind_kmh) for w in wind]
    exposure = np.array([NODE_EXPOSURE[n] for n in NODES])
    base = np.clip((wind / wind_ref_kmh) ** 1.5, 0.0, 1.0)[:, None] * exposure[None, :]
    node_vuln = np.clip(base + rng.normal(0.0, 0.03, base.shape), 0.0, 1.0)

    top_at_risk = []
    for t in range(n_steps):
        order = np.argsort(-node_vuln[t])
        top_at_risk.append([NODES[i] for i in order[: _TOP_COUNT[stage[t]]]])

    # --- 2. overhead-feeder trips: absorbing (a damaged line stays out) ---
    boost = 1.0 + (params.ood_trip_boost if params.ood else 0.0)
    tripped: list[set[str]] = [set() for _ in range(n_steps)]
    out: set[str] = set()
    for t in range(n_steps):
        for line in OVERHEAD_LINES:
            if line in out:
                continue
            end = NODES.index(OVERHEAD_END_NODE[line])
            p = min(1.0, params.trip_scale * float(node_vuln[t, end]) ** 3 * boost)
            if rng.random() < p:
                out.add(line)
        tripped[t] = set(out)

    # --- 1. PV derate: linear from cut-in to cut-out wind ---
    cin, cout = pv_cut_in_out_kmh
    pv = np.clip(1.0 - (wind - cin) / max(cout - cin, 1e-6), 0.0, 1.0)

    # --- 3. diesel availability: one unit lost (permanently) once the storm is severe ---
    diesel = np.ones(n_steps)
    lost = False
    for t in range(n_steps):
        if (
            not lost
            and stage[t] in ("severe", "extreme")
            and rng.random() < params.diesel_loss_prob
        ):
            lost = True
        if lost:
            diesel[t] = 0.5

    # --- 4. communications cut: outages start more often at high wind and last a few steps ---
    comms = np.ones(n_steps, dtype=bool)
    t = 0
    while t < n_steps:
        p_drop = params.comms_rate * min(1.0, float(wind[t]) / wind_ref_kmh) ** 2
        if comms[t] and rng.random() < p_drop:
            length = int(rng.integers(params.comms_len[0], params.comms_len[1] + 1))
            comms[t : t + length] = False
            t += length
        else:
            t += 1

    onset = max(0.0, (params.peak_frac - 2.0 * params.width_frac) * n_steps)
    return CycloneEpisode(
        params=params,
        n_steps=n_steps,
        wind=wind,
        stage=stage,
        node_vuln=node_vuln,
        tripped=[frozenset(s) for s in tripped],
        pv_factor=pv,
        diesel_factor=diesel,
        comms_up=comms,
        onset_step=onset,
        top_at_risk=top_at_risk,
    )


def with_overrides(params: CycloneParams, **kw) -> CycloneParams:
    """Return a copy of `params` with some fields changed (handy for the UI / ablations)."""
    return replace(params, **kw)
