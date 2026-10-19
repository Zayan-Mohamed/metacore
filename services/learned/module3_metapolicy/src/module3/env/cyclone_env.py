"""CycloneGatingEnv — the training environment for the Meta-RL gate.

One episode = one simulated cyclone. At every step the gate sees a 14-d observation and
picks System 1 (0) or System 2 (1). The chosen control path proposes an action; the grid
twin scores it under the storm's damage; the physical reward (reward.py) comes back.

    storm (scenario.cyclone) --\
    M2 uncertainty stream -----+--> observation --> GATE --> S1 or S2 --> action
                                                                            |
                            reward <-- twin (surrogate or live M4) <--------+

Differences from the older `gating_env.GatingEnv` (which is left untouched for the demo):
  * the grid is scored by a twin (physical reward) instead of a hand-written case table;
  * the storm causes real damage (feeder trips, PV/diesel loss, comms loss);
  * two extra observation slots carry breaker status and generation availability.

Budget rule is unchanged: when `budget_remaining <= 0` System 1 is forced and
`budget_exhausted_fallback` is reported. That is not an agent action.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from module3._flat import ensure_flat_on_path
from module3.control.planner import deliberative_plan
from module3.control.reactive import reactive_plan
from module3.env.reward import RewardConfig, step_reward
from module3.scenario.cyclone import HAZARD_STAGES, generate_episode, resolve_params
from module3.twin.base import Twin
from module3.twin.scoring import DEFAULT_TIER_WEIGHTS
from module3.twin.surrogate import SurrogateTwin
from module3.twin.topology import OVERHEAD_LINES

ensure_flat_on_path()
from m2_stream import _by_reason, _jitter, load_jsonl  # noqa: E402  (flat helpers)

OBS_DIM = 15

# One row per observation slot. Slots 0-11 are identical in meaning/order to the older env,
# so anything that reads u / reason / severity by index keeps working.
OBS_FIELDS = [
    {
        "index": 0,
        "name": "epistemic_uncertainty",
        "source": "M2",
        "meaning": "u: 0 sure ... 1 no evidence",
    },
    {
        "index": 1,
        "name": "reason_value",
        "source": "M2",
        "meaning": "1 if trigger_reason is value/both",
    },
    {
        "index": 2,
        "name": "reason_sensing",
        "source": "M2",
        "meaning": "1 if trigger_reason is sensing/both",
    },
    {"index": 3, "name": "state_class", "source": "M2", "meaning": "safety class scaled to 0..1"},
    {"index": 4, "name": "class_prob_0", "source": "M2", "meaning": "P(class 0)"},
    {"index": 5, "name": "class_prob_1", "source": "M2", "meaning": "P(class 1)"},
    {"index": 6, "name": "class_prob_2", "source": "M2", "meaning": "P(class 2)"},
    {
        "index": 7,
        "name": "max_node_vulnerability",
        "source": "M1",
        "meaning": "worst node exposure 0..1",
    },
    {
        "index": 8,
        "name": "severity",
        "source": "M1",
        "meaning": "hazard stage scaled 0/0.33/0.66/1",
    },
    {
        "index": 9,
        "name": "time_to_hazard",
        "source": "M1",
        "meaning": "minutes to onset, clipped +-30, /30",
    },
    {
        "index": 10,
        "name": "budget_remaining",
        "source": "env",
        "meaning": "S2 calls left, fraction of budget",
    },
    {
        "index": 11,
        "name": "observed_fraction",
        "source": "M2",
        "meaning": "share of features measured",
    },
    {
        "index": 12,
        "name": "tripped_visible",
        "source": "SCADA",
        "meaning": "share of overhead feeders known to be tripped (stale if comms down)",
    },
    {
        "index": 13,
        "name": "gen_available_visible",
        "source": "SCADA",
        "meaning": "mean PV/diesel availability, last seen",
    },
    {
        "index": 14,
        "name": "grid_lost_visible",
        "source": "SCADA",
        "meaning": "1 if the main supply is known to be lost (stale if comms down)",
    },
]


def describe_observation(obs) -> list[dict]:
    """Attach names/sources to a 15-d observation (for the dashboard)."""
    vec = [float(x) for x in obs]
    if len(vec) != OBS_DIM:
        raise ValueError(f"expected {OBS_DIM}-d observation, got {len(vec)}")
    return [{**f, "value": vec[f["index"]]} for f in OBS_FIELDS]


def _pick_reason(stage_idx: int, ood: bool, comms_up: bool, rng: np.random.Generator) -> str:
    """Which M2 trigger_reason this step shows. Telemetry loss -> sensing; unseen storm -> value."""
    if not comms_up:
        return "both" if (stage_idx >= 2 and (ood or rng.random() < 0.3)) else "sensing"
    if ood and stage_idx >= 1:
        return "value" if rng.random() < 0.9 else "none"
    if stage_idx >= 2:
        return "value" if rng.random() < 0.25 else "none"  # in-distribution: M2 is mostly calm
    if stage_idx == 1:
        return "value" if rng.random() < 0.10 else "none"
    return "none"


class CycloneGatingEnv(gym.Env):
    """Gymnasium env: observation (15,) float32, action Discrete(2), episode = one cyclone."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, twin: Twin | None = None, rng: np.random.Generator | None = None):
        super().__init__()
        self.cfg = cfg
        env_cfg = cfg["env"]
        self.episode_len = int(env_cfg["episode_len"])
        self.budget_total = float(env_cfg["budget_per_episode"])
        self.k = int(cfg.get("k_classes", 3))
        self.reward_cfg = RewardConfig.from_dict(cfg.get("reward"))
        weights = {
            int(k): float(v) for k, v in (cfg.get("tier_weights") or DEFAULT_TIER_WEIGHTS).items()
        }
        self._tier_weights = weights
        self.twin: Twin = twin if twin is not None else SurrogateTwin(weights)
        self._scenario_cfg = cfg.get("scenario") or {}
        # Memory test (docs section 26): observation slots the gate is NOT shown (zeroed).
        self._hidden = sorted({int(i) for i in env_cfg.get("hide_obs", [])})
        if any(not 0 <= i < OBS_DIM for i in self._hidden):
            raise ValueError(f"env.hide_obs must be indices in 0..{OBS_DIM - 1}")
        self._rng =rng if rng is not None else np.random.default_rng(int(cfg.get("seed", 0)))

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self._m2_buckets = _by_reason(load_jsonl(cfg.get("m2_stream_path")))
        self._episode = None
        self._m2: list[dict] = []
        self._t = 0
        self._budget = self.budget_total

    # ------------------------------------------------------------------ episode set-up
    def reset(self, *, seed=None, options=None):
        """Start a new cyclone. options={"cyclone": None | "ditwah_like" | CycloneParams}."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        params = resolve_params((options or {}).get("cyclone"), self._rng, self._scenario_cfg)
        self._episode = generate_episode(params, self.episode_len, self._rng)
        self._m2 = self._draw_m2_stream()
        self._t = 0
        self._budget = self.budget_total
        return self._obs(0), {
            "cyclone": params.name,
            "ood": params.ood,
            "peak_wind_kmh": params.peak_wind_kmh,
        }

    def _draw_m2_stream(self) -> list[dict]:
        """One M2 record per step, drawn from the published mock to match the storm's state."""
        ep = self._episode
        out = []
        for t in range(self.episode_len):
            stage_idx = HAZARD_STAGES.index(ep.stage[t])
            reason = _pick_reason(stage_idx, ep.params.ood, bool(ep.comms_up[t]), self._rng)
            pool = self._m2_buckets[reason]
            base = pool[int(self._rng.integers(0, len(pool)))]
            out.append(_jitter(base, self._rng))
        return out

    # ------------------------------------------------------------------ observation
    def _obs(self, t: int) -> np.ndarray:
        ep, m2 = self._episode, self._m2[t]
        ctx = ep.context(t)
        known_tripped, pv_k, diesel_k, grid_lost_k = ep.known(t)
        probs = list(m2["class_probabilities"])[: self.k]
        probs += [0.0] * (self.k - len(probs))
        reason = str(m2.get("trigger_reason", "none"))
        sev = HAZARD_STAGES.index(ctx["severity"])
        tth = float(np.clip(ctx["time_to_hazard_onset_min"], -30.0, 30.0)) / 30.0
        tripped_visible = len(known_tripped & set(OVERHEAD_LINES)) / max(len(OVERHEAD_LINES), 1)
        obs = np.array(
            [
                float(m2["epistemic_uncertainty"]),
                1.0 if reason in ("value", "both") else 0.0,
                1.0 if reason in ("sensing", "both") else 0.0,
                float(m2["state_class"]) / max(self.k - 1, 1),
                probs[0],
                probs[1],
                probs[2],
                float(ctx["max_node_vulnerability"]),
                sev / 3.0,
                tth,
                self._budget / max(self.budget_total, 1e-6),
                float(m2.get("observed_fraction", 1.0)),
                tripped_visible,
                0.5 * (pv_k + diesel_k),
                1.0 if grid_lost_k else 0.0,
            ],
            dtype=np.float32,
        )
        if self._hidden:
            obs[self._hidden] = 0.0
        return obs

    # ------------------------------------------------------------------ one step
    def step(self, action):
        t, ep = self._t, self._episode
        m2, ctx = self._m2[t], ep.context(t)
        reason = str(m2.get("trigger_reason", "none"))
        exhausted = self._budget <= 0
        effective = 0 if exhausted else int(action)

        if effective == 1:
            known_tripped, pv_k, diesel_k, grid_lost_k = ep.known(t)
            proposed = deliberative_plan(
                known_tripped,
                ctx,
                grid_lost=grid_lost_k,
                pv_factor=pv_k,
                diesel_factor=diesel_k,
                tier_weights=self._tier_weights,
            )
            self._budget -= 1.0
        else:
            proposed = reactive_plan(ctx, reason)

        outcome = self.twin.evaluate(proposed, ep.disturbance(t))
        reward, parts = step_reward(outcome, effective == 1, self.reward_cfg)

        info = {
            "requested_action": int(action),
            "effective_action": effective,
            "budget_exhausted_fallback": bool(exhausted and int(action) == 1),
            "proposed": proposed,
            "approved": outcome.approved,
            "violating_buses": outcome.violating_buses,
            "served_fraction": outcome.served_fraction,
            "reward_parts": parts,
            "severity": ctx["severity"],
            "trigger_reason": reason,
            "epistemic_uncertainty": float(m2["epistemic_uncertainty"]),
            "observed_fraction": float(m2.get("observed_fraction", 1.0)),
            "n_tripped_truth": len(ep.tripped[t]),
            "grid_lost_truth": ep.truth_grid_lost(t),
            "comms_up": bool(ep.comms_up[t]),
            "twin_source": outcome.source,
        }

        self._t += 1
        truncated = self._t >= self.episode_len
        obs = np.zeros(OBS_DIM, dtype=np.float32) if truncated else self._obs(self._t)
        return obs, reward, False, truncated, info
