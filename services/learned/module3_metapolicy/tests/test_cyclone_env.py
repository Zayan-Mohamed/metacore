"""Cyclone generator and CycloneGatingEnv invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from module3.env import OBS_DIM, CycloneGatingEnv, describe_observation
from module3.scenario import PROFILES, generate_episode, sample_params
from module3.twin.topology import NODES, OVERHEAD_LINES

ROOT = Path(__file__).resolve().parents[1]


def _cfg(**env):
    cfg = yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8"))
    cfg["env"].update(env)
    return cfg


def test_same_seed_same_storm():
    a = generate_episode(PROFILES["ditwah_like"], 80, np.random.default_rng(3))
    b = generate_episode(PROFILES["ditwah_like"], 80, np.random.default_rng(3))
    assert np.array_equal(a.wind, b.wind) and a.tripped == b.tripped


def test_storm_only_names_real_nodes_and_lines():
    ep = generate_episode(sample_params(np.random.default_rng(1)), 80, np.random.default_rng(1))
    assert all(n in NODES for t in range(80) for n in ep.top_at_risk[t])
    assert all(set(s) <= set(OVERHEAD_LINES) for s in ep.tripped)


def test_trips_are_permanent_and_vulnerability_bounded():
    ep = generate_episode(PROFILES["ditwah_like"], 80, np.random.default_rng(0))
    for t in range(1, 80):
        assert ep.tripped[t - 1] <= ep.tripped[t]
    assert ep.node_vuln.min() >= 0.0 and ep.node_vuln.max() <= 1.0


def test_known_state_is_stale_when_comms_are_down():
    for seed in range(30):  # find a storm that actually has an outage after step 0
        ep = generate_episode(PROFILES["ditwah_like"], 80, np.random.default_rng(seed))
        down = [t for t in range(1, 80) if not ep.comms_up[t] and ep.comms_up[t - 1]]
        if down:
            t = down[0]
            assert (
                ep.known(t)[0] == ep.tripped[t - 1]
            )  # frozen at the last value seen with comms up
            return
    raise AssertionError("no storm with a comms outage in 30 seeds")


def test_env_shapes_and_reward_parts():
    env = CycloneGatingEnv(_cfg(), rng=np.random.default_rng(0))
    obs, meta = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,) and len(describe_observation(obs)) == OBS_DIM
    assert obs.min() >= -1.0 and obs.max() <= 1.0
    _, r, term, trunc, info = env.step(0)
    parts = info["reward_parts"]
    assert r == parts["serve"] - parts["violation"] - parts["deliberation"] - parts["latency"]
    assert not term and not trunc


def test_budget_exhaustion_forces_system1():
    env = CycloneGatingEnv(_cfg(budget_per_episode=2, episode_len=10), rng=np.random.default_rng(0))
    env.reset(seed=0)
    eff = [env.step(1)[4]["effective_action"] for _ in range(5)]
    assert eff == [1, 1, 0, 0, 0]


def test_episode_ends_by_truncation_and_is_reproducible():
    def run(seed):
        env = CycloneGatingEnv(_cfg(episode_len=12), rng=np.random.default_rng(9))
        env.reset(seed=seed)
        total, done = 0.0, False
        while not done:
            _, r, _, done, _ = env.step(1)
            total += r
        return total

    assert run(4) == run(4)


def test_policies_face_the_identical_storm():
    env = CycloneGatingEnv(_cfg(), rng=np.random.default_rng(0))
    env.reset(seed=7)
    w1 = env._episode.wind.copy()
    env.step(1)
    env.reset(seed=7)
    assert np.array_equal(w1, env._episode.wind)
