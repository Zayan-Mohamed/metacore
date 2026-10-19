from pathlib import Path

import numpy as np
import yaml
from module3.env.cyclone_env import CycloneGatingEnv
from module3.policy.recurrent_gate import RecurrentGate
from module3.train.evaluate import run_trials

ROOT = Path(__file__).resolve().parents[1]


def _setup():
    cfg = yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8"))
    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(0))
    return env, RecurrentGate(hidden=16, arch="gru").eval()


def test_exact_replay_repeats_the_same_storm():
    env, model = _setup()
    rows = run_trials(env, model, 7, 3, "exact")
    assert [r["trial"] for r in rows] == [1, 2, 3]
    assert len({round(r["always_s1"], 9) for r in rows}) == 1
    assert len({round(r["oracle"], 9) for r in rows}) == 1


def test_fresh_replay_redraws_the_damage():
    env, model = _setup()
    rows = run_trials(env, model, 7, 4, "fresh")
    assert len({round(r["always_s1"], 6) for r in rows}) > 1
