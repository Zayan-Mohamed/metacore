"""RecurrentGate, GAE and a tiny end-to-end PPO run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from module3.env import CycloneGatingEnv
from module3.policy import EXTRA_DIM, OBS_DIM, RecurrentGate, build_inputs
from module3.train.evaluate import evaluate_policy
from module3.train.ppo import PPOConfig, compute_gae, train_gate

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    cfg = yaml.safe_load((ROOT / "pipeline.yaml").read_text(encoding="utf-8"))
    cfg["env"].update(episode_len=20, budget_per_episode=8)
    return cfg


def _x(batch=2, steps=5, budget=1.0):
    obs = torch.rand(batch, steps, OBS_DIM) * 0.5
    obs[..., 10] = budget
    return build_inputs(
        obs,
        torch.zeros(batch, steps, dtype=torch.long),
        torch.zeros(batch, steps),
        torch.zeros(batch, steps),
    )


def test_forward_shapes_for_both_architectures():
    for arch in ("gru", "mlp"):
        m = RecurrentGate(hidden=16, arch=arch)
        logits, value, _ = m(_x())
        assert logits.shape == (2, 5, 2) and value.shape == (2, 5)
    assert _x().shape[-1] == OBS_DIM + EXTRA_DIM


def test_gru_has_memory_and_mlp_does_not():
    torch.manual_seed(0)
    a = _x(1, 4)
    b = a.clone()
    b[0, 0] = b[0, 0] + 0.4  # change only the FIRST step
    gru, mlp = RecurrentGate(hidden=16, arch="gru"), RecurrentGate(hidden=16, arch="mlp")
    assert not torch.allclose(gru(a)[0][0, 3], gru(b)[0][0, 3])  # last step differs (memory)
    assert torch.allclose(mlp(a)[0][0, 3], mlp(b)[0][0, 3])  # last step identical


def test_exhausted_budget_masks_system2():
    logits, _, _ = RecurrentGate(hidden=16)(_x(budget=0.0))
    assert (logits[..., 1] < -1e8).all() and (logits[..., 0] > -1e8).all()


def test_gae_matches_a_hand_computation():
    r, v = torch.tensor([[1.0, 1.0]]), torch.tensor([[0.5, 0.5]])
    adv, ret = compute_gae(r, v, gamma=1.0, lam=1.0)
    # t=1: delta=1-0.5=0.5 ; t=0: delta=1+0.5-0.5=1.0 -> adv0 = 1.0 + 0.5
    assert adv.tolist() == [[1.5, 0.5]] and ret.tolist() == [[2.0, 1.0]]


def test_tiny_ppo_run_is_finite_and_eval_works():
    cfg = _cfg()
    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(0))
    model = RecurrentGate(hidden=16)
    hist = train_gate(
        env,
        model,
        PPOConfig(updates=3, tasks_per_update=4, epochs=2, minibatch_tasks=2),
        trials_per_task=1,
        seed=0,
        log=lambda *_: None,
    )
    assert len(hist) == 3 and all(np.isfinite(h["mean_return_per_trial"]) for h in hist)
    res = evaluate_policy(env, "gate", [1, 2], None, model)
    assert 0.0 <= res["s2_rate"] <= 1.0 and np.isfinite(res["return"])


def test_multi_trial_tasks_carry_memory_without_error():
    cfg = _cfg()
    env = CycloneGatingEnv(cfg, rng=np.random.default_rng(0))
    hist = train_gate(
        env,
        RecurrentGate(hidden=16),
        PPOConfig(updates=1, tasks_per_update=2, epochs=1, minibatch_tasks=2),
        trials_per_task=2,
        seed=0,
        log=lambda *_: None,
    )
    assert len(hist) == 1


# --------------------------------------------------------------- stability fixes (section 25)
def test_stabilisers_are_off_by_default_so_old_results_reproduce():
    cfg = PPOConfig()
    assert cfg.value_norm is False and cfg.ent_coef_final is None
    assert cfg.target_kl is None and cfg.critic_warmup_updates == 0 and cfg.lr_final_frac == 1.0
    m = RecurrentGate(hidden=8)
    v = torch.tensor([3.0])
    assert torch.equal(m.to_return_units(v), v)  # identity until normalisation is switched on


def test_value_norm_fixes_the_critic_target_scale_from_the_first_batch():
    env = CycloneGatingEnv(_cfg(), rng=np.random.default_rng(0))
    model = RecurrentGate(hidden=16)
    train_gate(
        env,
        model,
        PPOConfig(updates=2, tasks_per_update=4, epochs=1, minibatch_tasks=2, value_norm=True),
        trials_per_task=1,
        seed=0,
        log=lambda *_: None,
    )
    assert bool(model.v_ready) and float(model.v_sd) > 0.0
    assert float(model.v_mu) != 0.0  # returns of a storm are far from zero
    # the constants travel with the weights
    clone = RecurrentGate(hidden=16)
    clone.load_state_dict(model.state_dict())
    assert float(clone.v_mu) == float(model.v_mu)


def test_critic_warmup_leaves_the_policy_untouched():
    env = CycloneGatingEnv(_cfg(), rng=np.random.default_rng(0))
    model = RecurrentGate(hidden=16)
    before = [p.detach().clone() for p in model.actor.parameters()]
    train_gate(
        env,
        model,
        PPOConfig(
            updates=2, tasks_per_update=4, epochs=1, minibatch_tasks=2, critic_warmup_updates=2
        ),
        trials_per_task=1,
        seed=0,
        log=lambda *_: None,
    )
    after = list(model.actor.parameters())
    # During critic warm-up only the value loss is applied, which does not touch the actor head,
    # so its own weights must be bit-for-bit unchanged.
    assert all(torch.equal(a, b) for a, b in zip(after, before, strict=True))


def test_policy_does_learn_once_the_warmup_is_over():
    env = CycloneGatingEnv(_cfg(), rng=np.random.default_rng(0))
    model = RecurrentGate(hidden=16)
    before = [p.detach().clone() for p in model.actor.parameters()]
    train_gate(
        env,
        model,
        PPOConfig(
            updates=2, tasks_per_update=4, epochs=1, minibatch_tasks=2, critic_warmup_updates=1
        ),
        trials_per_task=1,
        seed=0,
        log=lambda *_: None,
    )
    assert any(not torch.equal(a, b) for a, b in zip(model.actor.parameters(), before, strict=True))


def test_warm_start_copies_the_teacher_rule():
    from module3.train.warmstart import WarmStartConfig, behaviour_clone, collect_teacher_data

    env = CycloneGatingEnv(_cfg(), rng=np.random.default_rng(0))
    wcfg = WarmStartConfig(tasks=12, epochs=25)
    x, a = collect_teacher_data(env, wcfg, np.random.default_rng(1))
    assert x.shape[0] == 12 and a.shape == x.shape[:2]
    model = RecurrentGate(hidden=16)
    stats = behaviour_clone(model, x, a, wcfg, seed=0)
    assert stats["agreement_with_teacher"] > 0.8
