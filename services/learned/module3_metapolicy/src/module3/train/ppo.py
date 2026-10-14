"""PPO (Proximal Policy Optimization) for the recurrent gate.

How one training iteration works, in plain words:
  1. COLLECT   — run the current gate through `tasks_per_update` fresh random cyclones,
                 recording what it saw, what it chose, what reward came back, and how
                 likely it thought each choice was.
  2. SCORE     — turn rewards into "advantages" with GAE (Generalized Advantage Estimation):
                 for every step, was the outcome better or worse than the critic expected?
  3. IMPROVE   — nudge the gate to make good choices likelier and bad ones less likely, but
                 only a little: PPO clips the change (`clip`), so one noisy batch cannot
                 wreck what was already learned. This is repeated for a few `epochs` over
                 the same data.

Each storm is one sequence fed whole through the GRU (memory starts empty at the storm's
first step), which is why minibatches are made of whole storms, not shuffled single steps.

References: Schulman et al. 2017 (PPO), Schulman et al. 2015 (GAE). See docs/module3-research.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from module3.env.cyclone_env import CycloneGatingEnv
from module3.policy.recurrent_gate import RecurrentGate, build_inputs
from module3.scenario.cyclone import sample_params


@dataclass
class PPOConfig:
    updates: int = 120
    tasks_per_update: int = 16
    epochs: int = 4
    minibatch_tasks: int = 8
    lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    @classmethod
    def from_dict(cls, d: dict | None) -> PPOConfig:
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


@torch.no_grad()
def collect_tasks(
    env: CycloneGatingEnv,
    model: RecurrentGate,
    n_tasks: int,
    trials_per_task: int,
    rng: np.random.Generator,
) -> dict:
    """Run the gate through `n_tasks` storms. Returns tensors shaped (n_tasks, L, ...).

    A *task* is one storm (its parameters). With `trials_per_task` > 1 the SAME storm
    parameters are replayed (with fresh randomness) and the GRU memory is carried across
    those trials — the RL^2 'multiple trials per task' setting.
    """
    inputs, actions, logps, values, rewards = [], [], [], [], []
    for _ in range(n_tasks):
        params = sample_params(rng, env._scenario_cfg)
        h = model.initial_state(1)
        prev_a, prev_r, done_flag = 0, 0.0, 0.0
        ti, ta, tl, tv, tr = [], [], [], [], []
        for _trial in range(trials_per_task):
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)), options={"cyclone": params})
            done = False
            while not done:
                x = build_inputs(
                    torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1),
                    torch.tensor([[prev_a]]),
                    torch.tensor([[prev_r]], dtype=torch.float32),
                    torch.tensor([[done_flag]], dtype=torch.float32),
                )
                logits, value, h = model(x, h)
                dist = torch.distributions.Categorical(logits=logits[0, 0])
                a = dist.sample()
                obs_next, r, _, done, _ = env.step(int(a.item()))
                ti.append(x[0, 0])
                ta.append(a)
                tl.append(dist.log_prob(a))
                tv.append(value[0, 0])
                tr.append(float(r))
                prev_a, prev_r, done_flag = int(a.item()), float(r), 0.0
                obs = obs_next
            done_flag = 1.0  # tell the next trial the previous one just ended
        inputs.append(torch.stack(ti))
        actions.append(torch.stack(ta))
        logps.append(torch.stack(tl))
        values.append(torch.stack(tv))
        rewards.append(torch.tensor(tr))
    return {
        "x": torch.stack(inputs),
        "a": torch.stack(actions),
        "logp": torch.stack(logps),
        "v": torch.stack(values),
        "r": torch.stack(rewards),
    }


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float):
    """GAE advantages and return targets. rewards/values: (N, L); the sequence ends at L."""
    n, length = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(n)
    for t in reversed(range(length)):
        next_v = values[:, t + 1] if t + 1 < length else torch.zeros(n)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv, adv + values


def ppo_update(
    model: RecurrentGate,
    opt: torch.optim.Optimizer,
    batch: dict,
    cfg: PPOConfig,
    rng: np.random.Generator,
) -> dict:
    """Several epochs of clipped-PPO updates on one batch of storms. Returns diagnostics."""
    adv, ret = compute_gae(batch["r"], batch["v"], cfg.gamma, cfg.gae_lambda)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = batch["x"].shape[0]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "kl": 0.0}
    n_mb = 0
    for _ in range(cfg.epochs):
        order = rng.permutation(n)
        for s in range(0, n, cfg.minibatch_tasks):
            idx = torch.as_tensor(order[s : s + cfg.minibatch_tasks])
            logits, value, _ = model(batch["x"][idx])
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(batch["a"][idx])
            log_ratio = new_logp - batch["logp"][idx]
            ratio = log_ratio.exp()
            a = adv[idx]
            pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
            vloss = 0.5 * (value - ret[idx]).pow(2).mean()
            ent = dist.entropy().mean()
            loss = pg + cfg.vf_coef * vloss - cfg.ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            stats["policy_loss"] += pg.item()
            stats["value_loss"] += vloss.item()
            stats["entropy"] += ent.item()
            stats["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
            stats["kl"] += (-log_ratio).mean().item()
            n_mb += 1
    return {k: v / max(n_mb, 1) for k, v in stats.items()}


def train_gate(
    env: CycloneGatingEnv,
    model: RecurrentGate,
    cfg: PPOConfig,
    trials_per_task: int,
    seed: int,
    log=print,
) -> list[dict]:
    """Full PPO training loop. Returns one log row per update."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    for u in range(cfg.updates):
        batch = collect_tasks(env, model, cfg.tasks_per_update, trials_per_task, rng)
        stats = ppo_update(model, opt, batch, cfg, rng)
        row = {
            "update": u + 1,
            "mean_return_per_trial": float(batch["r"].sum(dim=1).mean() / trials_per_task),
            "s2_rate": float(batch["a"].float().mean()),
            **stats,
        }
        history.append(row)
        if (u + 1) % max(1, cfg.updates // 10) == 0 or u == 0:
            ret = row["mean_return_per_trial"]
            log(
                f"  update {u + 1:4d}/{cfg.updates}  return/trial={ret:7.2f}"
                f"  S2 rate={row['s2_rate']:.2f}  entropy={row['entropy']:.3f}"
            )
    return history
