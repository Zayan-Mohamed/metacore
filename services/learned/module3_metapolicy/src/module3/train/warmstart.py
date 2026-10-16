"""Behaviour-cloning warm start — teach the gate a sensible starting policy before PPO.

Why: from random weights the gate sees a noisy reward and in some seeds slides into a fixed
habit (always System 2, or never System 2) that PPO then cannot escape. Starting it from a
reasonable rule removes that failure.

How: run a simple rule (default: `breaker_rule` — escalate when an outage is visible and the
storm is at least 'elevated') on random storms, record what the gate would have seen and what
the rule chose, then train the gate's action head to copy the rule (supervised learning).
Soft labels (`smoothing`) stop the copy from becoming over-confident, which keeps the gate
exploring when PPO takes over. The critic is untouched here — PPO's critic warm-up handles it.

Honesty note: a warm-started gate starts as a copy of the rule, so its final score must be
reported next to BOTH the rule and the warm-start-only snapshot (`gate_bc`). The part PPO adds
is the difference between `gate` and `gate_bc`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from module3.env.cyclone_env import CycloneGatingEnv
from module3.policy.recurrent_gate import BUDGET_SLOT, RecurrentGate, build_inputs
from module3.scenario.cyclone import sample_params
from module3.train.evaluate import _baseline_action


@dataclass
class WarmStartConfig:
    teacher: str = "breaker_rule"
    tasks: int = 64  # storms to imitate
    epochs: int = 30
    lr: float = 1e-3
    smoothing: float = 0.1  # label smoothing: teacher's choice gets 1 - smoothing/2
    batch_tasks: int = 16
    thr_u: float = 0.4

    @classmethod
    def from_dict(cls, d: dict | None) -> WarmStartConfig:
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


def collect_teacher_data(
    env: CycloneGatingEnv, cfg: WarmStartConfig, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Play the teacher rule on random storms. Returns gate inputs (N, L, D) and actions (N, L)."""
    xs, acts = [], []
    for _ in range(cfg.tasks):
        params = sample_params(rng, env._scenario_cfg)
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)), options={"cyclone": params})
        prev_a, prev_r, truth = 0, 0.0, 0
        tx, ta = [], []
        done = False
        while not done:
            x = build_inputs(
                torch.as_tensor(obs, dtype=torch.float32).view(1, 1, -1),
                torch.tensor([[prev_a]]),
                torch.tensor([[prev_r]], dtype=torch.float32),
                torch.tensor([[0.0]]),
            )
            a = _baseline_action(cfg.teacher, obs, truth, cfg.thr_u)
            obs, r, _, done, info = env.step(a)
            tx.append(x[0, 0])
            ta.append(a)
            prev_a, prev_r = a, float(r)
            truth = info["n_tripped_truth"] + int(info["grid_lost_truth"])
        xs.append(torch.stack(tx))
        acts.append(torch.tensor(ta))
    return torch.stack(xs), torch.stack(acts)


def behaviour_clone(
    model: RecurrentGate, x: torch.Tensor, a: torch.Tensor, cfg: WarmStartConfig, seed: int
) -> dict:
    """Supervised training of the action head on (input, teacher action) pairs."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    can_s2 = x[..., BUDGET_SLOT] > 1e-6  # where System 2 is possible at all
    target = torch.nn.functional.one_hot(a, 2).float() * (1 - cfg.smoothing) + cfg.smoothing / 2
    loss_v = acc = 0.0
    for _ in range(cfg.epochs):
        order = torch.randperm(x.shape[0], generator=gen)
        for s in range(0, x.shape[0], cfg.batch_tasks):
            idx = order[s : s + cfg.batch_tasks]
            logits, _, _ = model(x[idx])
            logp = torch.log_softmax(logits, dim=-1)
            # Where S2 is masked out the label is forced to S1; train only where there is a choice.
            per_step = -(target[idx] * logp).sum(-1)
            m = can_s2[idx].float()
            loss = (per_step * m).sum() / m.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_v = loss.item()
    with torch.no_grad():
        logits, _, _ = model(x)
        acc = float(((logits.argmax(-1) == a) & can_s2).sum() / can_s2.sum().clamp(min=1))
    return {"final_loss": loss_v, "agreement_with_teacher": acc}
