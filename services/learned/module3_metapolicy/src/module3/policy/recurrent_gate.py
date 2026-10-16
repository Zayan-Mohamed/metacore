"""RecurrentGate — the learned System1/System2 switch (actor + critic, with memory).

The network reads, at every step, the 14-d observation PLUS what happened last step:

    input_t = [ obs_t (14) | previous action (2, one-hot) | previous reward (1) | trial_done (1) ]

and passes it through a GRU (a small recurrent layer that keeps a memory vector between
steps). That is the RL^2 recipe (Duan et al. 2016): because the previous action and reward
are fed back in, the GRU's memory can infer *which storm it is in* from experience and adapt
its behaviour within the episode — "learning to reinforcement-learn" — while ordinary PPO
trains the weights.

Two heads read the memory:
    actor  -> 2 logits  (System 1 / System 2)     [the policy]
    critic -> 1 number  (expected future reward)  [the baseline PPO needs]

`arch="mlp"` swaps the GRU for a memoryless layer with the SAME inputs — the ablation that
shows whether memory actually helps.

Budget masking: when `budget_remaining` (obs slot 10) is 0 the System 2 logit is set to a
huge negative number, so the policy cannot even propose it. The env enforces the same rule.
"""

from __future__ import annotations

import torch
import torch.nn as nn

OBS_DIM = 15
BUDGET_SLOT = 10
EXTRA_DIM = 4  # prev action one-hot (2) + prev reward (1) + trial_done (1)
REWARD_SCALE = 2.0  # keeps the fed-back reward roughly in [-1, 1]


def build_inputs(obs, prev_action, prev_reward, trial_done) -> torch.Tensor:
    """Stack the per-step inputs. All args are tensors of shape (..., ) matching `obs[..., 0]`.

    obs: (..., 14); prev_action: long (...,); prev_reward / trial_done: float (...,).
    Returns (..., 18).
    """
    a = torch.nn.functional.one_hot(prev_action.long(), 2).float()
    return torch.cat(
        [obs, a, (prev_reward / REWARD_SCALE).unsqueeze(-1), trial_done.unsqueeze(-1)], dim=-1
    )


class RecurrentGate(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64, arch: str = "gru") -> None:
        super().__init__()
        if arch not in ("gru", "mlp"):
            raise ValueError(f"arch must be 'gru' or 'mlp', got {arch!r}")
        self.arch, self.hidden, self.obs_dim = arch, hidden, obs_dim
        self.embed = nn.Sequential(nn.Linear(obs_dim + EXTRA_DIM, hidden), nn.Tanh())
        if arch == "gru":
            self.core = nn.GRU(hidden, hidden, batch_first=True)
        else:
            self.core = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, 2)
        self.critic = nn.Linear(hidden, 1)
        # Value normalisation (optional, see PPOConfig.value_norm): the critic predicts
        # (return - v_mu) / v_sd so its loss has unit scale. Stored with the weights.
        self.register_buffer("v_mu", torch.zeros(()))
        self.register_buffer("v_sd", torch.ones(()))
        self.register_buffer("v_ready", torch.zeros((), dtype=torch.bool))
        # Start slightly biased toward System 1 so early exploration does not burn the budget.
        with torch.no_grad():
            self.actor.bias.copy_(torch.tensor([0.3, -0.3]))

    def to_return_units(self, value: torch.Tensor) -> torch.Tensor:
        """Convert a critic output back to real reward units."""
        return value * self.v_sd + self.v_mu

    def initial_state(self, batch: int = 1) -> torch.Tensor:
        """Empty memory (zeros). Reset at the start of every new storm/task."""
        return torch.zeros(1, batch, self.hidden)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """x: (B, T, obs_dim+4). Returns logits (B,T,2), value (B,T), new memory h."""
        z = self.embed(x)
        if self.arch == "gru":
            z, h = self.core(z, h if h is not None else self.initial_state(x.shape[0]))
        else:
            z = self.core(z)
        logits = self.actor(z)
        exhausted = x[..., BUDGET_SLOT] <= 1e-6
        logits = torch.where(
            exhausted.unsqueeze(-1) & torch.tensor([False, True]),
            torch.full_like(logits, -1e9),
            logits,
        )
        return logits, self.critic(z).squeeze(-1), h
