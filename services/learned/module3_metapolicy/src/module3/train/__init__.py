"""Meta-RL training: PPO trainer, evaluation against baselines, and the end-to-end pipeline."""

from .evaluate import BASELINES, evaluate_policy, run_episode
from .ppo import PPOConfig, train_gate

__all__ = ["BASELINES", "PPOConfig", "evaluate_policy", "run_episode", "train_gate"]
