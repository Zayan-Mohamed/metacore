"""Learned gate policy (recurrent actor-critic)."""

from .recurrent_gate import EXTRA_DIM, OBS_DIM, RecurrentGate, build_inputs

__all__ = ["EXTRA_DIM", "OBS_DIM", "RecurrentGate", "build_inputs"]
