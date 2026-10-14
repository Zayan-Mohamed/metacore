"""System 1 for the Meta-RL pipeline — the existing reactive rule, unchanged.

Thin wrapper around `system1.system1_action` (flat file) so the pipeline and the old
dashboard demo share ONE System 1. It sheds load in proportion to node vulnerability,
lowest-priority tiers first, never tier 1, and sheds less when telemetry is lost.
It does not look at which lines are tripped — that is the whole point of the contrast
with System 2.
"""

from __future__ import annotations

from module3._flat import ensure_flat_on_path

ensure_flat_on_path()
from system1 import system1_action  # noqa: E402  (needs the flat dir on sys.path first)


def reactive_plan(context: dict, trigger_reason: str) -> dict:
    """Input: M1-style context dict + M2 trigger_reason. Output: ProposedControlAction dict."""
    return system1_action({**context, "trigger_reason": trigger_reason})
