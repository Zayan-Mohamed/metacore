"""LatencyGuard - keeps System 2 from blowing the real-time deadline.

Why it exists: the control loop has a hard budget (50 ms for the whole decision, the same figure
M4's latency test uses). System 1 is nearly free; System 2 is the path that can get slow (today a
greedy planner of ~0.02 ms, but the real optimiser / OpenDSS-in-the-loop version may take much
longer - see docs/module3.md section 27). A late plan is a wrong plan in real time, so:

    1. DEADLINE   - if System 2's plan takes longer than `s2_deadline_ms`, its answer is thrown
                    away and the System 1 plan is used for this step.
    2. BREAKER    - after `trip_after` overruns in a row the guard stops calling System 2 at all
                    for `cooldown_steps` steps (System 1 only), then tries again. This is the
                    part that protects the *next* steps; a Python call cannot be interrupted
                    half-way, so the guard bounds the damage instead of pre-empting it.
    3. BUDGET     - a System 2 request is also refused up front if the time already used this
                    step leaves less than `s2_deadline_ms` before `step_deadline_ms`.

    guard = LatencyGuard(GuardConfig())
    plan, event = guard.plan(requested=1, s1_fn=..., s2_fn=..., elapsed_ms=gate_ms + verify_ms)
    event.path        -> "S1" or "S2"   (what was actually used)
    event.reason      -> "ok" | "gate_chose_s1" | "breaker_open" | "no_time_left" | "s2_overrun"

The clock is injectable (`clock`, seconds) so tests do not depend on real timing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardConfig:
    step_deadline_ms: float = 50.0  # whole decision: gate + plan + verification
    s2_deadline_ms: float = 5.0  # System 2 planning alone
    trip_after: int = 3  # consecutive overruns that open the breaker
    cooldown_steps: int = 10  # steps System 2 stays off once the breaker opens

    @classmethod
    def from_dict(cls, d: dict | None) -> GuardConfig:
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class GuardEvent:
    requested: int  # what the gate asked for (0 = S1, 1 = S2)
    path: str  # "S1" or "S2" actually used
    reason: str
    plan_ms: float  # time spent in the plan function(s) this step
    breaker_open: bool  # breaker state AFTER this step


class LatencyGuard:
    def __init__(self, cfg: GuardConfig | None = None, clock: Callable[[], float] | None = None):
        self.cfg = cfg or GuardConfig()
        self._clock = clock or time.perf_counter
        self._overruns = 0
        self._cooldown = 0
        self.events: list[GuardEvent] = []

    @property
    def breaker_open(self) -> bool:
        return self._cooldown > 0

    def plan(
        self,
        requested: int,
        s1_fn: Callable[[], dict],
        s2_fn: Callable[[], dict],
        elapsed_ms: float = 0.0,
    ) -> tuple[dict, GuardEvent]:
        """Produce this step's plan within the deadline. `elapsed_ms` = time already spent."""
        cfg = self.cfg
        was_open = self.breaker_open
        if was_open:
            self._cooldown -= 1  # one cooldown step consumed
        t0 = self._clock()

        def done(plan, path, reason):
            ms = (self._clock() - t0) * 1000.0
            ev = GuardEvent(requested, path, reason, ms, self.breaker_open)
            self.events.append(ev)
            return plan, ev

        if not requested:
            return done(s1_fn(), "S1", "gate_chose_s1")
        if was_open:
            return done(s1_fn(), "S1", "breaker_open")
        if elapsed_ms + cfg.s2_deadline_ms > cfg.step_deadline_ms:
            return done(s1_fn(), "S1", "no_time_left")

        t_s2 = self._clock()
        s2_plan = s2_fn()
        s2_ms = (self._clock() - t_s2) * 1000.0
        if s2_ms > cfg.s2_deadline_ms:
            self._overruns += 1
            if self._overruns >= cfg.trip_after:
                self._cooldown = cfg.cooldown_steps
                self._overruns = 0
            return done(s1_fn(), "S1", "s2_overrun")
        self._overruns = 0
        return done(s2_plan, "S2", "ok")

    def summary(self) -> dict:
        """Counts per reason plus the slowest plan step - what the PP1 latency table reports."""
        reasons: dict[str, int] = {}
        for e in self.events:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1
        return {
            "steps": len(self.events),
            "reasons": reasons,
            "max_plan_ms": max((e.plan_ms for e in self.events), default=0.0),
        }
