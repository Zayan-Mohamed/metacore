"""LatencyGuard: deadline, circuit breaker and time-budget rules, with a fake clock."""

from __future__ import annotations

from module3.serve import GuardConfig, LatencyGuard


class FakeClock:
    """Time only moves when a plan function 'takes' time, so the tests are exact."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def spend_ms(self, ms: float) -> None:
        self.now += ms / 1000.0


def _guard(**kw):
    clock = FakeClock()
    cfg = GuardConfig(**{"s2_deadline_ms": 5.0, "trip_after": 3, "cooldown_steps": 4, **kw})
    return LatencyGuard(cfg, clock), clock


def _fns(clock, s2_ms):
    def s1():
        clock.spend_ms(0.01)
        return {"who": "s1"}

    def s2():
        clock.spend_ms(s2_ms)
        return {"who": "s2"}

    return s1, s2


def test_fast_system2_is_used():
    g, clock = _guard()
    s1, s2 = _fns(clock, 1.0)
    plan, ev = g.plan(1, s1, s2)
    assert plan == {"who": "s2"} and ev.path == "S2" and ev.reason == "ok"


def test_gate_choosing_system1_never_calls_system2():
    g, clock = _guard()
    called = []

    def s2():
        called.append(1)
        return {"who": "s2"}

    plan, ev = g.plan(0, lambda: {"who": "s1"}, s2)
    assert plan == {"who": "s1"} and not called and ev.reason == "gate_chose_s1"


def test_slow_system2_answer_is_discarded_for_system1():
    g, clock = _guard()
    s1, s2 = _fns(clock, 9.0)  # over the 5 ms deadline
    plan, ev = g.plan(1, s1, s2)
    assert plan == {"who": "s1"} and ev.path == "S1" and ev.reason == "s2_overrun"


def test_breaker_opens_after_consecutive_overruns_and_then_recovers():
    g, clock = _guard()
    s1, slow = _fns(clock, 9.0)
    _, fast = _fns(clock, 1.0)
    for _ in range(3):
        g.plan(1, s1, slow)
    assert g.breaker_open
    calls = []

    def probe():
        calls.append(1)
        return fast()

    reasons = []
    for _ in range(4):  # cooldown_steps = 4 -> System 2 is not even tried
        _, ev = g.plan(1, s1, probe)
        reasons.append(ev.reason)
    assert reasons == ["breaker_open"] * 4 and not calls
    assert not g.breaker_open
    _, ev = g.plan(1, s1, probe)  # cooldown over: System 2 is tried again
    assert calls and ev.path == "S2"


def test_a_fast_step_resets_the_overrun_count():
    g, clock = _guard()
    s1, slow = _fns(clock, 9.0)
    _, fast = _fns(clock, 1.0)
    g.plan(1, s1, slow)
    g.plan(1, s1, slow)
    g.plan(1, s1, fast)  # streak broken
    g.plan(1, s1, slow)
    g.plan(1, s1, slow)
    assert not g.breaker_open  # never 3 in a row


def test_system2_refused_when_not_enough_time_is_left_in_the_step():
    g, clock = _guard(step_deadline_ms=50.0, s2_deadline_ms=5.0)
    s1, s2 = _fns(clock, 1.0)
    plan, ev = g.plan(1, s1, s2, elapsed_ms=46.0)  # 46 + 5 > 50
    assert plan == {"who": "s1"} and ev.reason == "no_time_left"
    plan, ev = g.plan(1, s1, s2, elapsed_ms=44.0)  # 44 + 5 <= 50
    assert ev.path == "S2"


def test_summary_counts_reasons():
    g, clock = _guard()
    s1, s2 = _fns(clock, 1.0)
    g.plan(0, s1, s2)
    g.plan(1, s1, s2)
    s = g.summary()
    assert s["steps"] == 2 and s["reasons"] == {"gate_chose_s1": 1, "ok": 1}
