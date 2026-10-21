"""Clopper-Pearson bounds, the three-way verdict and early stopping (C8)."""

from __future__ import annotations

import pytest
from verification.risk.statistics import (
    APPROVE,
    REJECT,
    UNVERIFIED,
    SequentialTest,
    binom_cdf,
    cp_lower,
    cp_upper,
    decide,
    required_scenarios,
)


def test_sample_sizes_match_the_published_zero_failure_tables() -> None:
    assert required_scenarios(0.01, 0.999) == 688
    assert required_scenarios(0.05, 0.95) == 59
    assert cp_upper(0, 688) < 0.01 < cp_upper(0, 687)
    assert cp_upper(0, 59, 0.95) < 0.05 < cp_upper(0, 58, 0.95)


# The four worked rows of the step-by-step guide (Step 5), as computed there.
@pytest.mark.parametrize(
    ("k", "n", "lo_pct", "hi_pct", "verdict"),
    [
        (0, 688, 0.0, 0.999, APPROVE),
        (3, 688, 0.028, 1.885, UNVERIFIED),
        (20, 688, 1.312, 5.457, REJECT),
        (20, 100, 9.451, 34.644, REJECT),
    ],
)
def test_worked_rows_from_the_guide(k: int, n: int, lo_pct: float, hi_pct: float,
                                    verdict: str) -> None:
    assert cp_lower(k, n) * 100 == pytest.approx(lo_pct, abs=5e-4)
    assert cp_upper(k, n) * 100 == pytest.approx(hi_pct, abs=5e-4)
    assert decide(k, n) == verdict


def test_bounds_hit_their_tail_probabilities() -> None:
    for k, n in [(1, 50), (5, 200), (17, 688)]:
        assert binom_cdf(k, n, cp_upper(k, n)) == pytest.approx(0.001, rel=1e-6)
        assert 1.0 - binom_cdf(k - 1, n, cp_lower(k, n)) == pytest.approx(0.001, rel=1e-6)


def test_edges() -> None:
    assert cp_lower(0, 10) == 0.0
    assert cp_upper(10, 10) == 1.0
    assert binom_cdf(-1, 10, 0.3) == 0.0 and binom_cdf(10, 10, 0.3) == 1.0


def _run(failures_per_batch: list[int]) -> SequentialTest:
    test = SequentialTest()
    for b, size in zip(failures_per_batch, test.boundaries(), strict=False):
        trials = size - test.n
        if test.record(b, trials):
            break
    return test


def test_batches_of_100_up_to_688() -> None:
    assert SequentialTest().boundaries() == [100, 200, 300, 400, 500, 600, 688]
    assert SequentialTest().looks == 7


def test_a_clean_run_approves_only_at_the_full_sample() -> None:
    test = SequentialTest()
    for size in test.boundaries()[:-1]:
        assert not test.record(0, size - test.n)
        assert test.verdict() == UNVERIFIED  # 600 clean scenarios are not enough
    assert test.record(0, 688 - test.n)
    assert test.verdict() == APPROVE and test.n == 688 and not test.stopped_early


def test_a_clearly_bad_action_is_rejected_after_the_first_batch() -> None:
    test = _run([20])
    assert test.stopped_early and test.n == 100
    assert test.verdict() == REJECT


def test_a_few_failures_run_to_the_end_and_straddle() -> None:
    test = _run([1, 0, 1, 0, 0, 1, 0])
    assert test.n == 688 and test.k == 3 and not test.stopped_early
    assert test.verdict() == UNVERIFIED


def test_early_rejection_pays_for_its_looks() -> None:
    """With 7 looks the REJECT bound runs at delta / 7, so it is harder to reject than once."""
    k, n = 12, 688
    assert cp_lower(k, n) > cp_lower(k, n, 1 - 0.001 / 7)
    assert decide(20, 688, looks=7) == REJECT
