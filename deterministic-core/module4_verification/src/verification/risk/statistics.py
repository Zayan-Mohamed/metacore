"""From a count of failing scenarios to a verdict (Step 5 of the method).

k of the N sampled scenarios broke a limit. The exact Clopper-Pearson bounds (Clopper & Pearson
1934) say where the true failure rate p can be, with confidence 1 - delta, using nothing but the
binomial distribution:

    upper(k, N):  the p with  P(X <= k | N, p) = delta      (one-sided, k < N)
    lower(k, N):  the p with  P(X >= k | N, p) = delta      (one-sided, k > 0)

Against the risk target eps:

    APPROVE     upper <= eps                  ("even the top of the range is below eps")
    REJECT      lower >  eps                  ("even the bottom is above eps")
    UNVERIFIED  otherwise                     (the range straddles eps)

N = 688 is the smallest N at which k = 0 certifies p <= 1 % at 99.9 %: upper = 1 - delta^(1/N).
This is the scenario-approach sample size (Campi & Garatti 2008) and the statistical model
checking test (Legay et al. 2010).

Early stopping runs the scenarios in batches. Approval is judged once, at the full N, so it is a
single test at level delta and looking early cannot inflate a false approval. Rejection may be
declared at any batch boundary, so it splits delta over the looks (Bonferroni): a clearly bad
action is rejected after the first batch without weakening the guarantee.

Pure Python (math.lgamma): no scipy, nothing beyond the standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EPSILON = 0.01  # risk target: fewer than 1 in 100 realistic next-6-hour conditions break a limit
CONFIDENCE = 0.999
BATCH = 100


def required_scenarios(eps: float = EPSILON, confidence: float = CONFIDENCE) -> int:
    """Smallest N such that zero failures certifies p <= eps: ceil(ln(delta) / ln(1 - eps))."""
    return math.ceil(math.log(1.0 - confidence) / math.log(1.0 - eps))


def _log_pmf(i: int, n: int, p: float) -> float:
    return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    return min(1.0, sum(math.exp(_log_pmf(i, n, p)) for i in range(k + 1)))


def _bisect(f, lo: float = 0.0, hi: float = 1.0, iters: int = 200) -> float:
    """Root of a function that is decreasing in p (f(lo) > 0 > f(hi))."""
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cp_upper(k: int, n: int, confidence: float = CONFIDENCE) -> float:
    """One-sided Clopper-Pearson upper bound on p after k failures in n trials."""
    if k >= n:
        return 1.0
    delta = 1.0 - confidence
    if k == 0:
        return 1.0 - delta ** (1.0 / n)  # closed form
    return _bisect(lambda p: binom_cdf(k, n, p) - delta)


def cp_lower(k: int, n: int, confidence: float = CONFIDENCE) -> float:
    """One-sided Clopper-Pearson lower bound on p after k failures in n trials."""
    if k <= 0:
        return 0.0
    delta = 1.0 - confidence
    if k == n:
        return delta ** (1.0 / n)
    # P(X >= k | p) = 1 - cdf(k - 1) increases in p; find where it equals delta.
    return _bisect(lambda p: delta - (1.0 - binom_cdf(k - 1, n, p)))


APPROVE, REJECT, UNVERIFIED = "APPROVE", "REJECT", "UNVERIFIED"


def decide(k: int, n: int, eps: float = EPSILON, confidence: float = CONFIDENCE,
           looks: int = 1) -> str:
    """Three-way verdict from k failures in n scenarios.

    ``looks`` is how many times a REJECT could have been declared (Bonferroni on the lower bound).
    """
    if cp_upper(k, n, confidence) <= eps:
        return APPROVE
    if cp_lower(k, n, 1.0 - (1.0 - confidence) / looks) > eps:
        return REJECT
    return UNVERIFIED


@dataclass
class SequentialTest:
    """Batches of BATCH scenarios up to N; stops early only to REJECT."""

    n_max: int = required_scenarios()
    eps: float = EPSILON
    confidence: float = CONFIDENCE
    batch: int = BATCH
    early: bool = True  # False runs all N regardless (to measure what early stopping saves)
    n: int = 0
    k: int = 0
    stopped_early: bool = False

    @property
    def looks(self) -> int:
        return math.ceil(self.n_max / self.batch)

    def boundaries(self) -> list[int]:
        """Scenario counts at which the test looks: 100, 200, ..., N."""
        return [min(self.n_max, b) for b in range(self.batch, self.n_max + self.batch,
                                                  self.batch)]

    def record(self, failures: int, trials: int) -> bool:
        """Adds one batch; returns True when the test is finished (early REJECT or N reached)."""
        self.k += failures
        self.n += trials
        if self.n >= self.n_max:
            return True
        if self.early and decide(self.k, self.n, self.eps, self.confidence,
                                 self.looks) == REJECT:
            self.stopped_early = True
            return True
        return False

    def verdict(self) -> str:
        if self.n < self.n_max:
            # Only an early REJECT ends the test before N; approval needs the full sample.
            return REJECT if self.stopped_early else UNVERIFIED
        return decide(self.k, self.n, self.eps, self.confidence, self.looks)

    def bounds(self) -> tuple[float, float]:
        if self.n == 0:
            return 0.0, 1.0
        return cp_lower(self.k, self.n, self.confidence), cp_upper(self.k, self.n,
                                                                    self.confidence)
