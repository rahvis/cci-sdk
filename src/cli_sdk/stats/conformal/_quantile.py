"""The finite-sample conformal quantile shared by every split-conformal method.

See research.md, section 2.1, for the derivation. Given ``n`` calibration
nonconformity scores and a target miscoverage ``alpha``, the split
conformal threshold is the

    ceil((n + 1) * (1 - alpha)) / n

empirical quantile of the calibration scores. If that order statistic does
not exist (alpha is too small for n, i.e. n < 1/alpha - 1), the threshold
is +infinity and every candidate is included in the prediction set — the
"trivial" but still valid set.
"""

from __future__ import annotations

import math

import numpy as np


def minimum_calibration_size(alpha: float) -> int:
    """The smallest n for which a non-trivial conformal set exists.

    n >= (1 - alpha) / alpha, i.e. n >= 1/alpha - 1. Returns the ceiling.
    """
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return math.ceil((1 - alpha) / alpha)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The split-conformal threshold q_hat for nonconformity scores.

    Parameters
    ----------
    scores:
        1-D array of n calibration nonconformity scores. Higher means
        "less conforming" / "less plausible."
    alpha:
        Target miscoverage rate, in (0, 1).

    Returns
    -------
    q_hat such that C(x) = {y : s(x, y) <= q_hat} satisfies
    1 - alpha <= P(y in C(x)) <= 1 - alpha + 1/(n+1) under exchangeability.
    Returns +inf if n is too small to support alpha (see
    ``minimum_calibration_size``).
    """
    scores = np.asarray(scores, dtype=float).ravel()
    n = scores.shape[0]
    if n == 0:
        raise ValueError("cannot calibrate on an empty calibration set")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    level = math.ceil((n + 1) * (1 - alpha))
    if level > n:
        return math.inf
    # 1-indexed order statistic -> 0-indexed position `level - 1`.
    return float(np.partition(scores, level - 1)[level - 1])


def coverage_confidence_interval(
    n: int, alpha: float, confidence: float = 0.90
) -> tuple[float, float]:
    """Realized-coverage interval implied by a fixed calibration set of size n.

    Coverage given a *fixed* calibration set follows Beta(n + 1 - l, l)
    with l = floor((n + 1) * alpha) (research.md, section 10.1 / Vovk
    2012). This returns the two-sided interval at the requested
    confidence level, computed with the regularized incomplete beta
    function's inverse via a numerically stable bisection (no SciPy
    dependency).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    l = math.floor((n + 1) * alpha)
    a = n + 1 - l
    b = max(l, 1e-9)
    lower_tail = (1 - confidence) / 2
    upper_tail = 1 - lower_tail
    return (_beta_ppf(lower_tail, a, b), _beta_ppf(upper_tail, a, b))


def _beta_ppf(q: float, a: float, b: float, tol: float = 1e-10) -> float:
    """Inverse CDF of Beta(a, b) via bisection on the regularized incomplete beta function."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def _beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via a continued fraction (Numerical Recipes)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - ln_beta) / a
    if x < (a + 1) / (a + b + 2):
        return front * _beta_continued_fraction(x, a, b)
    return 1.0 - (
        math.exp(math.log(1 - x) * b + math.log(x) * a - ln_beta) / b
    ) * _beta_continued_fraction(1 - x, b, a)


def _beta_continued_fraction(x: float, a: float, b: float, max_iter: int = 300) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes ``betacf``)."""
    eps = 1e-15      # convergence tolerance
    fpmin = 1e-300   # guard against division by (near) zero
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    d = fpmin if abs(d) < fpmin else d
    d = 1.0 / d
    h = d
    for m in range(1, max_iter):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = fpmin if abs(d) < fpmin else d
        c = 1.0 + aa / c
        c = fpmin if abs(c) < fpmin else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h
