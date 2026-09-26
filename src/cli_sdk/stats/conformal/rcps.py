"""RCPS: risk-controlling prediction sets (Bates, Angelopoulos, Lei, Malik & Jordan, JACM 2021).

Gives a high-probability guarantee: P(risk(lambda_hat) <= alpha) >= 1-delta,
strictly stronger than CRC's in-expectation guarantee, using a
concentration bound on the empirical risk. Powers the ``Gate`` primitive's
``"risk_high_probability"`` guarantee mode. See research.md [P075].
"""

from __future__ import annotations

import math

import numpy as np


def _hoeffding_bentkus_p_value(risk_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for testing H0: R(lambda) > alpha.

    ``min(exp(-n * h1(min(risk_hat, alpha), alpha)), e * P(Bin(n, alpha) <= ceil(n * risk_hat)))``
    with ``h1`` the Bernoulli KL divergence (Bates et al., JACM 2021). Tighter
    than a plain Hoeffding bound; used throughout the RCPS / Learn-then-Test
    family (research.md, section 2.3). Valid for losses in [0, 1].
    """
    if n <= 0:
        return 1.0
    if risk_hat >= alpha:
        return 1.0
    hoeffding = math.exp(-n * _bernoulli_kl(risk_hat, alpha))
    k = math.ceil(n * risk_hat - 1e-12)
    bentkus = math.e * _binomial_cdf(k, n, alpha)
    return min(1.0, hoeffding, bentkus)


def _bernoulli_kl(a: float, b: float) -> float:
    """KL(Bernoulli(a) || Bernoulli(b)) for 0 <= a < b < 1."""
    a = min(max(a, 0.0), 1.0)
    term0 = a * math.log(a / b) if a > 0 else 0.0
    term1 = (1 - a) * math.log((1 - a) / (1 - b)) if a < 1 else 0.0
    return term0 + term1


def _binomial_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0.0
    # log-space binomial pmf accumulation for numerical stability.
    log_p, log_1mp = math.log(p), math.log(1 - p)
    for i in range(0, k + 1):
        log_pmf = (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_1mp
        )
        total += math.exp(log_pmf)
    return min(1.0, total)


def calibrate(
    cal_scores: np.ndarray,
    cal_losses: np.ndarray,
    alpha: float,
    delta: float,
    lambda_grid: np.ndarray | None = None,
) -> float:
    """Calibrate the RCPS threshold via fixed-sequence testing.

    Searches lambda from most to least conservative and returns the first
    (least conservative) threshold whose Hoeffding-Bentkus upper
    confidence bound on the risk is <= alpha at confidence 1 - delta.
    Fixed-sequence testing controls the family-wise error rate across the
    grid without a Bonferroni correction, which is why the search order
    matters: it must run from conservative to permissive and stop at the
    first failure.
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_losses = np.asarray(cal_losses, dtype=float)
    n = cal_scores.shape[0]
    if lambda_grid is None:
        lambda_grid = np.concatenate([[np.inf], np.sort(np.unique(cal_scores))[::-1]])
    else:
        lambda_grid = np.sort(np.asarray(lambda_grid))[::-1]

    best = float(lambda_grid[0])
    for lam in lambda_grid:
        accepted = cal_scores >= lam
        if not accepted.any():
            # Accepting nothing incurs zero loss: trivially safe, keep going.
            best = float(lam)
            continue
        # Joint risk over all n points (loss counts only when accepted), the
        # quantity RCPS controls; it is monotone in lambda, so fixed-sequence
        # testing from conservative to permissive controls the FWER.
        risk_hat = float(np.mean(cal_losses * accepted))
        p_value = _hoeffding_bentkus_p_value(risk_hat, n, alpha)
        if p_value <= delta:
            best = float(lam)
        else:
            break
    return best
