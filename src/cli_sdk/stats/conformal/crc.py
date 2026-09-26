"""Conformal Risk Control (Angelopoulos, Bates, Candes, Jordan & Lei, ICLR 2024).

Generalizes split conformal prediction from coverage to arbitrary bounded
losses that are monotone non-increasing in a threshold lambda. Guarantees
E[loss(lambda_hat)] <= alpha in finite samples under exchangeability,
essentially for free over plain split CP. Powers the ``Gate`` primitive's
``"risk"`` guarantee mode. See research.md [P062].
"""

from __future__ import annotations

import numpy as np


def calibrate(
    cal_scores: np.ndarray,
    cal_losses: np.ndarray,
    alpha: float,
    lambda_grid: np.ndarray | None = None,
    loss_upper_bound: float = 1.0,
) -> float:
    """Calibrate the CRC threshold lambda_hat.

    Parameters
    ----------
    cal_scores:
        (n,) confidence scores; a decision accepts input i when
        ``cal_scores[i] >= lambda``.
    cal_losses:
        (n,) the loss incurred *if accepted* for each calibration point
        (e.g. 1 if the accepted answer was wrong, 0 otherwise). CRC
        assumes this loss is bounded above by ``loss_upper_bound`` and
        that the realized risk R(lambda) = mean(loss[i] * accept(i, lambda))
        is non-increasing as lambda grows (raising the bar only ever
        removes accepted points, so this holds for any 0/1-style loss
        gated by acceptance).
    alpha:
        Target risk level.
    lambda_grid:
        Candidate thresholds to search, most conservative first. Defaults
        to the sorted distinct calibration scores plus +inf (never
        accept).

    Returns
    -------
    lambda_hat = inf{lambda : (n/(n+1)) * R_hat(lambda) + B/(n+1) <= alpha},
    the least conservative threshold that satisfies the CRC bound, where
    B is ``loss_upper_bound``.
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_losses = np.asarray(cal_losses, dtype=float)
    n = cal_scores.shape[0]
    if lambda_grid is None:
        lambda_grid = np.concatenate([np.sort(np.unique(cal_scores)), [np.inf]])
    lambda_grid = np.sort(lambda_grid)

    for lam in lambda_grid:
        accepted = cal_scores >= lam
        # Joint risk: the loss counts only when the point is accepted, and the
        # average runs over all n points. This is what makes R(lambda)
        # non-increasing in lambda, which the CRC guarantee requires.
        risk_hat = float(np.mean(cal_losses * accepted))
        bound = (n / (n + 1)) * risk_hat + loss_upper_bound / (n + 1)
        if bound <= alpha:
            return float(lam)
    return float(lambda_grid[-1])


def decide(score: float, lambda_hat: float) -> bool:
    """Whether to accept a single test point given its confidence score."""
    return score >= lambda_hat
