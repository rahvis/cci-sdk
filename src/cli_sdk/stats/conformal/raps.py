"""RAPS: regularized adaptive prediction sets (Angelopoulos et al., ICLR 2021).

APS with a rank penalty that discourages including low-probability tail
classes purely to reach the cumulative-mass threshold, which shrinks
average set size relative to plain APS with only a small conditional-
coverage cost. Useful when the option list is large. See research.md
[P071].
"""

from __future__ import annotations

import numpy as np

from cli_sdk.stats.conformal._quantile import conformal_quantile


def _penalized_cumulative_mass(
    probs_row: np.ndarray, k_reg: int, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sort classes by probability and return the cumulative probability
    mass plus cumulative rank penalty, in that order.

    Ranks are 1-indexed. The value at rank r is the published RAPS total
    (Angelopoulos et al., 2021): the probability mass of the r most likely
    classes plus lam * (r - k_reg)^+, so each rank beyond k_reg adds lam once.
    """
    order = np.argsort(-probs_row, kind="stable")
    sorted_probs = probs_row[order]
    ranks = np.arange(1, len(order) + 1)
    cum = np.cumsum(sorted_probs) + lam * np.maximum(ranks - k_reg, 0)
    return order, cum


def true_class_score(probs_row: np.ndarray, true_idx: int, k_reg: int, lam: float, u: float = 1.0) -> float:
    """RAPS nonconformity of the true class (Angelopoulos et al., 2021).

    E(x, y, u) = (mass of the classes ranked above y) + u * p(y) + lam * (o(y) - k_reg)^+,
    with o(y) the 1-indexed rank of y and u in [0, 1] (u = 1: non-randomized).
    """
    order = np.argsort(-probs_row, kind="stable")
    rank = int(np.flatnonzero(order == true_idx)[0])
    mass_before = float(np.sum(probs_row[order[:rank]]))
    return mass_before + u * float(probs_row[true_idx]) + lam * max(rank + 1 - k_reg, 0)


def calibrate(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float,
    k_reg: int = 1,
    lam: float = 0.01,
    randomize: bool = True,
    seed: int | None = None,
) -> float:
    """Calibrate the RAPS threshold.

    k_reg: the rank below which no penalty is applied (typically a small
    integer, e.g. the number of classes you consider "plausibly enough").
    lam: penalty weight per rank beyond k_reg.
    """
    cal_probs = np.asarray(cal_probs, dtype=float)
    cal_labels = np.asarray(cal_labels, dtype=int)
    rng = np.random.default_rng(seed) if randomize else None
    scores = np.empty(cal_probs.shape[0])
    for i in range(cal_probs.shape[0]):
        u = rng.uniform(0.0, 1.0) if rng is not None else 1.0
        scores[i] = true_class_score(cal_probs[i], int(cal_labels[i]), k_reg, lam, u)
    return conformal_quantile(scores, alpha)


def predict(
    test_probs: np.ndarray,
    q_hat: float,
    k_reg: int = 1,
    lam: float = 0.01,
) -> list[np.ndarray]:
    """Return the RAPS prediction set (as class indices) for each test row."""
    test_probs = np.asarray(test_probs, dtype=float)
    sets = []
    for row in test_probs:
        order, cum = _penalized_cumulative_mass(row, k_reg, lam)
        cutoff = np.searchsorted(cum, q_hat, side="left")
        included = order[: cutoff + 1] if cutoff < len(order) else order
        sets.append(np.array(sorted(included)))
    return sets
