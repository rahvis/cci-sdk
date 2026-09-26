"""APS: adaptive prediction sets (Romano, Sesia & Candes; Angelopoulos et al.).

Nonconformity score sorts classes by probability (descending) and sums
probability mass up to and including the true class. Produces larger
sets on ambiguous inputs and smaller sets on clear ones, which usually
gives better conditional coverage than LAC at a small cost in average
set size. See research.md [P064], [P071] (RAPS).

Randomization (the ``randomize`` flag) follows Romano, Sesia & Candes
(2020). With the defaults (randomized calibration scores, deterministic
prediction that includes the class crossing the threshold) coverage is at
least 1 - alpha and usually a little above it: conservative, not exact.
Exact 1 - alpha coverage in expectation needs ``randomize=True`` at both
calibration and prediction, at the cost of randomized sets.
"""

from __future__ import annotations

import numpy as np

from cli_sdk.stats.conformal._quantile import conformal_quantile


def _sorted_cumulative_mass(probs_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (order, cumulative mass in that order) for one row of probabilities."""
    order = np.argsort(-probs_row, kind="stable")
    cum = np.cumsum(probs_row[order])
    return order, cum


def _score_true_class(
    probs_row: np.ndarray, true_idx: int, rng: np.random.Generator | None
) -> float:
    order, cum = _sorted_cumulative_mass(probs_row)
    rank = int(np.flatnonzero(order == true_idx)[0])
    prev_mass = cum[rank - 1] if rank > 0 else 0.0
    true_mass = probs_row[true_idx]
    if rng is None:
        return float(prev_mass + true_mass)
    u = rng.uniform(0.0, 1.0)
    return float(prev_mass + u * true_mass)


def calibrate(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float,
    randomize: bool = True,
    seed: int | None = None,
) -> float:
    """Calibrate the APS threshold.

    cal_probs: (n, k) predicted probabilities. cal_labels: (n,) true class
    indices. alpha: target miscoverage rate.
    """
    cal_probs = np.asarray(cal_probs, dtype=float)
    cal_labels = np.asarray(cal_labels, dtype=int)
    rng = np.random.default_rng(seed) if randomize else None
    scores = np.array(
        [
            _score_true_class(cal_probs[i], int(cal_labels[i]), rng)
            for i in range(cal_probs.shape[0])
        ]
    )
    return conformal_quantile(scores, alpha)


def predict(
    test_probs: np.ndarray,
    q_hat: float,
    randomize: bool = False,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Return the APS prediction set (as class indices) for each test row.

    Sets are returned by including classes, in descending-probability
    order, until the cumulative mass first exceeds q_hat (i.e. every class
    whose *prefix-inclusive* cumulative mass is <= q_hat, plus the first
    class that crosses it deterministically; with ``randomize=True`` the
    crossing class is included only with the calibrated tie-breaking
    probability, matching how the score was constructed at calibration
    time).
    """
    test_probs = np.asarray(test_probs, dtype=float)
    rng = np.random.default_rng(seed) if randomize else None
    sets = []
    for row in test_probs:
        order, cum = _sorted_cumulative_mass(row)
        cutoff = np.searchsorted(cum, q_hat, side="left")
        included = list(order[: cutoff + 1]) if cutoff < len(order) else list(order)
        if randomize and cutoff < len(order):
            prev_mass = cum[cutoff - 1] if cutoff > 0 else 0.0
            true_mass = row[order[cutoff]]
            u = rng.uniform(0.0, 1.0)  # type: ignore[union-attr]
            if prev_mass + u * true_mass > q_hat:
                included = included[:-1]
        sets.append(np.array(sorted(included)))
    return sets
