"""LAC: least-ambiguous-set-valued classification (Sadinle, Lei & Wasserman).

Nonconformity score s(x, y) = 1 - p_hat(y | x). Produces the smallest
average prediction sets among standard conformal classification scores,
at the cost of uneven conditional coverage across easy/hard inputs.
See research.md [P065].
"""

from __future__ import annotations

import numpy as np

from cli_sdk.stats.conformal._quantile import conformal_quantile


def calibrate(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float) -> float:
    """Calibrate the LAC threshold.

    Parameters
    ----------
    cal_probs: (n, k) array of predicted probabilities over k classes.
    cal_labels: (n,) array of true class indices in [0, k).
    alpha: target miscoverage rate.
    """
    cal_probs = np.asarray(cal_probs, dtype=float)
    cal_labels = np.asarray(cal_labels, dtype=int)
    n = cal_probs.shape[0]
    true_class_probs = cal_probs[np.arange(n), cal_labels]
    scores = 1.0 - true_class_probs
    return conformal_quantile(scores, alpha)


def predict(test_probs: np.ndarray, q_hat: float) -> list[np.ndarray]:
    """Return the LAC prediction set (as class indices) for each test row."""
    test_probs = np.asarray(test_probs, dtype=float)
    scores = 1.0 - test_probs  # (m, k)
    return [np.flatnonzero(row <= q_hat) for row in scores]
