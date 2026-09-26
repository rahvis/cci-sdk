"""CQR: conformalized quantile regression (Romano, Patterson & Candes, 2019).

Wraps any pair of lower/upper quantile predictors (from a fine-tuned
regressor, a rubric-scoring LLM's probability-weighted levels, or any
other point-estimate-producing model) with a conformal correction so the
resulting interval achieves the target coverage regardless of how well
the underlying quantile predictions were fit. Powers the ``Interval``
primitive (research.md, section 6.2 recipe-family for CP regression).
"""

from __future__ import annotations

import numpy as np

from cli_sdk.stats.conformal._quantile import conformal_quantile


def calibrate(
    cal_lower: np.ndarray,
    cal_upper: np.ndarray,
    cal_targets: np.ndarray,
    alpha: float,
) -> float:
    """Calibrate the CQR correction term.

    cal_lower, cal_upper: (n,) uncalibrated lower/upper quantile
    predictions from the underlying scorer. cal_targets: (n,) true values.
    """
    cal_lower = np.asarray(cal_lower, dtype=float)
    cal_upper = np.asarray(cal_upper, dtype=float)
    cal_targets = np.asarray(cal_targets, dtype=float)
    scores = np.maximum(cal_lower - cal_targets, cal_targets - cal_upper)
    return conformal_quantile(scores, alpha)


def predict(
    test_lower: np.ndarray, test_upper: np.ndarray, q_hat: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the conformalized (lower, upper) interval bounds."""
    test_lower = np.asarray(test_lower, dtype=float)
    test_upper = np.asarray(test_upper, dtype=float)
    return test_lower - q_hat, test_upper + q_hat
