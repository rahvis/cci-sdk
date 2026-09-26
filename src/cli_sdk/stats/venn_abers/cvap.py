"""Cross Venn-Abers Predictor: K-fold aggregation of IVAP.

Splits the calibration set into K folds, fits an IVAP on each K-1-fold
union, and evaluates every test point against all K calibrators. This
uses the calibration data more fully than a single train/calibration
split (the same motivation as cross-conformal prediction over split
conformal), at the cost of K times the computation.

Combining K separate [p0, p1] intervals into one is not uniquely
prescribed in the literature; this implementation combines them by
averaging in log-odds space, which is a standard, well-behaved choice for
combining probability estimates and keeps the result inside [0, 1]
without needing case-by-case tie-breaking. Treat the resulting interval as
an aggregate signal in the same spirit as the single-fold interval, not
as a claim of the tightest possible K-fold guarantee.
"""

from __future__ import annotations

import numpy as np

from cli_sdk.stats.venn_abers import ivap


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def calibrate_and_predict(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    n_folds: int = 5,
    seed: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (p0, p1) arrays combined across ``n_folds`` IVAP calibrators."""
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_labels = np.asarray(cal_labels, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)

    n = cal_scores.shape[0]
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2; use ivap.calibrate_and_predict for a single split")

    rng = np.random.default_rng(seed)
    fold_ids = rng.permutation(n) % n_folds

    all_p0 = np.empty((n_folds, test_scores.shape[0]))
    all_p1 = np.empty((n_folds, test_scores.shape[0]))
    for fold in range(n_folds):
        keep = fold_ids != fold
        p0, p1 = ivap.calibrate_and_predict(cal_scores[keep], cal_labels[keep], test_scores)
        all_p0[fold], all_p1[fold] = p0, p1

    combined_p0 = _sigmoid(_logit(all_p0).mean(axis=0))
    combined_p1 = _sigmoid(_logit(all_p1).mean(axis=0))
    return np.minimum(combined_p0, combined_p1), np.maximum(combined_p0, combined_p1)
