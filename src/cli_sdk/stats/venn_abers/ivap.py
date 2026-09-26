"""Inductive Venn-Abers Predictor (Vovk, Petej & Fedorova, 2015).

For a binary label and any real-valued score (a raw model probability, a
logit, a judge rating — anything monotonically related to "more likely
label 1"), IVAP produces a pair [p0, p1] with a proven validity property:
of the two, the probability computed under the true label is perfectly
calibrated on data exchangeable with the calibration set. It is not a
claim that the true conditional probability of one input lies in [p0, p1];
the width of the pair signals how thin the calibration data is near this
score. This holds regardless of how well the
underlying score was trained — a poorly separated score only widens the
interval, it never invalidates the guarantee. See research.md, section
2.1 / 5.1, and Vovk, Gammerman & Shafer's treatment of Venn predictors
(the same theoretical family as split conformal prediction).

Algorithm
---------
For a test point with score s*:

  p1 = fit isotonic regression of label-on-score over
       (calibration data) union {(s*, 1)}, evaluated at s*.
  p0 = the same, but with {(s*, 0)} instead.

This implementation is a direct, correct realization of that definition:
one isotonic regression (via pool-adjacent-violators) per hypothesized
label per test point, which is O(n log n) per test point. A production
deployment scoring very large batches can use the amortized algorithm
from the original paper, which precomputes the isotonic regression's
convex-hull structure once; that optimization does not change the
values returned here, only the speed.
"""

from __future__ import annotations

import numpy as np

from cli_sdk.stats.venn_abers._isotonic import pool_adjacent_violators


def _fit_at(
    cal_scores: np.ndarray, cal_labels: np.ndarray, query_score: float, query_label: float
) -> float:
    scores = np.append(cal_scores, query_score)
    labels = np.append(cal_labels, query_label)
    # Isotonic regression is a function of the score, so tied scores must get
    # one fitted value: pool each distinct score into a single weighted point
    # (mean label, weight = count) before running pool-adjacent-violators.
    # Fitting tied points separately would make the result depend on their
    # arbitrary order within the tie, which LLM scores (sampled frequencies,
    # rounded probabilities) produce constantly.
    unique_scores, inverse = np.unique(scores, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    means = np.bincount(inverse, weights=labels) / counts
    fitted = pool_adjacent_violators(means, counts)
    return float(fitted[inverse[-1]])


def calibrate_and_predict(
    cal_scores: np.ndarray, cal_labels: np.ndarray, test_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (p0, p1) arrays, one interval per test score.

    cal_scores: (n,) any real-valued score, monotonically related to
    "more likely label 1." cal_labels: (n,) binary labels in {0, 1}.
    test_scores: (m,) scores to calibrate.
    """
    cal_scores = np.asarray(cal_scores, dtype=float)
    cal_labels = np.asarray(cal_labels, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)

    p0 = np.empty(test_scores.shape[0])
    p1 = np.empty(test_scores.shape[0])
    for i, s in enumerate(test_scores):
        p0[i] = _fit_at(cal_scores, cal_labels, s, 0.0)
        p1[i] = _fit_at(cal_scores, cal_labels, s, 1.0)

    # p0 <= p1 by construction of the algorithm; guard against floating-
    # point crossings at degenerate (e.g. constant-score) inputs.
    return np.minimum(p0, p1), np.maximum(p0, p1)


def merge_to_probability(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Collapse a Venn-Abers interval to a single point probability.

    Uses the standard regularized-merging rule p = p1 / (1 - p0 + p1),
    which minimizes worst-case log loss over the interval (Vovk & Petej).
    Prefer reporting the interval itself when the interval's width is
    meaningful to the caller; use this only where a single scalar is
    required downstream.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    denom = 1.0 - p0 + p1
    return np.divide(p1, denom, out=np.full_like(p1, 0.5), where=denom > 0)


def interval_width(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """The interval width — itself a meaningful ambiguity signal.

    A narrow interval near 0 or 1 indicates a well-supported decision; a
    wide interval indicates genuine ambiguity or sparse calibration
    coverage near this score, independent of any downstream threshold.
    """
    return np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
