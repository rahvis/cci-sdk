from __future__ import annotations

import numpy as np

from cli_sdk.stats.venn_abers import cvap, ivap
from cli_sdk.stats.venn_abers._isotonic import pool_adjacent_violators


def test_pav_produces_non_decreasing_output():
    y = np.array([0.3, 0.1, 0.4, 0.2, 0.9, 0.8, 1.0])
    fitted = pool_adjacent_violators(y)
    assert np.all(np.diff(fitted) >= -1e-12)


def test_pav_leaves_already_monotone_input_unchanged():
    y = np.array([0.0, 0.2, 0.5, 0.5, 0.9, 1.0])
    fitted = pool_adjacent_violators(y)
    assert np.allclose(fitted, y)


def test_ivap_interval_is_ordered_and_bounded():
    rng = np.random.default_rng(0)
    n = 400
    cal_scores = rng.uniform(0, 1, size=n)
    cal_labels = (rng.uniform(0, 1, size=n) < cal_scores).astype(float)
    test_scores = np.array([0.05, 0.5, 0.95])

    p0, p1 = ivap.calibrate_and_predict(cal_scores, cal_labels, test_scores)

    assert np.all(p0 <= p1 + 1e-9)
    assert np.all((p0 >= 0) & (p0 <= 1))
    assert np.all((p1 >= 0) & (p1 <= 1))


def test_ivap_tracks_the_true_relationship_on_strong_signal():
    rng = np.random.default_rng(1)
    n = 2000
    cal_scores = rng.uniform(0, 1, size=n)
    cal_labels = (rng.uniform(0, 1, size=n) < cal_scores**3).astype(float)  # strong, monotone signal

    p0_lo, p1_lo = ivap.calibrate_and_predict(cal_scores, cal_labels, np.array([0.05]))
    p0_hi, p1_hi = ivap.calibrate_and_predict(cal_scores, cal_labels, np.array([0.95]))

    midpoint_lo = ivap.merge_to_probability(p0_lo, p1_lo)[0]
    midpoint_hi = ivap.merge_to_probability(p0_hi, p1_hi)[0]
    assert midpoint_lo < 0.15
    assert midpoint_hi > 0.75


def test_interval_width_is_nonnegative():
    p0 = np.array([0.1, 0.4])
    p1 = np.array([0.3, 0.6])
    width = ivap.interval_width(p0, p1)
    assert np.all(width >= 0)
    assert np.allclose(width, [0.2, 0.2])


def test_cvap_returns_ordered_interval_and_runs_without_error():
    rng = np.random.default_rng(3)
    n = 300
    cal_scores = rng.uniform(0, 1, size=n)
    cal_labels = (rng.uniform(0, 1, size=n) < cal_scores).astype(float)
    test_scores = np.array([0.1, 0.5, 0.9])

    p0, p1 = cvap.calibrate_and_predict(cal_scores, cal_labels, test_scores, n_folds=4, seed=0)
    assert np.all(p0 <= p1 + 1e-9)
    assert np.all((p0 >= 0) & (p1 <= 1))
