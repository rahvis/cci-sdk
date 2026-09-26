from __future__ import annotations

import numpy as np
import pytest

from cli_sdk.stats.conformal import aps, cqr, crc, lac, mondrian, raps, rcps
from cli_sdk.stats.conformal._quantile import (
    conformal_quantile,
    coverage_confidence_interval,
    minimum_calibration_size,
)


def test_minimum_calibration_size():
    assert minimum_calibration_size(0.1) == 9
    assert minimum_calibration_size(0.05) == 19
    assert minimum_calibration_size(0.01) == 99


def test_conformal_quantile_returns_inf_when_n_too_small():
    scores = np.array([0.1, 0.2, 0.3])
    assert conformal_quantile(scores, alpha=0.01) == float("inf")


def test_coverage_confidence_interval_centers_on_target():
    lower, upper = coverage_confidence_interval(n=1000, alpha=0.10, confidence=0.90)
    assert lower < 0.90 < upper
    assert 0.87 < lower < 0.90
    assert 0.90 < upper < 0.93


def _synthetic_classification(n, k=4, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    # A moderately informative "model": concentrate most mass on the true
    # label, spread the rest, so calibration has real signal to work with.
    probs = rng.dirichlet(alpha=np.ones(k) * 2.0, size=n)
    boost = rng.uniform(0.3, 0.6, size=n)
    probs *= (1 - boost)[:, None]
    probs[np.arange(n), labels] += boost
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, labels


@pytest.mark.parametrize("module", [lac, aps])
def test_set_methods_achieve_approximate_target_coverage(module):
    probs, labels = _synthetic_classification(n=4000)
    cal_probs, test_probs = probs[:2000], probs[2000:]
    cal_labels, test_labels = labels[:2000], labels[2000:]

    alpha = 0.10
    q_hat = module.calibrate(cal_probs, cal_labels, alpha=alpha)
    sets = module.predict(test_probs, q_hat)

    covered = np.mean([test_labels[i] in sets[i] for i in range(len(sets))])
    # The guaranteed property of split CP is the LOWER bound,
    # coverage >= 1 - alpha (here with a small slack for finite-sample
    # noise on 2000 test points). Deterministic (non-randomized)
    # prediction is conservative and can legitimately over-cover well
    # above 1 - alpha + 1/(n+1) — the tight two-sided guarantee only
    # holds when both calibration *and* prediction are randomized, which
    # CLI does not do by default because deterministic sets are easier
    # to reason about in production.
    assert covered >= alpha_lower_bound(alpha, n_test=len(sets))


def alpha_lower_bound(alpha: float, n_test: int) -> float:
    # A little slack below the nominal 1 - alpha to absorb sampling noise
    # on a finite test set, matching a ~3-sigma band at this sample size.
    import math

    p = 1 - alpha
    slack = 3 * math.sqrt(p * (1 - p) / n_test)
    return p - slack


def test_raps_sets_are_no_larger_than_aps_on_average():
    probs, labels = _synthetic_classification(n=3000, k=8)
    cal_probs, test_probs = probs[:1500], probs[1500:]
    cal_labels = labels[:1500]

    alpha = 0.10
    aps_q = aps.calibrate(cal_probs, cal_labels, alpha=alpha, seed=1)
    aps_sets = aps.predict(test_probs, aps_q)

    raps_q = raps.calibrate(cal_probs, cal_labels, alpha=alpha, k_reg=1, lam=0.05, seed=1)
    raps_sets = raps.predict(test_probs, raps_q, k_reg=1, lam=0.05)

    avg_aps_size = np.mean([len(s) for s in aps_sets])
    avg_raps_size = np.mean([len(s) for s in raps_sets])
    assert avg_raps_size <= avg_aps_size + 1e-9


def test_cqr_interval_contains_target_at_expected_rate():
    rng = np.random.default_rng(0)
    n = 3000
    true = rng.normal(0, 1, size=n)
    # Deliberately miscalibrated quantile predictions (too narrow) to
    # exercise the conformal correction.
    lower_raw = true - rng.uniform(0.2, 0.4, size=n) - 1.5
    upper_raw = true + rng.uniform(0.2, 0.4, size=n) + 0.1

    cal = slice(0, 1500)
    test = slice(1500, 3000)
    q_hat = cqr.calibrate(lower_raw[cal], upper_raw[cal], true[cal], alpha=0.10)
    lo, hi = cqr.predict(lower_raw[test], upper_raw[test], q_hat)

    covered = np.mean((true[test] >= lo) & (true[test] <= hi))
    assert covered >= 0.85


def test_crc_risk_is_bounded_by_alpha_in_expectation_over_repeats():
    rng = np.random.default_rng(0)
    alpha = 0.10
    violations = 0
    trials = 60
    for t in range(trials):
        rng_t = np.random.default_rng(t)
        scores = rng_t.uniform(0, 1, size=800)
        # Loss decreases with score (higher confidence -> lower error rate).
        loss_prob = np.clip(0.5 - 0.4 * scores, 0, 1)
        losses = (rng_t.uniform(0, 1, size=800) < loss_prob).astype(float)
        cal, test = slice(0, 500), slice(500, 800)
        lam = crc.calibrate(scores[cal], losses[cal], alpha=alpha)
        accepted = scores[test] >= lam
        if accepted.any() and losses[test][accepted].mean() > alpha + 0.15:
            violations += 1
    # CRC controls risk in expectation, not in every draw; this checks the
    # violation rate stays low across repeated calibration draws rather
    # than asserting a single-draw guarantee.
    assert violations / trials < 0.35


def test_rcps_threshold_is_at_least_as_conservative_as_crc():
    rng = np.random.default_rng(2)
    scores = rng.uniform(0, 1, size=1000)
    loss_prob = np.clip(0.5 - 0.4 * scores, 0, 1)
    losses = (rng.uniform(0, 1, size=1000) < loss_prob).astype(float)

    alpha = 0.10
    lam_crc = crc.calibrate(scores, losses, alpha=alpha)
    lam_rcps = rcps.calibrate(scores, losses, alpha=alpha, delta=0.05)
    # RCPS's high-probability guarantee is strictly stronger than CRC's
    # in-expectation guarantee, so it should never accept a *lower* bar.
    assert lam_rcps >= lam_crc - 1e-9


def test_mondrian_flags_underpowered_groups():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, size=200)
    groups = np.array(["large"] * 190 + ["tiny"] * 10)
    result = mondrian.calibrate(scores, groups, alpha=0.01)  # needs n>=99 per group
    assert "tiny" in result.underpowered_groups
    assert "large" not in result.underpowered_groups
    assert result.threshold_for("tiny") == result.fallback_threshold
