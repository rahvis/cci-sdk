"""Regressions for statistics-engine defects found in review.

Each test pins the published definition of a method, so a future change
that quietly departs from it fails here.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from cli_sdk.calibration.audit import clopper_pearson
from cli_sdk.stats.conformal import crc, mondrian, raps, rcps
from cli_sdk.stats.evalues import ppi
from cli_sdk.stats.evalues.eprocess import CoverageMonitor, RiskMonitor


def test_raps_penalty_is_linear_in_rank():
    probs = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01])
    _, cum = raps._penalized_cumulative_mass(probs, k_reg=1, lam=0.1)
    np.testing.assert_allclose(cum, [0.5, 0.9, 1.1, 1.25, 1.38, 1.49, 1.6])
    # the true-class score is the same quantity for the class at rank 3 (u = 1)
    assert raps.true_class_score(probs, 2, k_reg=1, lam=0.1) == pytest.approx(1.1)


def test_mondrian_accepts_non_string_group_labels():
    rng = np.random.default_rng(0)
    groups = np.repeat([0, 1, 2], 200)
    scores = rng.uniform(size=600) + (groups == 2) * 0.5
    fit = mondrian.calibrate(scores, groups, alpha=0.1)
    assert fit.threshold_for(2) == fit.threshold_for("2") != fit.fallback_threshold
    assert fit.is_calibrated(2) and not fit.is_calibrated(7)


def test_crc_bounds_the_joint_rate_across_draws():
    alpha, rates = 0.1, []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        scores = rng.uniform(size=400)
        losses = (rng.uniform(size=400) > scores).astype(float)
        lam = crc.calibrate(scores[:200], losses[:200], alpha)
        rates.append(np.mean(losses[200:] * (scores[200:] >= lam)))
    assert np.mean(rates) <= alpha + 0.01


def test_rcps_returns_a_finite_threshold_and_controls_risk():
    alpha, delta, violations = 0.1, 0.1, 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        scores = rng.uniform(size=2000)
        losses = (rng.uniform(size=2000) > scores).astype(float)
        lam = rcps.calibrate(scores[:1000], losses[:1000], alpha, delta)
        assert np.isfinite(lam)
        violations += np.mean(losses[1000:] * (scores[1000:] >= lam)) > alpha
    assert violations / 100 <= delta + 0.05


def test_hoeffding_bentkus_uses_the_ceiling_count():
    # 15 losses in 100 read back as 14.999999999999998 in floating point.
    risk_hat = float(np.mean([1.0] * 15 + [0.0] * 85))
    exact = rcps._hoeffding_bentkus_p_value(0.15, 100, 0.3)
    assert rcps._hoeffding_bentkus_p_value(risk_hat, 100, 0.3) == pytest.approx(exact)


def test_clopper_pearson_matches_reference_values():
    assert clopper_pearson(45, 50) == pytest.approx((0.7818646, 0.9667249), abs=1e-6)
    assert clopper_pearson(90, 100) == pytest.approx((0.8237774, 0.9509953), abs=1e-6)


def test_coverage_monitor_detects_moderate_drift_and_never_underflows():
    rng = random.Random(0)
    monitor = CoverageMonitor(target=0.90)
    alarm = None
    for i in range(50_000):
        if monitor.update(rng.random() < 0.87):
            alarm = i
            break
    assert alarm is not None
    healthy = RiskMonitor(target=0.05)
    for _ in range(20_000):
        healthy.update(False)
    assert healthy._martingale.log_e_value > -1e6  # finite log-wealth, no 0.0 components
    assert not healthy._martingale.triggered


def test_monitor_false_alarm_rate_is_controlled():
    alarms = 0
    for seed in range(200):
        rng = random.Random(seed)
        monitor = RiskMonitor(target=0.05, false_alarm_rate=0.05)
        for _ in range(2000):
            if monitor.update(rng.random() < 0.05):
                alarms += 1
                break
    assert alarms / 200 <= 0.05 + 0.03


def test_ppi_is_never_much_worse_than_labels_alone_with_a_small_pool():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 1000).astype(float)
    f = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 1000), 0, 1)
    pool = np.clip(rng.binomial(1, 0.3, 100) * 0.6 + rng.normal(0.2, 0.2, 100), 0, 1)
    result = ppi.estimate_mean(f, y, pool)
    human_only_se = np.std(y, ddof=1) / np.sqrt(len(y))
    assert result.standard_error <= 1.1 * human_only_se
    with pytest.raises(ValueError):
        ppi.estimate_mean(f, y, np.array([]))


def test_judge_multiplier_is_capped_by_the_pool_size():
    from cli_sdk.calibration import judge_quality

    human = [1, 0, 1, 1, 0, 1, 0, 0] * 10
    q = judge_quality(human, human, pool_size=320)
    assert q.effective_label_multiplier == pytest.approx((80 + 320) / 80)
