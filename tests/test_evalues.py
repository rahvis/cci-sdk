from __future__ import annotations

import numpy as np

from cli_sdk.stats.evalues import ebh, ppi
from cli_sdk.stats.evalues.eprocess import CoverageMonitor, RiskMonitor


def test_risk_monitor_does_not_trigger_on_in_spec_stream():
    rng = np.random.default_rng(0)
    monitor = RiskMonitor(target=0.10, false_alarm_rate=0.01)
    triggered = False
    for _ in range(3000):
        is_loss = rng.uniform() < 0.08  # true rate below target
        if monitor.update(is_loss):
            triggered = True
            break
    assert not triggered


def test_risk_monitor_triggers_on_clearly_out_of_spec_stream():
    rng = np.random.default_rng(1)
    monitor = RiskMonitor(target=0.05, false_alarm_rate=0.05)
    triggered = False
    for _ in range(3000):
        is_loss = rng.uniform() < 0.40  # far above target
        alert = monitor.update(is_loss)
        if alert:
            triggered = True
            assert alert["type"] == "risk"
            assert alert["target"] == 0.05
            break
    assert triggered


def test_coverage_monitor_triggers_when_coverage_drops():
    rng = np.random.default_rng(2)
    monitor = CoverageMonitor(target=0.90, false_alarm_rate=0.05)
    triggered = False
    for _ in range(3000):
        covered = rng.uniform() < 0.55  # well below target coverage
        alert = monitor.update(covered)
        if alert:
            triggered = True
            assert alert["type"] == "coverage"
            assert alert["target"] == 0.90
            break
    assert triggered


def test_ebh_selects_strong_evidence_and_controls_false_selection_rate():
    rng = np.random.default_rng(0)
    # e-BH's threshold at rank k is m / (q * k): with m = 20 hypotheses
    # and q = 0.20, the k = 5 threshold is 20/(0.2*5) = 20, so signal
    # e-values need to clear roughly that order of magnitude to be
    # selected -- this is deliberately a large, unambiguous effect size
    # so the test exercises "does e-BH select strong evidence" rather
    # than probing exactly where its conservative threshold sits.
    signal = rng.uniform(500, 2000, size=5)
    null = rng.exponential(1.0, size=15)  # E[e] = 1 under the null
    e_values = np.concatenate([signal, null])

    selected = ebh.select(e_values, q=0.20)
    assert set(range(5)).issubset(set(selected.tolist()))
    false_selections = [i for i in selected if i >= 5]
    assert len(false_selections) == 0


def test_ebh_is_conservative_on_weak_evidence():
    # A separate, deliberately weak-evidence case: e-BH's arbitrary-
    # dependence correction is strict relative to the number of
    # hypotheses, so modest e-values relative to m should select little
    # or nothing rather than over-claiming discoveries.
    rng = np.random.default_rng(0)
    signal = rng.uniform(50, 200, size=8)
    null = rng.exponential(1.0, size=92)
    e_values = np.concatenate([signal, null])

    selected = ebh.select(e_values, q=0.10)
    false_selections = [i for i in selected if i >= 8]
    assert len(false_selections) == 0


def test_ebh_selects_nothing_when_no_evidence():
    e_values = np.ones(20)
    selected = ebh.select(e_values, q=0.10)
    assert selected.shape[0] == 0


def test_calibrator_from_p_value_is_monotone_decreasing_in_p():
    e_small_p = ebh.calibrator_from_p_value(0.01)
    e_large_p = ebh.calibrator_from_p_value(0.5)
    assert e_small_p > e_large_p


def test_ppi_estimate_recovers_true_mean_with_informative_predictor():
    rng = np.random.default_rng(0)
    true_mean = 0.3
    n, big_n = 200, 5000

    population_labels = (rng.uniform(0, 1, size=big_n) < true_mean).astype(float)
    # A predictor correlated with, but not identical to, the true label.
    predictions = np.clip(population_labels + rng.normal(0, 0.3, size=big_n), 0, 1)

    labelled_idx = rng.choice(big_n, size=n, replace=False)
    unlabelled_idx = np.setdiff1d(np.arange(big_n), labelled_idx)

    result = ppi.estimate_mean(
        labelled_predictions=predictions[labelled_idx],
        labelled_labels=population_labels[labelled_idx],
        unlabelled_predictions=predictions[unlabelled_idx],
        alpha=0.05,
    )

    assert result.ci_lower <= true_mean <= result.ci_upper
    assert abs(result.estimate - true_mean) < 0.08
