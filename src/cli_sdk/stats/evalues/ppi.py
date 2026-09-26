"""Prediction-powered inference (PPI / PPI++) for label-efficient calibration.

Combines a small human-labelled sample with a large judge-labelled (or
otherwise machine-predicted) pool into a single valid estimate of a
population mean, with a confidence interval that is never worse than
using the human-labelled sample alone (with power tuning) and is valid
*regardless* of how good the predictions are — a bad judge only widens
the interval, it never invalidates it. Powers calibration profiles'
label-efficient labelling path. See research.md, section 10.3, and
Angelopoulos, Bates, Fannjiang, Jordan & Zrnic (Science, 2023) and
Angelopoulos, Duchi & Zrnic (PPI++, 2023).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cli_sdk.stats.evalues.anytime import normal_ppf


@dataclass
class PPIResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    lam: float
    standard_error: float


def estimate_mean(
    labelled_predictions: np.ndarray,
    labelled_labels: np.ndarray,
    unlabelled_predictions: np.ndarray,
    alpha: float = 0.05,
    tune_lambda: bool = True,
) -> PPIResult:
    """Prediction-powered estimate of E[label] over the full (labelled + unlabelled) population.

    Parameters
    ----------
    labelled_predictions, labelled_labels:
        (n,) arrays: the predictor's output and the true label, on the
        human-labelled sample.
    unlabelled_predictions:
        (N,) array: the predictor's output on the large, unlabelled (or
        only machine-labelled) pool.
    alpha:
        Confidence-interval level (default 95% CI).
    tune_lambda:
        If True (default), choose the power-tuning parameter lambda to
        minimize estimator variance (PPI++); if False, use lambda = 1
        (classical PPI).
    """
    f_lab = np.asarray(labelled_predictions, dtype=float)
    y_lab = np.asarray(labelled_labels, dtype=float)
    f_unlab = np.asarray(unlabelled_predictions, dtype=float)
    n, big_n = f_lab.shape[0], f_unlab.shape[0]
    if n < 2:
        raise ValueError("need at least 2 human-labelled examples")
    if big_n < 1:
        raise ValueError("need at least 1 unlabelled prediction")

    if tune_lambda:
        var_f_lab = float(np.var(f_lab, ddof=1))
        if var_f_lab > 0:
            cov = float(np.cov(f_lab, y_lab, ddof=1)[0, 1])
            # PPI++ power tuning (Angelopoulos et al., 2023): the variance-minimizing
            # lambda is Cov(f, y) / ((1 + n / N) Var(f)).
            lam = float(np.clip(cov / ((1.0 + n / big_n) * var_f_lab), 0.0, 1.0))
        else:
            lam = 0.0
    else:
        lam = 1.0

    rectifier = y_lab - lam * f_lab
    theta_hat = lam * float(f_unlab.mean()) + float(rectifier.mean())

    var_f_unlab = float(np.var(f_unlab, ddof=1)) if big_n > 1 else 0.0
    var_rectifier = float(np.var(rectifier, ddof=1))
    standard_error = float(np.sqrt((lam**2) * var_f_unlab / big_n + var_rectifier / n))

    z = normal_ppf(1 - alpha / 2)
    return PPIResult(
        estimate=theta_hat,
        ci_lower=theta_hat - z * standard_error,
        ci_upper=theta_hat + z * standard_error,
        lam=lam,
        standard_error=standard_error,
    )
