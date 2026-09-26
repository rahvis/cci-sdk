"""Learn-then-Test (Angelopoulos, Bates, Fisch, Lei & Schuster, AoAS 2025).

Recasts multi-parameter risk control as multiple hypothesis testing:
each candidate configuration lambda gets a null H0(lambda): "risk(lambda)
> alpha," a Hoeffding-Bentkus p-value, and a family-wise-error-rate
procedure (fixed-sequence testing by default) selects every lambda that
survives. Unlike RCPS's single monotone threshold search, LTT supports
non-monotone losses and several jointly-tuned knobs (e.g. a sample count
together with a quality threshold). See research.md [P063].
"""

from __future__ import annotations

import numpy as np

from cli_sdk.stats.conformal.rcps import _hoeffding_bentkus_p_value


def calibrate_grid(
    configurations: list[dict],
    risk_fn,
    alpha: float,
    delta: float,
    order: str = "conservative_first",
) -> list[dict]:
    """Return every configuration in ``configurations`` that is (alpha, delta)-valid.

    Parameters
    ----------
    configurations:
        A list of candidate configuration dicts (e.g.
        ``{"sample_count": 10, "quality_threshold": 0.4}``), ordered from
        most to least conservative if ``order="conservative_first"``.
    risk_fn:
        A callable ``risk_fn(config) -> (risk_hat: float, n: int)`` that
        evaluates the empirical risk and effective calibration count for
        one configuration.
    alpha, delta:
        Target risk level and failure probability.
    order:
        ``"conservative_first"`` runs fixed-sequence testing (stops at the
        first rejection, no multiplicity correction needed, but requires
        risk to be non-increasing along the given order).
        ``"bonferroni"`` tests every configuration independently at
        ``delta / len(configurations)``, valid for any ordering or lack
        of monotonicity, at the cost of a looser bound.
    """
    if order not in ("conservative_first", "bonferroni"):
        raise ValueError("order must be 'conservative_first' or 'bonferroni'")

    valid: list[dict] = []
    if order == "conservative_first":
        for config in configurations:
            risk_hat, n = risk_fn(config)
            p_value = _hoeffding_bentkus_p_value(risk_hat, n, alpha)
            if p_value <= delta:
                valid.append(config)
            else:
                break
        return valid

    per_test_delta = delta / max(len(configurations), 1)
    for config in configurations:
        risk_hat, n = risk_fn(config)
        p_value = _hoeffding_bentkus_p_value(risk_hat, n, alpha)
        if p_value <= per_test_delta:
            valid.append(config)
    return valid


def select_smallest(
    valid_configurations: list[dict], size_fn
) -> dict | None:
    """From a set of LTT-valid configurations, pick the one minimizing ``size_fn``."""
    if not valid_configurations:
        return None
    return min(valid_configurations, key=size_fn)
