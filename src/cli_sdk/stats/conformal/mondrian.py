"""Mondrian (group-conditional) conformal calibration.

Fits one conformal threshold per predefined group instead of one global
threshold, so a marginal coverage guarantee cannot silently hide much
lower coverage on a subgroup that matters to the business (research.md,
section 9.8). Groups without enough calibration examples for the target
alpha fall back to the marginal threshold, and this fallback is reported
so callers can see which groups are under-calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cli_sdk.stats.conformal._quantile import conformal_quantile, minimum_calibration_size


@dataclass
class MondrianCalibration:
    alpha: float
    group_thresholds: dict[str, float] = field(default_factory=dict)
    group_sizes: dict[str, int] = field(default_factory=dict)
    fallback_threshold: float = float("inf")
    underpowered_groups: set[str] = field(default_factory=set)

    def threshold_for(self, group) -> float:
        """The group's threshold; the marginal fallback for underpowered or unseen groups.

        Keys are compared as ``str(group)``, the form ``calibrate`` stores, so
        integer, boolean or enum group labels find their own threshold.
        Check ``is_calibrated(group)`` to know whether the fallback was used.
        """
        key = str(group)
        if key in self.underpowered_groups or key not in self.group_thresholds:
            return self.fallback_threshold
        return self.group_thresholds[key]

    def is_calibrated(self, group) -> bool:
        key = str(group)
        return key in self.group_thresholds and key not in self.underpowered_groups


def calibrate(
    scores: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> MondrianCalibration:
    """Calibrate one conformal threshold per group.

    scores: (n,) nonconformity scores of the true label for each
    calibration example (as produced by, e.g., ``lac`` or ``aps``).
    groups: (n,) group label for each calibration example (e.g. account
    tier, language). alpha: target miscoverage rate, applied per group.
    """
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    min_n = minimum_calibration_size(alpha)

    result = MondrianCalibration(alpha=alpha)
    result.fallback_threshold = conformal_quantile(scores, alpha)

    for group in np.unique(groups):
        group_scores = scores[groups == group]
        result.group_sizes[str(group)] = int(group_scores.shape[0])
        if group_scores.shape[0] < min_n:
            result.underpowered_groups.add(str(group))
            continue
        result.group_thresholds[str(group)] = conformal_quantile(group_scores, alpha)

    return result
