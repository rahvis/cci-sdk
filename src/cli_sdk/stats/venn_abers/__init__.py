"""Venn-Abers calibration: distribution-free probability intervals.

See ``ivap`` for the single-split Inductive Venn-Abers Predictor and
``cvap`` for a K-fold aggregated variant. Both return a [p0, p1] interval
per test point rather than a single probability.
"""

from cli_sdk.stats.venn_abers import cvap, ivap
from cli_sdk.stats.venn_abers.ivap import interval_width, merge_to_probability

__all__ = ["ivap", "cvap", "merge_to_probability", "interval_width"]
