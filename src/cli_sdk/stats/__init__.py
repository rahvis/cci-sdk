"""Standalone, network-free statistics engine for Conformal Logit Inference.

This package implements conformal prediction, Venn-Abers calibration, and
e-value procedures directly on NumPy arrays. It has no dependency on the
``cli_sdk`` HTTP client and no network access of any kind — it is the same
engine the hosted API runs, packaged so it can be used entirely offline,
including inside air-gapped environments.

Submodules
----------
``cli_sdk.stats.conformal``
    Split conformal prediction: LAC, APS, RAPS, CQR, conformal risk
    control (CRC), Learn-then-Test (LTT), risk-controlling prediction
    sets (RCPS), and Mondrian (group-conditional) calibration.
``cli_sdk.stats.venn_abers``
    Inductive and cross Venn-Abers predictors (IVAP / CVAP) for
    distribution-free probability calibration.
``cli_sdk.stats.evalues``
    E-value based procedures: e-BH selection, anytime-valid e-process
    monitors, and a simple prediction-powered inference (PPI) estimator.
"""

from cli_sdk.stats import conformal, evalues, venn_abers

__all__ = ["conformal", "venn_abers", "evalues"]
