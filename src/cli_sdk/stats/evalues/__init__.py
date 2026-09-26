"""E-value based procedures: anytime-valid monitoring, FDR control, and
label-efficient calibration.

See the individual submodules for what each implements:

``anytime``
    Shared betting-martingale construction and the normal-quantile helper.
``eprocess``
    ``RiskMonitor`` / ``CoverageMonitor`` — anytime-valid drift alarms.
``ebh``
    e-BH selection for FDR control under arbitrary dependence, and a
    p-value-to-e-value calibrator.
``ppi``
    Prediction-powered mean estimation for label-efficient calibration.
"""

from cli_sdk.stats.evalues import anytime, ebh, eprocess, ppi
from cli_sdk.stats.evalues.eprocess import CoverageMonitor, RiskMonitor

__all__ = ["anytime", "eprocess", "ebh", "ppi", "CoverageMonitor", "RiskMonitor"]
