"""Split conformal prediction: sets, intervals, and risk control.

Every function here is a plain NumPy function with no side effects and no
network access — the calibration procedures from research.md, section 2,
implemented directly. See the individual submodules for the method each
one implements and which CLI primitive it powers.
"""

from cli_sdk.stats.conformal import aps, cqr, crc, lac, ltt, mondrian, raps, rcps
from cli_sdk.stats.conformal._quantile import (
    conformal_quantile,
    coverage_confidence_interval,
    minimum_calibration_size,
)

__all__ = [
    "lac",
    "aps",
    "raps",
    "cqr",
    "crc",
    "rcps",
    "ltt",
    "mondrian",
    "conformal_quantile",
    "minimum_calibration_size",
    "coverage_confidence_interval",
]
