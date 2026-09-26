"""Anytime-valid drift monitors for calibration profiles.

``RiskMonitor`` and ``CoverageMonitor`` wrap ``anytime.BettingMartingale``
to watch a stream of production outcomes and alarm the first time there
is strong sequential evidence that the deployed guarantee no longer holds
— with the false-alarm rate controlled no matter how long the monitor
runs or how often it is checked (research.md, section 9.1-9.2, [P096]).
"""

from __future__ import annotations

from cli_sdk.stats.evalues.anytime import BettingMartingale


class RiskMonitor:
    """Alarms when the realized risk (loss rate) drifts above ``target``."""

    def __init__(self, target: float, false_alarm_rate: float = 0.05):
        self.target = target
        self._martingale = BettingMartingale(null_rate=target, false_alarm_rate=false_alarm_rate)

    def update(self, is_loss: bool) -> dict | None:
        """Feed one production outcome. Returns an alert dict once triggered, else None."""
        e_value = self._martingale.update(1.0 if is_loss else 0.0)
        if self._martingale.triggered:
            return {
                "type": "risk",
                "severity": "critical",
                "target": self.target,
                "n": self._martingale.n,
                "e_value": e_value,
                "false_alarm_rate": self._martingale.false_alarm_rate,
            }
        return None


class CoverageMonitor:
    """Alarms when realized coverage drifts below ``target``.

    Internally monitors the *miss* rate (1 - covered) against
    ``1 - target``, which is the same risk-monitoring construction
    applied to misses instead of losses.
    """

    def __init__(self, target: float, false_alarm_rate: float = 0.05):
        self.target = target
        self._risk_monitor = RiskMonitor(target=1.0 - target, false_alarm_rate=false_alarm_rate)

    def update(self, covered: bool) -> dict | None:
        """Feed one production outcome. Returns an alert dict once triggered, else None."""
        alert = self._risk_monitor.update(is_loss=not covered)
        if alert:
            alert["type"] = "coverage"
            alert["target"] = self.target  # overwrite the inner monitor's (1 - target) value
        return alert
