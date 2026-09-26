"""Drift alerts raised by anytime-valid monitors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

ALERT_TYPES = ("coverage", "risk", "fingerprint")


@dataclass(frozen=True)
class Alert:
    """One monitor alert.

    ``e_value`` is the monitor's evidence against "the guarantee still
    holds"; it crossed ``1 / false_alarm_rate`` when the alert fired.
    Because the monitor is an e-process, this false-alarm rate holds no
    matter how often, or for how long, the monitor was checked.
    """

    type: str
    severity: str = "critical"
    profile: Optional[str] = None
    n: Optional[int] = None
    e_value: Optional[float] = None
    target: Optional[float] = None
    false_alarm_rate: Optional[float] = None
    detected_at: Optional[str] = None
    message: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "Alert":
        known = {"type", "severity", "profile", "n", "e_value", "target",
                 "false_alarm_rate", "detected_at", "message"}
        return cls(
            type=data.get("type", "coverage"),
            severity=data.get("severity", "critical"),
            profile=data.get("profile"),
            n=data.get("n"),
            e_value=data.get("e_value"),
            target=data.get("target"),
            false_alarm_rate=data.get("false_alarm_rate"),
            detected_at=data.get("detected_at"),
            message=data.get("message"),
            details={k: v for k, v in data.items() if k not in known},
        )
