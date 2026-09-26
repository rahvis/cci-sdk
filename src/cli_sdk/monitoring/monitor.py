"""Drift monitors: hosted resources and an offline equivalent.

Hosted monitors are created per calibration profile and polled for alerts:

    client.calibration_profiles.monitors.create("support-routing-v3", type="coverage",
                                                target=0.90, false_alarm_rate=0.05)
    for alert in client.calibration_profiles.monitors.poll("support-routing-v3"):
        ...

``LocalMonitor`` runs the same e-process in your own process, with no
network access, for air-gapped deployments or for replaying logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional
from urllib.parse import quote

from cli_sdk.exceptions import ConfigurationError
from cli_sdk.monitoring.alerts import Alert
from cli_sdk.stats.evalues.eprocess import CoverageMonitor, RiskMonitor

if TYPE_CHECKING:
    from cli_sdk.client.async_client import AsyncCLIClient
    from cli_sdk.client.sync_client import CLIClient

MONITOR_TYPES = ("coverage", "risk", "fingerprint")


def path_segment(value: Any) -> str:
    """Percent-encode a profile name or monitor id for use as one URL path segment."""
    return quote(str(value), safe="")


def _monitors_path(profile: str, monitor_id: Optional[str] = None) -> str:
    """The monitors collection of ``profile``, or one monitor's alerts feed."""
    path = f"/calibration-profiles/{path_segment(profile)}/monitors"
    return path if monitor_id is None else f"{path}/{path_segment(monitor_id)}/alerts"


@dataclass(frozen=True)
class Monitor:
    id: str
    profile: str
    type: str
    target: Optional[float] = None
    false_alarm_rate: Optional[float] = None
    labelled_sample_rate: Optional[float] = None
    status: str = "active"

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "Monitor":
        return cls(
            id=str(data.get("id", "")),
            profile=data.get("profile", ""),
            type=data.get("type", "coverage"),
            target=data.get("target"),
            false_alarm_rate=data.get("false_alarm_rate"),
            labelled_sample_rate=data.get("labelled_sample_rate"),
            status=data.get("status", "active"),
        )


def _monitor_body(
    type: str,
    target: Optional[float],
    false_alarm_rate: float,
    labelled_sample_rate: Optional[float],
    use_judge_pseudo_labels: bool,
) -> dict[str, Any]:
    if type not in MONITOR_TYPES:
        raise ConfigurationError(f"type must be one of {MONITOR_TYPES}")
    if type != "fingerprint" and target is None:
        raise ConfigurationError(f"a {type} monitor requires `target`")
    if not (0 < false_alarm_rate < 1):
        raise ConfigurationError("false_alarm_rate must be in (0, 1)")
    body: dict[str, Any] = {"type": type, "false_alarm_rate": false_alarm_rate}
    if target is not None:
        body["target"] = target
    if labelled_sample_rate is not None:
        body["labelled_sample_rate"] = labelled_sample_rate
    if use_judge_pseudo_labels:
        body["use_judge_pseudo_labels"] = True
    return body


class Monitors:
    """Hosted monitors (sync)."""

    def __init__(self, client: "CLIClient") -> None:
        self._client = client

    def create(
        self,
        profile: str,
        type: str = "coverage",
        target: Optional[float] = None,
        false_alarm_rate: float = 0.05,
        labelled_sample_rate: Optional[float] = None,
        use_judge_pseudo_labels: bool = False,
    ) -> Monitor:
        body = _monitor_body(type, target, false_alarm_rate, labelled_sample_rate, use_judge_pseudo_labels)
        data = self._client._request("POST", _monitors_path(profile), json=body)
        return Monitor.from_payload(data)

    def list(self, profile: str) -> list[Monitor]:
        data = self._client._request("GET", _monitors_path(profile))
        return [Monitor.from_payload(m) for m in data.get("monitors", [])]

    def poll(self, profile: str, monitor_id: Optional[str] = None) -> Iterator[Alert]:
        """Yield alerts raised since the last poll, across all monitors on ``profile``."""
        monitors = [monitor_id] if monitor_id else [m.id for m in self.list(profile)]
        for mid in monitors:
            data = self._client._request("GET", _monitors_path(profile, mid))
            for raw in data.get("alerts", []):
                yield Alert.from_payload({"profile": profile, **raw})


class AsyncMonitors:
    """Hosted monitors (async)."""

    def __init__(self, client: "AsyncCLIClient") -> None:
        self._client = client

    async def create(
        self,
        profile: str,
        type: str = "coverage",
        target: Optional[float] = None,
        false_alarm_rate: float = 0.05,
        labelled_sample_rate: Optional[float] = None,
        use_judge_pseudo_labels: bool = False,
    ) -> Monitor:
        body = _monitor_body(type, target, false_alarm_rate, labelled_sample_rate, use_judge_pseudo_labels)
        data = await self._client._request("POST", _monitors_path(profile), json=body)
        return Monitor.from_payload(data)

    async def list(self, profile: str) -> list[Monitor]:
        data = await self._client._request("GET", _monitors_path(profile))
        return [Monitor.from_payload(m) for m in data.get("monitors", [])]

    async def poll(self, profile: str, monitor_id: Optional[str] = None) -> list[Alert]:
        monitors = [monitor_id] if monitor_id else [m.id for m in await self.list(profile)]
        alerts: list[Alert] = []
        for mid in monitors:
            data = await self._client._request("GET", _monitors_path(profile, mid))
            alerts.extend(Alert.from_payload({"profile": profile, **raw}) for raw in data.get("alerts", []))
        return alerts


class LocalMonitor:
    """The hosted monitor's e-process, run in-process with no network access.

    Feed it one production outcome at a time: ``covered`` for a coverage
    monitor, ``is_loss`` for a risk monitor. Returns an ``Alert`` the first
    time the evidence crosses ``1 / false_alarm_rate``, else ``None``.
    """

    def __init__(self, type: str, target: float, false_alarm_rate: float = 0.05,
                 profile: Optional[str] = None) -> None:
        if type == "coverage":
            self._engine: Any = CoverageMonitor(target=target, false_alarm_rate=false_alarm_rate)
        elif type == "risk":
            self._engine = RiskMonitor(target=target, false_alarm_rate=false_alarm_rate)
        else:
            raise ValueError("LocalMonitor supports type='coverage' or type='risk'")
        self.type = type
        self.profile = profile
        self._fired = False

    def update(self, outcome: bool) -> Optional[Alert]:
        if self.type == "coverage":
            raw = self._engine.update(covered=outcome)
        else:
            raw = self._engine.update(is_loss=outcome)
        if raw and not self._fired:
            self._fired = True
            return Alert.from_payload({"profile": self.profile, **raw})
        return None
