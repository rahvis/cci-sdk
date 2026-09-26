"""Calibration profiles: the versioned, auditable data behind every guarantee.

See apps/docs/pages/calibration.mdx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from cli_sdk.backends.base import BackendLike, backend_payload
from cli_sdk.calibration.audit import AuditResult
from cli_sdk.calibration.examples import normalize_examples
from cli_sdk.exceptions import ConfigurationError
from cli_sdk.monitoring.monitor import AsyncMonitors, Monitors, path_segment
from cli_sdk.stats.conformal._quantile import coverage_confidence_interval, minimum_calibration_size

if TYPE_CHECKING:
    from cli_sdk.client.async_client import AsyncCLIClient
    from cli_sdk.client.sync_client import CLIClient

METHODS = ("LAC", "APS", "RAPS", "CQR", "ordinal-aps", "CRC", "RCPS", "LTT",
           "IVAP", "CVAP", "conformal-factuality", "trust-or-escalate", "calibrated-cascade")

# Recommended sizes for a stable guarantee at common alpha levels
# (apps/docs/pages/calibration.mdx, "Sizing").
_RECOMMENDED_N = {0.20: 300, 0.10: 1000, 0.05: 1500, 0.01: 2500}


def recommended_size(alpha: float) -> int:
    """The recommended calibration size for ``alpha`` (nearest tabulated level at or below)."""
    for level in sorted(_RECOMMENDED_N, reverse=True):
        if alpha >= level:
            return _RECOMMENDED_N[level]
    return _RECOMMENDED_N[0.01]


@dataclass(frozen=True)
class GroupStatus:
    n: int
    status: str  # "calibrated" | "underpowered" (falls back to the marginal threshold)


@dataclass(frozen=True)
class CalibrationProfile:
    name: str
    method: str
    alpha: float
    n: int = 0
    version: int = 1
    recommended_n: Optional[int] = None
    minimum_n: Optional[int] = None
    realized_coverage_ci: Optional[tuple[float, float]] = None
    last_audit: Optional[dict[str, Any]] = None
    backend_fingerprint: dict[str, Any] = field(default_factory=dict)
    group_by: Optional[str] = None
    groups: dict[str, GroupStatus] = field(default_factory=dict)
    status: str = "collecting"  # "collecting" | "serving" | "stale"

    @property
    def can_serve_guarantees(self) -> bool:
        """False while n is below the hard minimum 1/alpha - 1."""
        return self.n >= (self.minimum_n or minimum_calibration_size(self.alpha))

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "CalibrationProfile":
        alpha = float(data.get("alpha", 0.1))
        n = int(data.get("n", 0))
        ci = data.get("realized_coverage_ci")
        if ci is None and n > 0:
            ci = coverage_confidence_interval(n, alpha)
        groups = {
            name: GroupStatus(n=int(g.get("n", 0)), status=g.get("status", "calibrated"))
            for name, g in (data.get("groups") or {}).items()
        }
        return cls(
            name=data["name"],
            method=data.get("method", "APS"),
            alpha=alpha,
            n=n,
            version=int(data.get("version", 1)),
            recommended_n=data.get("recommended_n", recommended_size(alpha)),
            minimum_n=data.get("minimum_n", minimum_calibration_size(alpha)),
            realized_coverage_ci=tuple(ci) if ci is not None else None,  # type: ignore[arg-type]
            last_audit=data.get("last_audit"),
            backend_fingerprint=dict(data.get("backend_fingerprint") or {}),
            group_by=data.get("group_by"),
            groups=groups,
            status=data.get("status", "collecting"),
        )


def _profile_path(name: str, suffix: str = "") -> str:
    """``/calibration-profiles/{name}{suffix}``, with ``name`` escaped as one path segment."""
    return f"/calibration-profiles/{path_segment(name)}{suffix}"


def _create_body(
    name: str,
    backend: BackendLike,
    method: str,
    alpha: float,
    group_by: Optional[str],
    prompt_template_hash: Optional[str],
    strict_fingerprint: bool,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ConfigurationError(f"method must be one of {METHODS}")
    if not (0 < alpha < 1):
        raise ConfigurationError("alpha must be in (0, 1)")
    body: dict[str, Any] = {
        "name": name,
        "backend": backend_payload(backend),
        "method": method,
        "alpha": alpha,
        "strict_fingerprint": strict_fingerprint,
    }
    if group_by is not None:
        body["group_by"] = group_by
    if prompt_template_hash is not None:
        body["prompt_template_hash"] = prompt_template_hash
    return body


def _judge_body(
    judge: BackendLike, unlabelled_examples: list[Any], human_labelled_sample_size: int
) -> dict[str, Any]:
    if human_labelled_sample_size < 2:
        raise ConfigurationError("human_labelled_sample_size must be at least 2")
    return {
        "judge": backend_payload(judge),
        "unlabelled_examples": [
            e if isinstance(e, dict) and "context" in e else {"context": e} for e in unlabelled_examples
        ],
        "human_labelled_sample_size": human_labelled_sample_size,
    }


class CalibrationProfiles:
    """``client.calibration_profiles`` (sync)."""

    def __init__(self, client: "CLIClient") -> None:
        self._client = client
        self.monitors = Monitors(client)

    def create(
        self,
        name: str,
        backend: BackendLike,
        method: str = "APS",
        alpha: float = 0.10,
        group_by: Optional[str] = None,
        prompt_template_hash: Optional[str] = None,
        strict_fingerprint: bool = True,
    ) -> CalibrationProfile:
        body = _create_body(name, backend, method, alpha, group_by, prompt_template_hash, strict_fingerprint)
        return CalibrationProfile.from_payload(self._client._request("POST", "/calibration-profiles", json=body))

    def get(self, name: str) -> CalibrationProfile:
        return CalibrationProfile.from_payload(self._client._request("GET", _profile_path(name)))

    def list(self) -> list[CalibrationProfile]:
        data = self._client._request("GET", "/calibration-profiles")
        return [CalibrationProfile.from_payload(p) for p in data.get("profiles", [])]

    def add_examples(self, name: str, examples: list[Any]) -> CalibrationProfile:
        body = {"examples": normalize_examples(examples)}
        return CalibrationProfile.from_payload(
            self._client._request("POST", _profile_path(name, "/examples"), json=body)
        )

    def label_with_judge(
        self,
        name: str,
        judge: BackendLike,
        unlabelled_examples: list[Any],
        human_labelled_sample_size: int = 300,
    ) -> dict[str, Any]:
        """Start label-efficient labelling. Returns a job with the human-labelling
        task list and a judge-quality diagnostic."""
        body = _judge_body(judge, unlabelled_examples, human_labelled_sample_size)
        return self._client._request("POST", _profile_path(name, "/label-with-judge"), json=body)

    def audit(self, name: str, fresh_examples: list[Any]) -> AuditResult:
        body = {"examples": normalize_examples(fresh_examples)}
        return AuditResult.from_payload(self._client._request("POST", _profile_path(name, "/audit"), json=body))


class AsyncCalibrationProfiles:
    """``client.calibration_profiles`` (async)."""

    def __init__(self, client: "AsyncCLIClient") -> None:
        self._client = client
        self.monitors = AsyncMonitors(client)

    async def create(
        self,
        name: str,
        backend: BackendLike,
        method: str = "APS",
        alpha: float = 0.10,
        group_by: Optional[str] = None,
        prompt_template_hash: Optional[str] = None,
        strict_fingerprint: bool = True,
    ) -> CalibrationProfile:
        body = _create_body(name, backend, method, alpha, group_by, prompt_template_hash, strict_fingerprint)
        return CalibrationProfile.from_payload(await self._client._request("POST", "/calibration-profiles", json=body))

    async def get(self, name: str) -> CalibrationProfile:
        return CalibrationProfile.from_payload(await self._client._request("GET", _profile_path(name)))

    async def list(self) -> list[CalibrationProfile]:
        data = await self._client._request("GET", "/calibration-profiles")
        return [CalibrationProfile.from_payload(p) for p in data.get("profiles", [])]

    async def add_examples(self, name: str, examples: list[Any]) -> CalibrationProfile:
        body = {"examples": normalize_examples(examples)}
        return CalibrationProfile.from_payload(
            await self._client._request("POST", _profile_path(name, "/examples"), json=body)
        )

    async def label_with_judge(
        self,
        name: str,
        judge: BackendLike,
        unlabelled_examples: list[Any],
        human_labelled_sample_size: int = 300,
    ) -> dict[str, Any]:
        body = _judge_body(judge, unlabelled_examples, human_labelled_sample_size)
        return await self._client._request("POST", _profile_path(name, "/label-with-judge"), json=body)

    async def audit(self, name: str, fresh_examples: list[Any]) -> AuditResult:
        body = {"examples": normalize_examples(fresh_examples)}
        return AuditResult.from_payload(
            await self._client._request("POST", _profile_path(name, "/audit"), json=body)
        )
