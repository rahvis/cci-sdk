"""Coverage and risk audits, computable locally with no network access.

The hosted ``POST /v1/calibration-profiles/{name}/audit`` endpoint returns
the same ``AuditResult`` shape; ``audit_coverage`` lets you run the exact
check yourself on a fresh labelled sample (for instance in CI).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from cli_sdk.stats.conformal._quantile import _beta_ppf


@dataclass(frozen=True)
class AuditResult:
    result: str  # "pass" | "fail"
    realized_coverage: float
    ci_lower: float
    ci_upper: float
    sample: int
    target: float
    date: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.result == "pass"

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "AuditResult":
        ci = data.get("ci") or data.get("realized_coverage_ci") or [0.0, 1.0]
        return cls(
            result=data.get("result", "fail"),
            realized_coverage=float(data.get("realized_coverage", 0.0)),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            sample=int(data.get("sample", 0)),
            target=float(data.get("target", 0.0)),
            date=data.get("date"),
        )


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial confidence interval for a success rate."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    tail = (1 - confidence) / 2
    lower = 0.0 if successes == 0 else _beta_ppf(tail, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else _beta_ppf(1 - tail, successes + 1, trials - successes)
    return lower, upper


def audit_coverage(
    covered: Sequence[bool], target: float, confidence: float = 0.95
) -> AuditResult:
    """Audit realized coverage on a fresh labelled sample.

    Fails only when the whole confidence interval sits below ``target``,
    i.e. when the sample gives real evidence of under-coverage, not merely
    because a finite sample landed a little under the target by chance.
    """
    trials = len(covered)
    successes = int(sum(bool(c) for c in covered))
    lower, upper = clopper_pearson(successes, trials, confidence)
    realized = successes / trials
    return AuditResult(
        result="fail" if upper < target else "pass",
        realized_coverage=realized,
        ci_lower=lower,
        ci_upper=upper,
        sample=trials,
        target=target,
    )


def audit_risk(losses: Sequence[float], target: float, confidence: float = 0.95) -> AuditResult:
    """Audit realized risk (0/1 losses) on a fresh labelled sample.

    Fails only when the whole confidence interval sits above ``target``.
    ``realized_coverage`` here holds 1 - risk so the result shape matches
    ``audit_coverage``.
    """
    trials = len(losses)
    bad = int(sum(1 for loss in losses if loss))
    lower, upper = clopper_pearson(bad, trials, confidence)
    return AuditResult(
        result="fail" if lower > target else "pass",
        realized_coverage=1 - bad / trials,
        ci_lower=1 - upper,
        ci_upper=1 - lower,
        sample=trials,
        target=1 - target,
    )
