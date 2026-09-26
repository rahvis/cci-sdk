"""The guarantee card attached to every CLI answer.

See apps/docs/pages/guarantees.mdx for the canonical meaning of each field.
Every guarantee is marginal or group-conditional over data exchangeable
with the named calibration profile; none is a statement about one specific
decision in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from cli_sdk.constants import (
    GUARANTEE_ANYTIME,
    GUARANTEE_CALIBRATION,
    GUARANTEE_COST_BUDGET,
    GUARANTEE_COVERAGE,
    GUARANTEE_FDR,
    GUARANTEE_HEURISTIC,
    GUARANTEE_RISK,
    GUARANTEE_RISK_HIGH_PROBABILITY,
)

GUARANTEE_TYPES = frozenset(
    {
        GUARANTEE_COVERAGE,
        GUARANTEE_CALIBRATION,
        GUARANTEE_RISK,
        GUARANTEE_RISK_HIGH_PROBABILITY,
        GUARANTEE_FDR,
        GUARANTEE_ANYTIME,
        GUARANTEE_COST_BUDGET,
        GUARANTEE_HEURISTIC,
    }
)

_KNOWN_FIELDS = (
    "type",
    "method",
    "alpha",
    "delta",
    "target",
    "target_cents",
    "realized_upper_bound",
    "statement",
    "calibration_profile",
    "calibration_n",
    "coverage_ci",
    "last_audited",
    "stats_version",
)


@dataclass(frozen=True)
class Guarantee:
    """What was proven about an answer, and under which calibration."""

    type: str
    method: Optional[str] = None
    alpha: Optional[float] = None
    delta: Optional[float] = None
    target: Optional[float] = None
    target_cents: Optional[float] = None
    realized_upper_bound: Optional[float] = None
    statement: Optional[str] = None
    calibration_profile: Optional[str] = None
    calibration_n: Optional[int] = None
    coverage_ci: Optional[tuple[float, float]] = None
    last_audited: Optional[str] = None
    stats_version: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_heuristic(self) -> bool:
        """True when this answer carries no formal statistical guarantee."""
        return self.type == GUARANTEE_HEURISTIC

    @classmethod
    def from_payload(cls, data: dict[str, Any] | None) -> "Guarantee":
        if not data:
            # An answer with no guarantee block is treated as heuristic,
            # never as silently guaranteed.
            return cls(type=GUARANTEE_HEURISTIC)
        guarantee_type = data.get("type", GUARANTEE_HEURISTIC)
        if guarantee_type not in GUARANTEE_TYPES:
            guarantee_type = GUARANTEE_HEURISTIC
        coverage_ci = data.get("coverage_ci")
        return cls(
            type=guarantee_type,
            method=data.get("method"),
            alpha=data.get("alpha"),
            delta=data.get("delta"),
            target=data.get("target"),
            target_cents=data.get("target_cents"),
            realized_upper_bound=data.get("realized_upper_bound"),
            statement=data.get("statement"),
            calibration_profile=data.get("calibration_profile"),
            calibration_n=data.get("calibration_n"),
            coverage_ci=tuple(coverage_ci) if coverage_ci is not None else None,  # type: ignore[arg-type]
            last_audited=data.get("last_audited"),
            stats_version=data.get("stats_version"),
            extra={k: v for k, v in data.items() if k not in _KNOWN_FIELDS},
        )

    def describe(self) -> str:
        """A one-line, plain-language statement of the guarantee.

        Levels are printed without ever rounding up (``alpha=0.005`` reads
        "99.5%", never "100%"), and a formal guarantee whose card omits its
        level is still described as a guarantee, never as heuristic.
        """
        if self.statement and not self.is_heuristic:
            return self.statement
        profile = f" on profile '{self.calibration_profile}'" if self.calibration_profile else ""
        n = f" (n={self.calibration_n})" if self.calibration_n is not None else ""
        # Risk-type cards may carry their level as `target` (Gate, Judge) or
        # as `alpha` (Claim's "at most alpha"); both mean the same bound.
        level = self.target if self.target is not None else self.alpha
        if self.type == GUARANTEE_COVERAGE and self.alpha is not None:
            return f"Contains the correct answer at least {_percent(1 - self.alpha)} of the time{profile}{n}."
        if self.type == GUARANTEE_RISK and level is not None:
            return f"Expected loss at most {level:g}{profile}{n}."
        if self.type == GUARANTEE_RISK_HIGH_PROBABILITY and level is not None:
            conf = _percent(1 - self.delta) if self.delta is not None else "high"
            return f"With {conf} confidence, risk at most {level:g}{profile}{n}."
        if self.type == GUARANTEE_FDR and level is not None:
            if self.delta is not None:
                # Per-request selective form (Learn-then-Test): a high-probability bound.
                conf = _percent(1 - self.delta)
                return (f"With {conf} confidence, the wrong fraction among approved decisions "
                        f"is at most {level:g}{profile}{n}.")
            return f"Expected wrong fraction among approved decisions at most {level:g}{profile}{n}."
        if self.type == GUARANTEE_CALIBRATION:
            return (f"Venn-Abers pair: the probability computed under the true label is calibrated on "
                    f"data exchangeable with the calibration set{profile}{n}; a wide pair means the "
                    "calibration data cannot pin the probability down.")
        if self.type == GUARANTEE_COST_BUDGET and self.target_cents is not None:
            conf = _percent(1 - self.alpha) if self.alpha is not None else "high"
            return f"Cost per request at most {self.target_cents:g} cents with {conf} probability{profile}{n}."
        if self.type == GUARANTEE_ANYTIME:
            return f"False-alarm rate controlled at any stopping time{profile}."
        if self.type in _LABELS:
            method = f" via {self.method}" if self.method else ""
            return f"{_LABELS[self.type]} guarantee{method}{profile}{n}."
        return "Heuristic value: no formal statistical guarantee."


_LABELS = {
    GUARANTEE_COVERAGE: "Coverage",
    GUARANTEE_CALIBRATION: "Calibration",
    GUARANTEE_RISK: "Risk-control",
    GUARANTEE_RISK_HIGH_PROBABILITY: "High-probability risk-control",
    GUARANTEE_FDR: "False-discovery-rate",
    GUARANTEE_COST_BUDGET: "Cost-budget",
}


def _percent(fraction: float) -> str:
    """Format a guaranteed level as a percentage, truncated (never rounded up) to 0.1%."""
    tenths = math.floor(round(fraction * 1000, 9))
    return f"{tenths / 10:g}%"
