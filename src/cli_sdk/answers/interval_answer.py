from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from cli_sdk.answers.base import Answer, _pair
from cli_sdk.answers.guarantee import Guarantee


@dataclass(frozen=True)
class IntervalAnswer(Answer):
    """A conformalized ordinal or continuous interval."""

    type: ClassVar[str] = "interval"
    point_estimate: float = 0.0
    interval: tuple[float, float] = (0.0, 0.0)
    legend: dict[str, str] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.interval[1] - self.interval[0]

    def contains(self, value: float) -> bool:
        return self.interval[0] <= value <= self.interval[1]

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "IntervalAnswer":
        return cls(
            guarantee=Guarantee.from_payload(data.get("guarantee")),
            raw=data,
            point_estimate=float(data.get("point_estimate", 0.0)),
            interval=_pair(data.get("interval")) or (0.0, 0.0),
            legend=dict(data.get("legend") or {}),
        )
