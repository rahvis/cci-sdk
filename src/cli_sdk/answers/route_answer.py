from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from cli_sdk.answers.base import Answer
from cli_sdk.answers.guarantee import Guarantee


@dataclass(frozen=True)
class RouteAnswer(Answer):
    """Which backend in a cascade served the request, and at what cost."""

    type: ClassVar[str] = "route"
    served_by: dict[str, Any] = field(default_factory=dict)
    escalated: bool = False
    cost_cents: Optional[float] = None
    output: Any = None

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "RouteAnswer":
        return cls(
            guarantee=Guarantee.from_payload(data.get("guarantee")),
            raw=data,
            served_by=dict(data.get("served_by") or {}),
            escalated=bool(data.get("escalated", False)),
            cost_cents=data.get("cost_cents"),
            output=data.get("output"),
        )
