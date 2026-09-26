from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from cli_sdk.answers.base import Answer
from cli_sdk.answers.guarantee import Guarantee


@dataclass(frozen=True)
class DroppedClaim:
    text: str
    reason: str = "unsupported"
    score: float | None = None


@dataclass(frozen=True)
class ClaimAnswer(Answer):
    """Long-form output filtered to the jointly-supported claims."""

    type: ClassVar[str] = "claim"
    retained_claims: list[str] = field(default_factory=list)
    dropped_claims: list[DroppedClaim] = field(default_factory=list)

    @property
    def retention_rate(self) -> float:
        total = len(self.retained_claims) + len(self.dropped_claims)
        return len(self.retained_claims) / total if total else 0.0

    def as_text(self, separator: str = " ") -> str:
        """The retained claims joined back into prose."""
        return separator.join(self.retained_claims)

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "ClaimAnswer":
        dropped = []
        for item in data.get("dropped_claims", []):
            if isinstance(item, str):
                dropped.append(DroppedClaim(text=item))
            else:
                dropped.append(
                    DroppedClaim(
                        text=item.get("text", ""),
                        reason=item.get("reason", "unsupported"),
                        score=item.get("score"),
                    )
                )
        return cls(
            guarantee=Guarantee.from_payload(data.get("guarantee")),
            raw=data,
            retained_claims=list(data.get("retained_claims", [])),
            dropped_claims=dropped,
        )
