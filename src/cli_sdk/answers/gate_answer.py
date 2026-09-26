from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from cli_sdk.answers.base import Answer
from cli_sdk.answers.guarantee import Guarantee

DECISIONS = ("auto_approve", "escalate", "abstain")


@dataclass(frozen=True)
class GateAnswer(Answer):
    """A risk-controlled accept / escalate / abstain decision."""

    type: ClassVar[str] = "gate"
    decision: str = "escalate"

    @property
    def approved(self) -> bool:
        return self.decision == "auto_approve"

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "GateAnswer":
        decision = data.get("decision", "escalate")
        if decision not in DECISIONS:
            # Fail closed: an unrecognized decision is never treated as approval.
            decision = "escalate"
        return cls(
            guarantee=Guarantee.from_payload(data.get("guarantee")),
            raw=data,
            decision=decision,
        )
