from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from cli_sdk.answers.base import Answer
from cli_sdk.answers.guarantee import Guarantee


@dataclass(frozen=True)
class JudgeAnswer(Answer):
    """An LLM-as-judge verdict with a human-agreement guarantee."""

    type: ClassVar[str] = "judge"
    winner: Optional[str] = None
    escalated_to: Optional[str] = None

    @property
    def needs_human(self) -> bool:
        """True when the cascade ended at the human review queue."""
        return self.escalated_to == "human_queue"

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "JudgeAnswer":
        return cls(
            guarantee=Guarantee.from_payload(data.get("guarantee")),
            raw=data,
            winner=data.get("winner"),
            escalated_to=data.get("escalated_to"),
        )
