"""Shared base for typed answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from cli_sdk.answers.guarantee import Guarantee


@dataclass(frozen=True)
class Answer:
    """Base class for every answer type. ``raw`` keeps the untouched payload."""

    type: ClassVar[str] = ""
    guarantee: Guarantee
    raw: dict[str, Any]

    @property
    def is_heuristic(self) -> bool:
        return self.guarantee.is_heuristic


def _pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    lo, hi = value
    return (float(lo), float(hi))
