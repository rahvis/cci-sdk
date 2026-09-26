from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from cli_sdk.answers.base import Answer, _pair
from cli_sdk.answers.guarantee import Guarantee


@dataclass(frozen=True)
class SetAnswer(Answer):
    """A conformal prediction set over a fixed list of options."""

    type: ClassVar[str] = "set"
    set: list[str] = field(default_factory=list)
    probabilities: dict[str, float] = field(default_factory=dict)
    venn_abers: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def is_singleton(self) -> bool:
        """A single-option set: the calibrated analogue of "confident"."""
        return len(self.set) == 1

    @property
    def is_empty(self) -> bool:
        """An empty set means no option cleared the calibrated threshold; abstain."""
        return len(self.set) == 0

    @property
    def top(self) -> str | None:
        """The highest-probability option inside the set, if any."""
        if not self.set:
            return None
        return max(self.set, key=lambda option: self.probabilities.get(option, 0.0))

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "SetAnswer":
        venn_abers = {k: _pair(v) for k, v in (data.get("venn_abers") or {}).items()}
        return cls(
            guarantee=Guarantee.from_payload(data.get("guarantee")),
            raw=data,
            set=list(data.get("set", [])),
            probabilities={k: float(v) for k, v in (data.get("probabilities") or {}).items()},
            venn_abers={k: v for k, v in venn_abers.items() if v is not None},
        )
