from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from cli_sdk.answers.base import Answer, _pair
from cli_sdk.answers.guarantee import Guarantee


@dataclass(frozen=True)
class BeliefAnswer(Answer):
    """A Venn-Abers calibrated probability interval for one statement."""

    type: ClassVar[str] = "belief"
    probability: float = 0.0
    venn_abers: tuple[float, float] | None = None

    @property
    def interval_width(self) -> float | None:
        """Width of [p0, p1]: a meaningful ambiguity signal on its own."""
        if self.venn_abers is None:
            return None
        return self.venn_abers[1] - self.venn_abers[0]

    def straddles(self, threshold: float) -> bool:
        """True when ``p0 < threshold < p1``.

        A decision rule, not a guarantee: act only when the whole pair is on
        one side of the threshold and escalate a straddle, because the
        calibration data near this score cannot settle which side applies.
        The rule has no error-rate guarantee of its own; pair it with a
        ``Gate`` when a stated error rate is required.
        """
        if self.venn_abers is None:
            return False
        return self.venn_abers[0] < threshold < self.venn_abers[1]

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> "BeliefAnswer":
        return cls(
            guarantee=Guarantee.from_payload(data.get("guarantee")),
            raw=data,
            probability=float(data.get("probability", 0.0)),
            venn_abers=_pair(data.get("venn_abers")),
        )
