from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cli_sdk.exceptions import ConfigurationError
from cli_sdk.queries.base import Instructions, Query, check_probability

_VALID_GUARANTEES = ("risk", "risk_high_probability", "fdr")


@dataclass
class Gate(Query):
    """A risk-controlled accept / escalate / abstain decision.

    See apps/docs/pages/primitives/gate.mdx.
    """

    type: str = field(default="gate", init=False)
    instructions: Instructions = ""
    calibration_profile: str = ""
    guarantee: str = "risk"
    target: float = 0.0
    delta: Optional[float] = None
    loss: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ConfigurationError("Gate requires `instructions`")
        if not self.calibration_profile:
            raise ConfigurationError("Gate requires `calibration_profile`")
        if self.guarantee not in _VALID_GUARANTEES:
            raise ConfigurationError(f"guarantee must be one of {_VALID_GUARANTEES}")
        if not (0 < self.target < 1):
            raise ConfigurationError("target must be in (0, 1)")
        if self.guarantee == "risk_high_probability" and self.delta is None:
            raise ConfigurationError("guarantee='risk_high_probability' requires `delta`")
        check_probability("delta", self.delta)
