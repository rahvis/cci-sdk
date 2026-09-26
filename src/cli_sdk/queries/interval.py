from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cli_sdk.exceptions import ConfigurationError
from cli_sdk.queries.base import Instructions, Query, check_probability


@dataclass
class Interval(Query):
    """A conformalized ordinal or continuous interval with a coverage guarantee.

    See apps/docs/pages/primitives/interval.mdx.
    """

    type: str = field(default="interval", init=False)
    instructions: Instructions = ""
    levels: list[str] = field(default_factory=list)
    calibration_profile: str = ""
    alpha: Optional[float] = None
    method: Optional[str] = None  # "CQR" | "ordinal-aps"

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ConfigurationError("Interval requires `instructions`")
        if not (2 <= len(self.levels) <= 10):
            raise ConfigurationError("Interval requires between 2 and 10 `levels`")
        if not self.calibration_profile:
            raise ConfigurationError("Interval requires `calibration_profile`")
        if self.method is not None and self.method not in ("CQR", "ordinal-aps"):
            raise ConfigurationError("method must be one of 'CQR', 'ordinal-aps'")
        check_probability("alpha", self.alpha)
