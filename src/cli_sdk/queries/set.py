from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cli_sdk.exceptions import ConfigurationError
from cli_sdk.queries.base import Instructions, Query, check_probability


@dataclass
class Set(Query):
    """A conformal prediction set over a fixed list of options.

    See apps/docs/pages/primitives/set.mdx.
    """

    type: str = field(default="set", init=False)
    instructions: Instructions = ""
    options: dict[str, str | None] = field(default_factory=dict)
    calibration_profile: str = ""
    alpha: Optional[float] = None
    method: Optional[str] = None  # "LAC" | "APS" | "RAPS"; server default is "APS"
    group_by: Optional[str] = None
    backend_access_hint: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ConfigurationError("Set requires `instructions`")
        if len(self.options) < 2:
            raise ConfigurationError("Set requires at least 2 `options`")
        if not self.calibration_profile:
            raise ConfigurationError("Set requires `calibration_profile`")
        if self.method is not None and self.method not in ("LAC", "APS", "RAPS"):
            raise ConfigurationError("method must be one of 'LAC', 'APS', 'RAPS'")
        check_probability("alpha", self.alpha)
