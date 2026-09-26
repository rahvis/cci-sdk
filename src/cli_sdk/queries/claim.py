from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cli_sdk.exceptions import ConfigurationError
from cli_sdk.queries.base import Query, check_probability


@dataclass
class Claim(Query):
    """Decompose long-form output into atomic claims and filter to the
    jointly-supported subset. See apps/docs/pages/primitives/claim.mdx.
    """

    type: str = field(default="claim", init=False)
    instructions: str = ""
    calibration_profile: str = ""
    alpha: Optional[float] = None
    support_source: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ConfigurationError("Claim requires `instructions`")
        if not self.calibration_profile:
            raise ConfigurationError("Claim requires `calibration_profile`")
        check_probability("alpha", self.alpha)
