from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cli_sdk.exceptions import ConfigurationError
from cli_sdk.queries.base import Instructions, Query


@dataclass
class Belief(Query):
    """A single-statement probability, returned as a Venn-Abers calibrated
    interval. See apps/docs/pages/primitives/belief.mdx.
    """

    type: str = field(default="belief", init=False)
    instructions: Instructions = ""
    calibration_profile: str = ""
    criteria: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ConfigurationError("Belief requires `instructions`")
        if not self.calibration_profile:
            raise ConfigurationError("Belief requires `calibration_profile`")
