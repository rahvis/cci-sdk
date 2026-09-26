from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from cli_sdk.exceptions import ConfigurationError
from cli_sdk.queries.base import Query, check_probability


@dataclass
class Judge(Query):
    """An LLM-as-judge verdict with a calibrated human-agreement guarantee
    and an optional escalation cascade. See apps/docs/pages/primitives/judge.mdx.
    """

    type: str = field(default="judge", init=False)
    instructions: str = ""
    calibration_profile: str = ""
    alpha: Optional[float] = None
    cascade: Optional[list[dict[str, Any]]] = None

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ConfigurationError("Judge requires `instructions`")
        if not self.calibration_profile:
            raise ConfigurationError("Judge requires `calibration_profile`")
        check_probability("alpha", self.alpha)
