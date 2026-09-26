from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from cli_sdk.exceptions import ConfigurationError
from cli_sdk.queries.base import Query, check_probability

_VALID_GUARANTEES = ("cost_budget", "accuracy")


@dataclass
class Route(Query):
    """A calibrated cascade across model backends, with a cost or
    accuracy guarantee. See apps/docs/pages/primitives/route.mdx.
    """

    type: str = field(default="route", init=False)
    cascade: list[dict[str, Any]] = field(default_factory=list)
    calibration_profile: str = ""
    guarantee: str = "cost_budget"
    target_cents: Optional[float] = None
    alpha: Optional[float] = None
    task: Optional[Query] = None  # the decision each tier makes (a ``Set``); required in local mode

    def __post_init__(self) -> None:
        if len(self.cascade) < 1:
            raise ConfigurationError("Route requires at least one `cascade` stage")
        if not self.calibration_profile:
            raise ConfigurationError("Route requires `calibration_profile`")
        if self.guarantee not in _VALID_GUARANTEES:
            raise ConfigurationError(f"guarantee must be one of {_VALID_GUARANTEES}")
        if self.guarantee == "cost_budget" and self.target_cents is None:
            raise ConfigurationError("guarantee='cost_budget' requires `target_cents`")
        check_probability("alpha", self.alpha)
