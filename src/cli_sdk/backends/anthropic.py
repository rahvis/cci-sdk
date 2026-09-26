from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cli_sdk.backends.base import RemoteBackend


@dataclass
class AnthropicBackend(RemoteBackend):
    """Anthropic Claude (Messages API).

    Reaches L0 only: no token log-probabilities and no ``n`` parameter, so
    every score is built from repeated sampling (``sample_count`` calls per
    query). Current models reject non-default sampling temperature, so CLI
    never sets it; hold ``effort`` fixed between calibration and serving
    because it is part of the scoring function.
    """

    provider: str = field(default="anthropic", init=False)
    effort: Optional[str] = None
    sample_count: Optional[int] = None
