from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cli_sdk.backends.base import RemoteBackend


@dataclass
class OpenRouterBackend(RemoteBackend):
    """OpenRouter.

    Access level depends on the upstream provider. With ``pin_upstream``
    (the default) CLI sends ``provider.only``, ``allow_fallbacks: false``
    and ``require_parameters: true`` so calibration and serving always hit
    the same upstream; otherwise a silent fallback would change the
    scoring model and break exchangeability.
    """

    provider: str = field(default="openrouter", init=False)
    pin_upstream: bool = True
    upstream: Optional[str] = None
    quantization: Optional[str] = None
