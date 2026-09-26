from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cli_sdk.backends.base import RemoteBackend
from cli_sdk.exceptions import ConfigurationError


@dataclass
class SGLangBackend(RemoteBackend):
    """A self-hosted SGLang server, reachable by the CLI service.

    Reaches up to L4: ``token_ids_logprob`` for exact label probabilities,
    ``logprob_start_len`` for cheap candidate scoring against a shared
    prefix, and ``return_hidden_states`` for probe scores. Note that
    SGLang returns temperature-scaled log-probabilities by default and
    ignores per-request seeds unless deterministic inference is enabled.
    """

    provider: str = field(default="sglang", init=False)
    base_url: str = ""
    engine_version: Optional[str] = None
    deterministic: Optional[bool] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.base_url:
            raise ConfigurationError("SGLangBackend requires `base_url`")
