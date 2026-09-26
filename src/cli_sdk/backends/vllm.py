from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cli_sdk.backends.base import RemoteBackend
from cli_sdk.exceptions import ConfigurationError


@dataclass
class VLLMBackend(RemoteBackend):
    """A self-hosted vLLM server, reachable by the CLI service.

    Reaches up to L4: ``logprob_token_ids`` for exact label probabilities,
    ``prompt_logprobs`` for scoring supplied text, and hidden-state
    extraction for probe scores. Run the server with
    ``VLLM_BATCH_INVARIANT=1`` for bitwise-reproducible calibration, and
    record ``engine_version`` and ``quantization``: both are part of the
    backend fingerprint.
    """

    provider: str = field(default="vllm", init=False)
    base_url: str = ""
    engine_version: Optional[str] = None
    quantization: Optional[str] = None
    logprobs_mode: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.base_url:
            raise ConfigurationError("VLLMBackend requires `base_url`")
