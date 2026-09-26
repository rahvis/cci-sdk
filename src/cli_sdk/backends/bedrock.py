from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cli_sdk.backends.base import RemoteBackend


@dataclass
class BedrockBackend(RemoteBackend):
    """AWS Bedrock.

    Natively hosted models reach L0. Models you import through Bedrock
    Custom Model Import (``custom_model_import=True``) can reach L2,
    including scoring of text you supply.
    """

    provider: str = field(default="bedrock", init=False)
    region: Optional[str] = None
    custom_model_import: bool = False
