from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cli_sdk.backends.base import RemoteBackend


@dataclass
class GeminiBackend(RemoteBackend):
    """Google Gemini API / Vertex AI.

    Log-probabilities are deprecated for Gemini 3.x, so treat 3.x models as
    L0. Use stable model codes, never ``-latest`` aliases.
    """

    provider: str = field(default="gemini", init=False)
    thinking_level: Optional[str] = None
