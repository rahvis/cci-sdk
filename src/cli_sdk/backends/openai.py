from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cli_sdk.backends.base import RemoteBackend


@dataclass
class OpenAIBackend(RemoteBackend):
    """OpenAI Chat Completions / Responses.

    Reaches L1 (top-20 generated-token log-probabilities) only on
    non-reasoning configurations: ``gpt-4.1`` and models run with
    ``reasoning_effort="none"``. Any other reasoning effort, and GPT-6
    Astra at any effort, reaches L0 and CLI falls back to sampling.
    Prefer dated snapshots (e.g. ``gpt-4.1-2025-04-14``) so a calibration
    profile is never silently served by a different model.
    """

    provider: str = field(default="openai", init=False)
    reasoning_effort: Optional[str] = None
