from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cli_sdk.backends.base import RemoteBackend


@dataclass
class AzureOpenAIBackend(RemoteBackend):
    """Azure OpenAI (Microsoft Foundry).

    Same access levels as OpenAI. Set the deployment's version-upgrade
    policy to ``NoAutoUpgrade`` for any deployment backing a calibration
    profile; CLI records the deployment's reported model version in every
    guarantee card's backend fingerprint.
    """

    provider: str = field(default="azure-openai", init=False)
    deployment: Optional[str] = None
    api_version: Optional[str] = None
    reasoning_effort: Optional[str] = None
