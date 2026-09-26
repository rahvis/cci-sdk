"""Provider settings shared by every agent example: which model, which key, which server.

Every key is a placeholder. Put your own in environment variables (see
``examples/agents/.env.example``); nothing here is ever sent anywhere except
to the provider you choose. ``--provider mock`` needs no key and no network.

Two models are involved in each example, and they can differ:

- the **agent model** decides which tool to call (built per framework, from
  these settings, in each example file);
- the **evidence model** scores the proposed action so the calibrated guard
  can decide it (``evidence_backend`` below). Use ``--evidence-provider`` to
  pick a different one, for example Claude as the agent and a logprob-capable
  Gemma on vLLM as the evidence model.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from cli_sdk.backends.base import CustomBackend

PROVIDERS = ("mock", "openai", "azure", "anthropic", "gemini", "vllm", "sglang")
PLACEHOLDER_PREFIX = "YOUR_"


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    key_env: Optional[str] = None

    @property
    def is_placeholder(self) -> bool:
        return bool(self.api_key) and self.api_key.startswith(PLACEHOLDER_PREFIX)

    def require_key(self) -> "ProviderSettings":
        """Stop with a clear message when the key is still a placeholder."""
        if self.is_placeholder:
            raise SystemExit(
                f"--provider {self.provider} needs a real key: set {self.key_env} "
                "(see examples/agents/.env.example), or run with --provider mock."
            )
        return self


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def settings(provider: str) -> ProviderSettings:
    """Model, key and endpoint for ``provider``, from the environment with placeholder defaults."""
    if provider == "mock":
        return ProviderSettings("mock", "mock-model")
    if provider == "openai":
        return ProviderSettings(
            "openai",
            # A non-reasoning model returns token log-probabilities (access level L1).
            # Prefer a dated snapshot in production, e.g. gpt-4.1-mini-2025-04-14.
            model=_env("OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=_env("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            key_env="OPENAI_API_KEY",
        )
    if provider == "azure":
        return ProviderSettings(
            "azure",
            model=_env("AZURE_OPENAI_DEPLOYMENT", "YOUR_DEPLOYMENT_NAME"),
            api_key=_env("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY"),
            base_url=_env("AZURE_OPENAI_ENDPOINT", "https://YOUR-RESOURCE.openai.azure.com"),
            api_version=_env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            key_env="AZURE_OPENAI_API_KEY",
        )
    if provider == "anthropic":
        return ProviderSettings(
            "anthropic",
            model=_env("ANTHROPIC_MODEL", "claude-sonnet-5"),
            api_key=_env("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY"),
            key_env="ANTHROPIC_API_KEY",
        )
    if provider == "gemini":
        return ProviderSettings(
            "gemini",
            model=_env("GEMINI_MODEL", "gemini-3.8-flash"),
            api_key=os.environ.get("GOOGLE_API_KEY") or _env("GEMINI_API_KEY", "YOUR_GOOGLE_API_KEY"),
            key_env="GOOGLE_API_KEY",
        )
    if provider == "vllm":
        # vllm serve google/gemma-4-12B-it --max-logprobs 20 --generation-config vllm \
        #   --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
        #   --chat-template examples/tool_chat_template_gemma4.jinja
        return ProviderSettings(
            "vllm",
            model=_env("VLLM_MODEL", "google/gemma-4-12B-it"),
            api_key=_env("VLLM_API_KEY", "EMPTY"),
            base_url=_env("VLLM_BASE_URL", "http://localhost:8000/v1"),
            key_env="VLLM_API_KEY",
        )
    if provider == "sglang":
        # python -m sglang.launch_server --model-path google/gemma-4-12B-it --port 30000 \
        #   --tool-call-parser gemma4 --reasoning-parser gemma4
        return ProviderSettings(
            "sglang",
            model=_env("SGLANG_MODEL", "google/gemma-4-12B-it"),
            api_key=_env("SGLANG_API_KEY", "EMPTY"),
            base_url=_env("SGLANG_BASE_URL", "http://localhost:30000/v1"),
            key_env="SGLANG_API_KEY",
        )
    raise SystemExit(f"unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}")


def evidence_backend(provider: str, *, mock_scorer: Optional[Callable[..., Any]] = None) -> CustomBackend:
    """The model that scores proposed actions for the calibrated guard."""
    s = settings(provider)
    if provider == "mock":
        from cli_sdk.evidence import MockEvidenceBackend

        return MockEvidenceBackend(scorer=mock_scorer, name="mock")
    s.require_key()
    if provider == "openai":
        from cli_sdk.evidence import OpenAIEvidenceBackend

        return OpenAIEvidenceBackend(s.model, api_key=s.api_key, base_url=s.base_url)
    if provider == "azure":
        from cli_sdk.evidence import AzureOpenAIEvidenceBackend

        return AzureOpenAIEvidenceBackend(s.model, azure_endpoint=s.base_url, api_key=s.api_key,
                                          api_version=s.api_version)
    if provider == "anthropic":
        from cli_sdk.evidence import AnthropicEvidenceBackend

        return AnthropicEvidenceBackend(s.model, api_key=s.api_key)
    if provider == "gemini":
        from cli_sdk.evidence import gemini_backend

        return gemini_backend(s.model, api_key=s.api_key)
    if provider == "vllm":
        from cli_sdk.evidence import vllm_backend

        return vllm_backend(s.model, base_url=s.base_url, api_key=s.api_key)
    if provider == "sglang":
        from cli_sdk.evidence import sglang_backend

        return sglang_backend(s.model, base_url=s.base_url, api_key=s.api_key)
    raise SystemExit(f"unknown provider {provider!r}")


def sample_count(provider: str) -> int:
    """Samples per score for sampling-only (L0) evidence models; unused at L1."""
    return 8 if provider in ("anthropic", "gemini") else 20


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--provider", choices=PROVIDERS, default="mock",
                        help="model provider for the agent (default: mock, keyless and offline)")
    parser.add_argument("--evidence-provider", choices=PROVIDERS, default=None,
                        help="model that scores actions for the guard (default: same as --provider)")
    parser.add_argument("--store", default=None,
                        help="directory for calibration profiles (default: .cli_profiles next to the example)")
    parser.add_argument("--recalibrate", action="store_true", help="rebuild calibration profiles from the data")
    parser.add_argument("--interactive", action="store_true",
                        help="ask you to approve or reject escalations instead of using the scripted reviewer")
    return parser
