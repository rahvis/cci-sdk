"""Backend adapters for every supported provider and for your own models."""

from cli_sdk.backends.anthropic import AnthropicBackend
from cli_sdk.backends.azure_openai import AzureOpenAIBackend
from cli_sdk.backends.base import (
    ACCESS_HINTS,
    ACCESS_LEVELS,
    BackendLike,
    CustomBackend,
    RemoteBackend,
    access_rank,
    backend_payload,
)
from cli_sdk.backends.bedrock import BedrockBackend
from cli_sdk.backends.gemini import GeminiBackend
from cli_sdk.backends.openai import OpenAIBackend
from cli_sdk.backends.openrouter import OpenRouterBackend
from cli_sdk.backends.sglang import SGLangBackend
from cli_sdk.backends.vllm import VLLMBackend

__all__ = [
    "RemoteBackend",
    "CustomBackend",
    "BackendLike",
    "OpenAIBackend",
    "AzureOpenAIBackend",
    "AnthropicBackend",
    "GeminiBackend",
    "BedrockBackend",
    "OpenRouterBackend",
    "VLLMBackend",
    "SGLangBackend",
    "ACCESS_LEVELS",
    "ACCESS_HINTS",
    "access_rank",
    "backend_payload",
]
