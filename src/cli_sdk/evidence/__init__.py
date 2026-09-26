"""Evidence backends: your own model, producing the numbers CLI calibrates.

=================================  ==========  =================================
Backend                            Access      Model
=================================  ==========  =================================
``OpenAIEvidenceBackend``          L1 / L0     OpenAI
``AzureOpenAIEvidenceBackend``     L1 / L0     Azure OpenAI deployment
``vllm_backend(...)``              L1          open weights on vLLM (e.g. Gemma)
``sglang_backend(...)``            L1          open weights on SGLang
``OpenAICompatibleEvidenceBackend`` L1 / L0    any OpenAI-compatible server
``gemini_backend(...)``            L0          Gemini (OpenAI-compatible endpoint)
``AnthropicEvidenceBackend``       L0          Claude (sampling only)
``LangChainEvidenceBackend``       L0 / L1     any LangChain chat model
``MockEvidenceBackend``            L0 / L1     keyless demo model (tests, docs)
=================================  ==========  =================================

Provider SDKs are optional extras and are imported only when you construct
the corresponding backend.
"""

from cli_sdk.evidence.base import EvidenceBackend, EvidenceError
from cli_sdk.evidence.mock import MockEvidenceBackend

_LAZY = {
    "OpenAICompatibleEvidenceBackend": "cli_sdk.evidence.openai_compatible",
    "OpenAIEvidenceBackend": "cli_sdk.evidence.openai_compatible",
    "AzureOpenAIEvidenceBackend": "cli_sdk.evidence.openai_compatible",
    "vllm_backend": "cli_sdk.evidence.openai_compatible",
    "sglang_backend": "cli_sdk.evidence.openai_compatible",
    "gemini_backend": "cli_sdk.evidence.openai_compatible",
    "AnthropicEvidenceBackend": "cli_sdk.evidence.anthropic",
    "LangChainEvidenceBackend": "cli_sdk.evidence.langchain",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'cli_sdk.evidence' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


__all__ = ["EvidenceBackend", "EvidenceError", "MockEvidenceBackend", *_LAZY]
