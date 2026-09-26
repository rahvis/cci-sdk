"""LangChain chat models for the LangChain and LangGraph examples, one per provider.

The agent model only has to call tools well; the calibrated guard scores
actions with its own evidence model (``providers.evidence_backend``).

Self-hosted models need tool calling switched on in the server, or the agent
never calls a tool (and the guard never runs)::

    vllm serve google/gemma-4-12B-it --enable-auto-tool-choice --tool-call-parser gemma4 \
        --reasoning-parser gemma4 --chat-template examples/tool_chat_template_gemma4.jinja
    python -m sglang.launch_server --model-path google/gemma-4-12B-it \
        --tool-call-parser gemma4 --reasoning-parser gemma4

Install the provider package you use: ``langchain-openai`` (OpenAI, Azure,
vLLM, SGLang), ``langchain-anthropic`` or ``langchain-google-genai``.
"""

from __future__ import annotations

from typing import Any

from _shared.providers import settings


def chat_model(provider: str, **overrides: Any) -> Any:
    """A LangChain chat model for ``provider`` (not used with ``--provider mock``)."""
    s = settings(provider).require_key()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=s.model, api_key=s.api_key, base_url=s.base_url, temperature=0,
                          use_responses_api=False, **overrides)
    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(azure_endpoint=s.base_url, azure_deployment=s.model, api_version=s.api_version,
                               api_key=s.api_key, temperature=0, **overrides)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=s.model, api_key=s.api_key, max_tokens=1024, **overrides)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=s.model, api_key=s.api_key, temperature=0, **overrides)
    if provider in ("vllm", "sglang"):
        from langchain_openai import ChatOpenAI

        # The model name must equal the name the server was started with.
        return ChatOpenAI(model=s.model, base_url=s.base_url, api_key=s.api_key, temperature=0,
                          use_responses_api=False, **overrides)
    raise SystemExit(f"no LangChain chat model for provider {provider!r}")
