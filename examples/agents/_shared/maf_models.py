"""Microsoft Agent Framework chat clients for the agent_framework examples, one per provider.

The agent model only has to call tools well. The calibrated guard scores
each proposed action with its own evidence model
(``providers.evidence_backend``), which can be a different provider.

=========  ====================================================  ===============================
provider   agent chat client                                     package
=========  ====================================================  ===============================
mock       ``ScriptedChatClient`` (below; keyless and offline)   agent-framework-core
openai     ``OpenAIChatClient`` (Responses API)                  agent-framework-openai
azure      ``OpenAIChatCompletionClient(azure_endpoint=...)``    agent-framework-openai
anthropic  ``AnthropicClient``                                   agent-framework-anthropic (beta)
gemini     ``GeminiChatClient``                                  agent-framework-gemini (beta)
vllm       ``OpenAIChatCompletionClient(base_url=...)``          agent-framework-openai
sglang     ``OpenAIChatCompletionClient(base_url=...)``          agent-framework-openai
=========  ====================================================  ===============================

Self-hosted servers are reached through the Chat Completions client:
``OpenAIChatClient`` calls the Responses API (``/v1/responses``), which vLLM
and SGLang do not serve. Start the server with tool calling switched on, or
the model answers in plain text, never calls a tool, and the guard never
runs::

    vllm serve google/gemma-4-12B-it --max-logprobs 20 --generation-config vllm \\
        --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \\
        --chat-template examples/tool_chat_template_gemma4.jinja
    python -m sglang.launch_server --model-path google/gemma-4-12B-it --port 30000 \\
        --tool-call-parser gemma4 --reasoning-parser gemma4

The model name must equal the name the server serves (the model path unless
``--served-model-name`` is set).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterable, Mapping, Sequence
from typing import Any, Callable, ClassVar, Optional

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionInvocationLayer,
    Message,
    ResponseStream,
)
from agent_framework.observability import ChatTelemetryLayer

from _shared.providers import settings

# (tool name, arguments) the scripted model calls for a request, or None to answer in text.
Planner = Callable[[str], Optional[tuple[str, dict[str, Any]]]]

REJECTED_BY_USER = "rejected by user"

CONNECTORS = {
    "openai": "agent-framework-openai",
    "azure": "agent-framework-openai",
    "anthropic": "agent-framework-anthropic",
    "gemini": "agent-framework-gemini",
    "vllm": "agent-framework-openai",
    "sglang": "agent-framework-openai",
}


def chat_client(provider: str, *, planner: Optional[Planner] = None) -> Any:
    """The agent's chat client for ``provider``. ``planner`` drives ``--provider mock``.

    Real providers stop with a clear message while their key is a placeholder,
    before any connector is imported or any request is sent.
    """
    if provider == "mock":
        if planner is None:
            raise ValueError("--provider mock needs a planner for the scripted model")
        return ScriptedChatClient(planner=planner)
    s = settings(provider).require_key()
    try:
        if provider == "openai":
            from agent_framework.openai import OpenAIChatClient

            return OpenAIChatClient(model=s.model, api_key=s.api_key, base_url=s.base_url)
        if provider == "azure":
            from agent_framework.openai import OpenAIChatCompletionClient

            # model is the deployment name. For Entra ID, pass credential=... instead of api_key.
            return OpenAIChatCompletionClient(model=s.model, azure_endpoint=s.base_url, api_key=s.api_key,
                                              api_version=s.api_version)
        if provider == "anthropic":
            from agent_framework.anthropic import AnthropicClient

            return AnthropicClient(model=s.model, api_key=s.api_key)
        if provider == "gemini":
            from agent_framework.gemini import GeminiChatClient

            return GeminiChatClient(model=s.model, api_key=s.api_key)
        if provider in ("vllm", "sglang"):
            from agent_framework.openai import OpenAIChatCompletionClient

            return OpenAIChatCompletionClient(model=s.model, base_url=s.base_url, api_key=s.api_key)
    except ImportError as exc:
        raise SystemExit(f"--provider {provider} needs the Agent Framework connector: "
                         f"pip install {CONNECTORS[provider]} ({exc})") from exc
    raise SystemExit(f"no Agent Framework chat client for provider {provider!r}")


# ---------------------------------------------------------------------------
# Keyless scripted model for --provider mock
# ---------------------------------------------------------------------------


def _latest_turn(messages: Sequence[Message]) -> tuple[str, list[Content]]:
    """The latest user text, and the tool results that arrived after it."""
    request, results = "", []
    for message in messages:
        texts = [c.text for c in message.contents if c.type == "text" and c.text]
        if message.role == "user" and texts:
            request, results = " ".join(texts), []
        results.extend(c for c in message.contents if c.type == "function_result")
    return request, results


def summarize_result(result: Any) -> str:
    """What the scripted model tells the user once its tool call has been answered."""
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    if REJECTED_BY_USER in text:
        return "The reviewer rejected the proposed action, so it was not carried out."
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, Mapping) and payload.get("status") == "not_executed":
        cli = payload.get("cli") if isinstance(payload.get("cli"), Mapping) else {}
        reason = cli.get("reason") or payload.get("message") or "the calibrated check did not allow it."
        return f"The action was not carried out automatically: {reason}"
    return f"Done. {text}"


class _ScriptedCore(BaseChatClient):
    """Calls the tool its planner names for the latest request, then summarizes the tool result.

    It stands in for a real model so the example runs offline; the tool
    loop, the function middleware and the approval flow are the framework's
    own, exactly as with a real provider.
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "scripted-mock"

    def __init__(self, *, planner: Planner, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.planner = planner
        self.model_calls = 0

    def _inner_get_response(self, *, messages: Sequence[Message], stream: bool = False,
                            options: Mapping[str, Any], **kwargs: Any):
        # A plain def (not async def): it returns an awaitable, or a ResponseStream when streaming.
        self.model_calls += 1
        request, results = _latest_turn(messages)
        plan = None if results else self.planner(request)
        if plan is not None:
            name, arguments = plan
            call_id = "call_" + hashlib.sha256(f"{name}:{request}".encode()).hexdigest()[:12]
            reply = Message("assistant", [Content.from_function_call(call_id=call_id, name=name,
                                                                     arguments=arguments)])
            finish = "tool_calls"
        else:
            text = summarize_result(results[-1].result) if results else "There is no case to act on."
            reply = Message("assistant", [text])
            finish = "stop"
        response = ChatResponse(messages=[reply], model="scripted-mock", finish_reason=finish)
        if not stream:
            async def _get() -> ChatResponse:
                return response

            return _get()

        async def _stream() -> AsyncIterable[ChatResponseUpdate]:
            yield ChatResponseUpdate(contents=list(reply.contents), role="assistant",
                                     model="scripted-mock", finish_reason=finish)

        return ResponseStream(_stream(), finalizer=ChatResponse.from_updates)


class ScriptedChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, ChatTelemetryLayer, _ScriptedCore):
    """The scripted model with the framework's tool loop, chat middleware and telemetry layers.

    The layer order matches the framework's own clients. A bare
    ``BaseChatClient`` subclass would return the function call without ever
    running the tool (or its middleware).
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "scripted-mock"
