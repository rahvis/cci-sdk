"""Google ADK 2.x agent models for the ADK examples, one per provider, plus a keyless scripted model.

The agent model only has to call tools well; the calibrated guard scores
each proposed action with its own evidence model (``providers.evidence_backend``),
because ADK's LiteLLM adapter does not pass token log-probabilities through.

==========  ==============================================================
provider    ADK model
==========  ==============================================================
mock        ``ScriptedLlm``: a ``BaseLlm`` subclass, no key, no network
openai      ``LiteLlm("openai/<model>")``
azure       ``LiteLlm("azure/<deployment>", api_base=..., api_version=...)``
anthropic   ``AnthropicLlm(model=...)``, the native Anthropic API client
            (a bare ``"claude-..."`` string would route to Vertex AI)
gemini      ``Gemini(model=...)``, ADK's native Gemini client
vllm        ``LiteLlm("hosted_vllm/<served name>", api_base=...)``
sglang      ``LiteLlm("openai/<served name>", api_base=..., api_key=...)``
            (LiteLLM has no ``sglang/`` provider; SGLang speaks the OpenAI API)
==========  ==============================================================

``pip install "google-adk[extensions]"`` adds LiteLLM and the Anthropic SDK;
plain ``google-adk`` is enough for Gemini and the mock.

Self-hosted servers must be started with tool calling switched on, or the
agent never calls a tool and the guard never runs::

    vllm serve google/gemma-4-12B-it --enable-auto-tool-choice --tool-call-parser gemma4
    python -m sglang.launch_server --model-path google/gemma-4-12B-it --port 30000 \\
        --tool-call-parser gemma4

The name after ``hosted_vllm/`` or ``openai/`` must equal the server's served
model name (``VLLM_MODEL`` / ``SGLANG_MODEL``).
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, AsyncGenerator, Callable, Iterable, Optional, Sequence

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

from _shared.providers import settings

# ADK 2.x marks tool confirmation (and its JSON-schema function declarations)
# experimental and warns once per process. The examples rely on both
# deliberately, so keep their output clean.
warnings.filterwarnings(
    "ignore", message=r"\[EXPERIMENTAL\] feature FeatureName\.(TOOL_CONFIRMATION|JSON_SCHEMA_FOR_FUNC_DECL)")
# LiteLLM otherwise fetches its model-cost map over the network on import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_USAGE = types.GenerateContentResponseUsageMetadata(prompt_token_count=0, candidates_token_count=0,
                                                    total_token_count=0)

# A scripted policy maps the conversation so far to the model's next part:
# a function call (``call(...)``) or a final answer (``say(...)``).
Policy = Callable[[Sequence[types.Content]], types.Part]


class ScriptedLlm(BaseLlm):
    """Keyless stand-in for the agent model, driven by a deterministic policy.

    It reads the conversation the way a model would (the user's request and
    the tool results) and never sees anything else, in particular not the
    calibration labels. Pass it as ``Agent(model=ScriptedLlm(policy=...))``.
    """

    model: str = "scripted-agent"
    policy: Any = None

    async def generate_content_async(self, llm_request: LlmRequest,
                                     stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        part = self.policy(list(llm_request.contents or []))
        yield LlmResponse(content=types.Content(role="model", parts=[part]), turn_complete=True,
                          usage_metadata=_USAGE)


def call(name: str, **args: Any) -> types.Part:
    """A function-call part (ADK assigns the call id)."""
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def say(text: str) -> types.Part:
    return types.Part(text=text)


def last_tool_result(history: Sequence[types.Content]) -> Optional[tuple[str, dict[str, Any]]]:
    """``(tool name, result)`` when the latest message is a tool result, else ``None``."""
    if not history:
        return None
    for part in reversed(history[-1].parts or []):
        if part.function_response is not None:
            return part.function_response.name, dict(part.function_response.response or {})
    return None


def last_user_text(history: Sequence[types.Content]) -> str:
    for content in reversed(history):
        if content.role == "user":
            texts = [p.text for p in (content.parts or []) if p.text]
            if texts:
                return " ".join(texts)
    return ""


def agent_model(provider: str, *, policy: Optional[Policy] = None) -> Any:
    """The ADK model for ``provider``. Real providers stop here, before any network call, until a key is set."""
    if provider == "mock":
        if policy is None:
            raise ValueError("the mock agent model needs a scripted policy")
        return ScriptedLlm(policy=policy)
    s = settings(provider).require_key()
    if provider == "gemini":
        from google.adk.models.google_llm import Gemini

        return Gemini(model=s.model, client_kwargs={"api_key": s.api_key})
    if provider == "anthropic":
        try:
            from anthropic import AsyncAnthropic
            from google.adk.models.anthropic_llm import AnthropicLlm
        except ImportError as exc:
            raise SystemExit('--provider anthropic needs the Anthropic SDK: '
                             'pip install "google-adk[extensions]"') from exc
        return AnthropicLlm(model=s.model, client=AsyncAnthropic(api_key=s.api_key))
    try:
        from google.adk.models.lite_llm import LiteLlm
    except ImportError as exc:
        raise SystemExit(f'--provider {provider} needs LiteLLM: pip install "google-adk[extensions]"') from exc
    if provider == "openai":
        extra = {"api_base": s.base_url} if s.base_url else {}
        return LiteLlm(model=f"openai/{s.model}", api_key=s.api_key, **extra)
    if provider == "azure":
        return LiteLlm(model=f"azure/{s.model}", api_base=s.base_url, api_version=s.api_version,
                       api_key=s.api_key)
    if provider == "vllm":
        return LiteLlm(model=f"hosted_vllm/{s.model}", api_base=s.base_url, api_key=s.api_key)
    if provider == "sglang":
        return LiteLlm(model=f"openai/{s.model}", api_base=s.base_url, api_key=s.api_key)
    raise SystemExit(f"no ADK model for provider {provider!r}")


# ---------------------------------------------------------------------------
# running turns and reading events
# ---------------------------------------------------------------------------


def user_message(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


async def run_turn(runner: Any, *, user_id: str, session_id: str, message: types.Content) -> list[Any]:
    """Run one turn and return all its events (a paused turn ends with a confirmation request)."""
    return [event async for event in runner.run_async(user_id=user_id, session_id=session_id,
                                                       new_message=message)]


def function_calls(events: Iterable[Any]) -> list[Any]:
    """Every function call the agent made in these events, in order."""
    out = []
    for event in events:
        out.extend(event.get_function_calls() or [])
    return out


def final_text(events: Iterable[Any]) -> str:
    """The agent's last text reply among these events ('' when the turn paused for review)."""
    text = ""
    for event in events:
        for part in (event.content.parts if event.content else None) or []:
            if part.text:
                text = part.text
    return text


def format_call(call_: Any) -> str:
    args = ", ".join(f"{k}={json.dumps(v)}" for k, v in (call_.args or {}).items())
    return f"{call_.name}({args})"


_CONFIRMATION = "adk_request_confirmation"
_HUMAN = {"human_approved": "reviewer approved; ADK re-ran the original call",
          "human_rejected": "reviewer rejected the call"}


def print_turn(events: Iterable[Any], *, audit_key: str = "cli_guard_audit", ran_status: str = "ok") -> None:
    """Print one turn in order: the agent's calls, each guard decision, what the tool did, the reply.

    Guard decisions are read from the audit entries the guard writes to
    session state; ADK attaches them to the tool's function-response event.
    ``ran_status`` is the ``status`` a guarded tool returns when it really ran.
    """
    paused = False
    for event in events:
        for fc in event.get_function_calls() or []:
            if fc.name == _CONFIRMATION:
                paused = True
            else:
                _say("agent:", format_call(fc))
        audit = (event.actions.state_delta or {}).get(audit_key) if event.actions else None
        for fr in event.get_function_responses() or []:
            if not audit:
                continue  # an unguarded tool: nothing to report
            entry = audit[-1]
            action = entry.get("action", "")
            if action in _HUMAN:
                _say("guard:", _HUMAN[action])
            else:
                _say("guard:", f"{action.upper()} {entry.get('tool')}: {entry.get('reason')}")
                statement = (entry.get("guarantee") or {}).get("statement")
                if statement:
                    _say("", statement, label="guarantee:")
            status = (fr.response or {}).get("status")
            if status == ran_status:
                _say("tool:", f"{fr.name} ran")
            elif status == "pending_human_review":
                _say("tool:", f"{fr.name} did not run; waiting for review")
            else:
                _say("tool:", f"{fr.name} did not run ({status})")
        if paused and event.get_function_responses():
            _say("ADK:", "adk_request_confirmation issued; the turn ends until a reviewer answers")
            paused = False
    text = final_text(events)
    if text:
        _say("agent:", text)


def _say(tag: str, text: str, *, label: str = "") -> None:
    """Print ``text`` under a 4-space margin and a 7-column tag, wrapped to 78 columns."""
    import textwrap

    first = f"    {tag:<7}{label + ' ' if label else ''}"
    rest = " " * len(first)
    for i, line in enumerate(textwrap.wrap(text, 78 - len(first))):
        print((first if i == 0 else rest) + line)


__all__ = ["ScriptedLlm", "Policy", "call", "say", "last_tool_result", "last_user_text", "agent_model",
           "user_message", "run_turn", "function_calls", "final_text", "format_call", "print_turn"]
