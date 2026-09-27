"""Google Agent Development Kit (ADK 2.x): calibrated tool guard with native tool confirmation.

::

    from google.adk.agents import Agent
    from cli_sdk.integrations import GuardRule, ToolGuard
    from cli_sdk.integrations.google_adk import cli_before_tool_callback

    guard = ToolGuard(client, [GuardRule(tool="place_account_hold", query=hold_gate, context=case_context)])
    agent = Agent(name="aml_investigator", model=model, tools=[place_account_hold],
                  before_tool_callback=cli_before_tool_callback(guard))

Decisions map onto ADK like this:

allow     the callback returns ``None``, so the tool runs
block     the callback returns a result dict, so the tool is skipped and the
          model sees why
escalate  ``tool_context.request_confirmation(...)`` asks for a human; ADK
          emits an ``adk_request_confirmation`` function call and ends the
          turn. Resume on the same session with ``confirmation_response(...)``;
          ADK re-runs the original call and this callback runs it only if the
          reviewer confirmed

For every agent and sub-agent at once, register ``CLIGuardPlugin(guard)``
on the ``App`` instead of the per-agent callback:
``Runner(app=App(name=..., root_agent=agent, plugins=[CLIGuardPlugin(guard)]), ...)``.
(``Runner(plugins=...)`` still works in ADK 2.x but is deprecated.)

Every decision is appended to ``session.state["cli_guard_audit"]``.

Tool confirmation is marked experimental in ADK 2.x and is supported with
``InMemorySessionService``; check ADK's documentation for other session
services. Requires ``pip install "cci-sdk[google-adk]"``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

try:
    from google.adk.plugins import BasePlugin
    from google.genai import types
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError('cli_sdk.integrations.google_adk needs Google ADK: pip install "cci-sdk[google-adk]"') from exc

from cli_sdk.integrations._guard import GuardDecision, ToolGuard, _jsonable

REQUEST_CONFIRMATION = "adk_request_confirmation"
AUDIT_KEY = "cli_guard_audit"


def review_hint(decision: GuardDecision) -> str:
    hint = f"{decision.tool} needs review: {decision.reason}"
    return f"{hint} Guarantee: {decision.guarantee}" if decision.guarantee else hint


def _audit(tool_context: Any, key: Optional[str], entry: dict[str, Any]) -> None:
    if not key:
        return
    state = tool_context.state
    history = list(state.get(key) or [])
    history.append(_jsonable(entry))
    state[key] = history  # assignment, so ADK records it in the event's state delta


async def _guarded(guard: ToolGuard, tool_name: str, args: dict[str, Any], tool_context: Any,
                   audit_key: Optional[str]) -> Optional[dict[str, Any]]:
    if not guard.guards(tool_name):
        return None
    confirmation = getattr(tool_context, "tool_confirmation", None)
    if confirmation is not None:
        # ADK is re-running a call a human has answered; ADK has already
        # checked that this call matches the confirmation it requested.
        approved = bool(confirmation.confirmed)
        _audit(tool_context, audit_key, {"tool": tool_name, "arguments": args,
                                         "action": "human_approved" if approved else "human_rejected",
                                         "reviewer_payload": confirmation.payload})
        if approved:
            return None
        return {"status": "rejected_by_reviewer", "tool": tool_name,
                "message": f"A reviewer declined {tool_name}. Do not retry it.",
                "reviewer_payload": _jsonable(confirmation.payload)}

    decision = await guard.acheck(tool_name, args, getattr(tool_context, "state", None))
    _audit(tool_context, audit_key, decision.to_dict())
    if decision.allowed:
        return None
    if decision.blocked:
        return {"status": "blocked", "message": decision.message(), "cli": decision.to_dict()}
    tool_context.request_confirmation(hint=review_hint(decision), payload=decision.to_dict())
    # End the turn here; otherwise the model is called again to narrate the pause.
    tool_context.actions.skip_summarization = True
    return {"status": "pending_human_review", "message": decision.message(), "cli": decision.to_dict()}


def cli_before_tool_callback(guard: ToolGuard, *, audit_key: Optional[str] = AUDIT_KEY):
    """An agent-level ``before_tool_callback`` enforcing ``guard``."""

    async def before_tool_callback(tool: Any, args: dict[str, Any], tool_context: Any) -> Optional[dict[str, Any]]:
        return await _guarded(guard, tool.name, dict(args or {}), tool_context, audit_key)

    return before_tool_callback


class CLIGuardPlugin(BasePlugin):
    """App-wide guard for every agent and sub-agent: pass it in ``App(plugins=[...])``."""

    def __init__(self, guard: ToolGuard, *, name: str = "cli_guard", audit_key: Optional[str] = AUDIT_KEY) -> None:
        super().__init__(name=name)
        self.guard = guard
        self.audit_key = audit_key

    async def before_tool_callback(self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any) -> Optional[dict]:
        return await _guarded(self.guard, tool.name, dict(tool_args or {}), tool_context, self.audit_key)


def confirmation_requests(events: Iterable[Any]) -> list[dict[str, Any]]:
    """Pending reviews among a turn's events: ``[{"id", "tool", "arguments", "hint", "cli"}]``.

    ``id`` is the id of the ``adk_request_confirmation`` call, which is what
    ``confirmation_response`` needs, not the original tool call id.
    """
    out = []
    for event in events:
        for call in (event.get_function_calls() if hasattr(event, "get_function_calls") else []) or []:
            if call.name != REQUEST_CONFIRMATION:
                continue
            args = call.args or {}
            original = args.get("originalFunctionCall") or {}
            confirmation = args.get("toolConfirmation") or {}
            out.append({
                "id": call.id,
                "tool": original.get("name"),
                "arguments": original.get("args"),
                "hint": confirmation.get("hint"),
                "cli": confirmation.get("payload"),
            })
    return out


def confirmation_response(request_id: str, approved: bool, payload: Any = None) -> "types.Content":
    """The message that answers a review; send it as ``new_message`` on the same session."""
    return types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
        id=request_id, name=REQUEST_CONFIRMATION, response={"confirmed": bool(approved), "payload": payload}))])


__all__ = ["cli_before_tool_callback", "CLIGuardPlugin", "confirmation_requests", "confirmation_response",
           "review_hint", "REQUEST_CONFIRMATION", "AUDIT_KEY"]
