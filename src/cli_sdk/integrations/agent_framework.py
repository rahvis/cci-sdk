"""Microsoft Agent Framework (Python, 1.x): calibrated function middleware with native approvals.

::

    from agent_framework import Agent
    from cli_sdk.integrations import GuardRule, ToolGuard
    from cli_sdk.integrations.agent_framework import CLIGuardMiddleware, pending_reviews, review_message

    guard = ToolGuard(client, [GuardRule(tool="set_risk_tier", query=tier_set, allow_labels=["A", "B"])])
    agent = Agent(client=chat_client, tools=[set_risk_tier], middleware=[CLIGuardMiddleware(guard)])
    session = agent.create_session()
    result = await agent.run("Underwrite application 2291", session=session)
    for review in pending_reviews(result):
        result = await agent.run(review_message(review, approved=True), session=session)

Decisions map onto the framework like this:

allow     ``await call_next()``, so the tool runs
block     ``context.result`` is set and the tool is skipped; the model sees why
escalate  a ``function_approval_request`` is returned and the run pauses;
          ``result.user_input_requests`` holds it. Answer it on the same
          session and the call runs only if the reviewer approved

Approvals are bound to the request this middleware issued: a ticket with
the call's argument fingerprint is stored in ``session.state`` and checked
on replay, so a forged or altered approval never runs a tool. Escalation
therefore needs an ``AgentSession``. Any failure raises
``MiddlewareFailure`` (fail closed), never a tool error the loop would
continue past.

Requires ``pip install "cci-sdk[agent-framework]"`` (agent-framework-core>=1.19).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping, Optional

try:
    from agent_framework import (
        Content,
        FunctionInvocationContext,
        FunctionMiddleware,
        Message,
        MiddlewareFailure,
        MiddlewareTermination,
    )
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'cli_sdk.integrations.agent_framework needs Microsoft Agent Framework: pip install "cci-sdk[agent-framework]"'
    ) from exc

from cli_sdk.integrations._guard import ToolGuard

STATE_KEY = "cli_guard"


def _fingerprint(name: str, arguments: Mapping[str, Any]) -> str:
    blob = json.dumps([name, dict(arguments)], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class CLIGuardMiddleware(FunctionMiddleware):
    """Function middleware that asks the guard before every guarded tool call."""

    def __init__(self, guard: ToolGuard, *, state_key: str = STATE_KEY) -> None:
        self.guard = guard
        self.state_key = state_key

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Any]) -> None:
        name = context.function.name
        if not self.guard.guards(name):
            await call_next()
            return
        arguments = dict(context.arguments or {})
        fingerprint = _fingerprint(name, arguments)
        session = context.session
        tickets: dict[str, Any] = session.state.setdefault(self.state_key, {}) if session is not None else {}

        approval = context.metadata.get("approval_response")
        if approval is not None:
            # Replay after a human answered: honour only an approval this
            # middleware issued, for exactly these arguments.
            ticket = tickets.pop(approval.id, None)
            if approval.approved and ticket is not None and ticket.get("fingerprint") == fingerprint:
                await call_next()
                return
            raise MiddlewareFailure(f"approval for {name} was not issued by the calibrated check, or its arguments changed")

        try:
            decision = await self.guard.acheck(name, arguments, session.state if session is not None else None)
        except Exception as exc:  # the guard only raises with on_error="raise"
            raise MiddlewareFailure(f"calibrated check failed, refusing to run {name}: {exc}") from exc

        if decision.allowed:
            await call_next()
            return
        if decision.blocked:
            context.result = json.dumps({"status": "not_executed", "message": decision.message(),
                                         "cli": decision.to_dict()})
            return
        if session is None:
            raise MiddlewareFailure(f"{name} needs human review, which requires an AgentSession: "
                                    "run the agent with session=agent.create_session()")
        call_id = context.metadata.get("call_id")
        request_id = context.metadata.get("function_call_occurrence_id", call_id)
        tickets[request_id] = {"fingerprint": fingerprint, "decision": decision.to_dict()}
        context.result = Content.from_function_approval_request(
            id=request_id,
            function_call=Content.from_function_call(call_id=call_id, name=name, arguments=arguments, id=request_id),
            additional_properties={"cli": decision.to_dict()},
        )
        raise MiddlewareTermination(f"{name} escalated to a human reviewer by the calibrated check")


def pending_reviews(result: Any) -> list[dict[str, Any]]:
    """Approval requests in an ``AgentResponse``: ``[{"request", "tool", "arguments", "cli"}]``."""
    out = []
    for request in getattr(result, "user_input_requests", None) or []:
        call = getattr(request, "function_call", None)
        arguments: Any = None
        if call is not None:
            try:
                arguments = call.parse_arguments()
            except Exception:
                arguments = getattr(call, "arguments", None)
        properties = getattr(request, "additional_properties", None) or {}
        out.append({"request": request, "tool": getattr(call, "name", None), "arguments": arguments,
                    "cli": properties.get("cli")})
    return out


def review_message(review: Mapping[str, Any] | Any, approved: bool) -> "Message":
    """The user message that answers one review; pass it to ``agent.run(..., session=session)``."""
    request = review["request"] if isinstance(review, Mapping) else review
    return Message("user", [request.to_function_approval_response(bool(approved))])


__all__ = ["CLIGuardMiddleware", "pending_reviews", "review_message", "STATE_KEY"]
