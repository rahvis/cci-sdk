"""LangChain 1.x: calibrated guardrails for ``create_agent``.

::

    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from cli_sdk.integrations import GuardRule, ToolGuard
    from cli_sdk.integrations.langchain import cli_middleware

    guard = ToolGuard(client, [GuardRule(tool="issue_refund", query=refund_gate, context=refund_context)])
    agent = create_agent(model, tools=[issue_refund], middleware=cli_middleware(guard),
                         checkpointer=InMemorySaver())

``cli_middleware(guard)`` returns two middleware:

1. ``HumanInTheLoopMiddleware`` whose ``when`` predicate reads the guard's
   decision: only escalated calls pause for a reviewer (approve / edit /
   reject), and every gated call of one model turn is batched into one
   interrupt.
2. ``CLIGuardMiddleware``, which scores every guarded tool call **once**,
   right after the model proposes it (``after_model``), and stores the
   decision in the agent's checkpointed state under ``cli_guard``. At
   execution time (``wrap_tool_call``) a blocked call never runs; the agent
   receives an error ``ToolMessage`` explaining why.

Why the decision is stored rather than recomputed: on resume, LangChain
re-evaluates the ``when`` predicate. If a fresh evaluation returned a
different answer (another worker, a restart, a sampling-based score, a
recalibrated profile), the interrupt would not be re-raised, the reviewer's
answer would be ignored, and a call the reviewer rejected could run. Reading
the checkpointed decision makes the resume honour exactly what the reviewer
saw. A call with no stored decision, or whose arguments no longer match it,
is treated as needing review.

With ``human_review=False`` there is no interrupt: escalated calls are
refused with an error ``ToolMessage`` instead (useful for batch jobs).

Requires ``pip install "cci-sdk[langchain]"`` (langchain>=1.0) and, for
human review, a checkpointer plus a ``thread_id`` in the run config (use a
durable checkpointer shared by all workers in production).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Mapping, Optional, Sequence

try:
    from typing import NotRequired
except ImportError:  # Python < 3.11
    from typing_extensions import NotRequired

try:
    from langchain.agents import AgentState
    from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware, InterruptOnConfig
    from langchain_core.messages import AIMessage, ToolMessage
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'cli_sdk.integrations.langchain needs LangChain 1.x: pip install "cci-sdk[langchain]"'
    ) from exc

from cli_sdk.integrations._guard import ALLOW, BLOCK, ESCALATE, GuardDecision, ToolGuard

DEFAULT_DECISIONS = ("approve", "edit", "reject")
STATE_KEY = "cli_guard"


class CLIGuardAgentState(AgentState):
    """Agent state plus the guard's stored decisions: ``tool_call_id -> {"args", "decision"}``."""

    cli_guard: NotRequired[dict[str, dict[str, Any]]]


def _args_key(args: Any) -> str:
    return json.dumps(args or {}, sort_keys=True, default=str)


def stored_decision(state: Any, tool_call: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """The decision recorded for this exact call, or ``None`` if unscored or its arguments changed."""
    records = (state or {}).get(STATE_KEY) if isinstance(state, Mapping) else None
    record = (records or {}).get(tool_call.get("id"))
    if record is None or record.get("args") != _args_key(tool_call.get("args")):
        return None
    return record.get("decision")


def _as_dict(decision: GuardDecision | Mapping[str, Any]) -> Mapping[str, Any]:
    return decision.to_dict() if isinstance(decision, GuardDecision) else decision


def review_card(decision: GuardDecision | Mapping[str, Any]) -> str:
    """Plain-text summary a reviewer sees next to the proposed call."""
    decision = _as_dict(decision)
    lines = [f"Calibrated check: {decision.get('action')}", f"Why: {decision.get('reason')}"]
    statement = (decision.get("guarantee") or {}).get("statement")
    if statement:
        lines.append(f"Guarantee: {statement}")
    lines.append(f"Tool: {decision.get('tool')}")
    lines.append(f"Arguments: {json.dumps(decision.get('arguments', {}), sort_keys=True)}")
    return "\n".join(lines)


def blocked_tool_message(decision: GuardDecision | Mapping[str, Any], tool_call: Mapping[str, Any]) -> ToolMessage:
    """The error result the agent sees instead of running a refused call."""
    decision = _as_dict(decision)
    action = decision.get("action")
    status = "blocked" if action == BLOCK else "needs_human_review"
    tool = tool_call.get("name")
    if action == BLOCK:
        message = (f"The {tool} action was blocked by a calibrated safety check: {decision.get('reason')} "
                   "Do not retry it; explain to the user that it cannot be done automatically.")
    else:
        message = (f"The {tool} action needs human review before it can run: {decision.get('reason')} "
                   "Do not retry it; tell the user it has been sent for review.")
    body = {"status": status, "message": message, "cli": dict(decision)}
    return ToolMessage(content=json.dumps(body, default=str), name=tool, tool_call_id=tool_call["id"], status="error")


class CLIGuardMiddleware(AgentMiddleware):
    """Scores guarded calls once (``after_model``) and enforces the stored decision (``wrap_tool_call``).

    ``block_escalations=True`` refuses escalated calls at execution; set it to
    ``False`` when a ``HumanInTheLoopMiddleware`` in front obtains a human
    decision for them (``cli_middleware`` does this for you).
    """

    state_schema = CLIGuardAgentState

    def __init__(self, guard: ToolGuard, *, block_escalations: bool = True) -> None:
        super().__init__()
        self.guard = guard
        self.block_escalations = block_escalations

    # -- scoring (once per proposed call) ---------------------------------

    def _score(self, state: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return None
        known = dict(state.get(STATE_KEY) or {})
        new = {}
        for call in last.tool_calls:
            if not self.guard.guards(call["name"]):
                continue
            existing = known.get(call["id"])
            if existing is not None and existing.get("args") == _args_key(call.get("args")):
                continue
            decision = self.guard.check(call["name"], call.get("args") or {}, state)
            new[call["id"]] = {"args": _args_key(call.get("args")), "decision": decision.to_dict()}
        return {STATE_KEY: {**known, **new}} if new else None

    def after_model(self, state: Any, runtime: Any) -> Optional[dict[str, Any]]:
        return self._score(state)

    async def aafter_model(self, state: Any, runtime: Any) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(self._score, state)

    # -- enforcement ----------------------------------------------------------

    def _refusal(self, request: Any) -> Optional[ToolMessage]:
        call = request.tool_call
        if not self.guard.guards(call["name"]):
            return None
        state = getattr(request, "state", None)
        decision = stored_decision(state, call)
        if decision is None:
            recorded = ((state or {}).get(STATE_KEY) or {}).get(call.get("id")) if isinstance(state, Mapping) else None
            if recorded is not None and not self.block_escalations:
                # The arguments differ from the scored call: a reviewer edited them
                # through human review, which is an explicit human approval.
                return None
            # Unscored (for example a call injected without a model turn): decide now.
            decision = self.guard.check(call["name"], call.get("args") or {}, state).to_dict()
        action = decision.get("action")
        if action == ALLOW or (action == ESCALATE and not self.block_escalations):
            return None
        return blocked_tool_message(decision, call)

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return self._refusal(request) or handler(request)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        refusal = await asyncio.to_thread(self._refusal, request)
        return refusal or await handler(request)


def cli_middleware(
    guard: ToolGuard,
    *,
    human_review: bool = True,
    allowed_decisions: Sequence[str] = DEFAULT_DECISIONS,
    description_prefix: str = "Calibrated check requires human review",
) -> list[AgentMiddleware]:
    """Middleware for ``create_agent(..., middleware=cli_middleware(guard))``.

    Human review needs a checkpointer and a ``thread_id`` in the run config.
    Resume with ``Command(resume={"decisions": [{"type": "approve"}]})``, one
    decision per pending action, in order.
    """
    scoring = CLIGuardMiddleware(guard, block_escalations=not human_review)
    if not human_review:
        return [scoring]

    def escalates(request: Any) -> bool:
        decision = stored_decision(getattr(request, "state", None), request.tool_call)
        # Unscored or changed calls go to a human: never auto-approve on a miss.
        return decision is None or decision.get("action") == ESCALATE

    def describe(tool_call: dict[str, Any], state: Any, runtime: Any) -> str:
        decision = stored_decision(state, tool_call)
        if decision is None:
            return (f"Calibrated check: no stored decision for this call, so it needs review.\n"
                    f"Tool: {tool_call.get('name')}\nArguments: {_args_key(tool_call.get('args'))}")
        return review_card(decision)

    hitl = HumanInTheLoopMiddleware(
        interrupt_on={
            tool: InterruptOnConfig(allowed_decisions=list(allowed_decisions), description=describe, when=escalates)
            for tool in sorted(guard.guarded_tools)
        },
        description_prefix=description_prefix,
    )
    # after_model hooks run in reverse list order, so the scoring hook (last)
    # records decisions before the human-in-the-loop hook (first) reads them.
    return [hitl, scoring]


def pending_reviews(result: Any) -> list[dict[str, Any]]:
    """Actions waiting for a reviewer in an ``invoke`` result (v1 dict or v2 ``GraphOutput``)."""
    interrupts = getattr(result, "interrupts", None)
    if interrupts is None and isinstance(result, dict):
        interrupts = result.get("__interrupt__", ())
    pending = []
    for interrupt in interrupts or ():
        value = getattr(interrupt, "value", {}) or {}
        for action in value.get("action_requests", []):
            pending.append({"interrupt_id": getattr(interrupt, "id", None), **action})
    return pending


def guard_decisions(state_or_result: Any) -> dict[str, dict[str, Any]]:
    """The stored decisions (``tool_call_id -> decision dict``) from agent state or an invoke result."""
    value = getattr(state_or_result, "value", state_or_result)
    records = (value or {}).get(STATE_KEY) if isinstance(value, Mapping) else None
    return {cid: rec.get("decision", {}) for cid, rec in (records or {}).items()}


__all__ = ["CLIGuardMiddleware", "CLIGuardAgentState", "cli_middleware", "pending_reviews", "guard_decisions",
           "stored_decision", "review_card", "blocked_tool_message", "STATE_KEY"]
