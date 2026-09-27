"""LangGraph: a calibrated guard node between the model and the tools.

::

    from langgraph.graph import StateGraph, START
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import InMemorySaver
    from cli_sdk.integrations.langgraph import CLIGuardState, add_cli_guard

    builder = StateGraph(CLIGuardState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_edge("tools", "agent")
    add_cli_guard(builder, guard)                  # agent -> cli_guard -> tools | review | blocked
    graph = builder.compile(checkpointer=InMemorySaver())

Topology added by ``add_cli_guard``::

    agent --(tool calls)--> cli_guard --allow----> tools
                                      --escalate-> cli_human_review --approve/edit--> tools
                                      |                              --reject-------> agent
                                      --block----> cli_blocked -------------------> agent

The guard runs once per model turn in its own node, so the calibrated
evaluation is never repeated when the review node re-runs on resume. When a
turn has several tool calls, the most conservative decision applies to the
whole turn (block > escalate > allow). Every refused call gets a matching
error ``ToolMessage``, so provider message histories stay valid.

Resume a paused thread with ``graph.invoke(Command(resume={"action": "approve"}), config)``;
``{"action": "edit", "args": {...}}`` (or ``"calls": {call_id: args}``) runs edited
arguments; anything else, including ``{"action": "reject", "note": "..."}``,
refuses the calls. Edited arguments are not re-checked by the guard: the
reviewer's decision is final.

The checkpoint keeps the audit trail in ``state["cli_guard"]``: the turn's
``action``, one entry per ``call`` (the proposed arguments, the reason, the
guarantee card and the evidence), and, after a review, ``review`` with the
``outcome`` (approve, edit or reject), the reviewer's ``note`` and, for an
edit, the ``executed_calls``.

Requires ``pip install "cci-sdk[langgraph]"`` (langgraph>=1.0).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

try:
    from langchain_core.messages import ToolMessage
    from langgraph.graph import END, MessagesState
    from langgraph.prebuilt import tools_condition
    from langgraph.types import Command, interrupt
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError('cli_sdk.integrations.langgraph needs LangGraph 1.x: pip install "cci-sdk[langgraph]"') from exc

from cli_sdk.integrations._guard import ALLOW, BLOCK, ESCALATE, ToolGuard, most_severe

GUARD_KEY = "cli_guard"
GUARD_NODE = "cli_guard"
REVIEW_NODE = "cli_human_review"
BLOCKED_NODE = "cli_blocked"


class CLIGuardState(MessagesState):
    """``MessagesState`` plus the guard's per-turn record (kept in the checkpoint for audit).

    ``cli_guard`` holds the latest guarded turn: ``action``, ``calls`` and,
    once a reviewer has answered, ``review``.
    """

    cli_guard: dict


def guard_node(guard: ToolGuard, *, key: str = GUARD_KEY, messages_key: str = "messages"):
    """A node that decides every tool call in the latest ``AIMessage``."""

    def cli_guard(state: Mapping[str, Any]) -> dict[str, Any]:
        message = state[messages_key][-1]
        calls = list(getattr(message, "tool_calls", None) or [])
        decisions = [guard.check(c["name"], c.get("args") or {}, state) for c in calls]
        return {key: {
            "action": most_severe(decisions),
            "calls": [{"id": c.get("id"), **d.to_dict()} for c, d in zip(calls, decisions)],
        }}

    return cli_guard


def route_after_guard(key: str = GUARD_KEY):
    def route(state: Mapping[str, Any]) -> str:
        return (state.get(key) or {}).get("action", ESCALATE)

    return route


def _refusals(message: Any, record: Mapping[str, Any], status: str, note: Optional[str] = None) -> list[ToolMessage]:
    by_id = {c.get("id"): c for c in record.get("calls", [])}
    out = []
    for call in getattr(message, "tool_calls", None) or []:
        info = by_id.get(call.get("id"), {})
        body = {"status": status, "reason": info.get("reason"), "guarantee": (info.get("guarantee") or {}).get("statement")}
        if note:
            body["reviewer_note"] = note
        body["instruction"] = "This action did not run. Do not retry it; tell the user what happens next."
        out.append(ToolMessage(content=json.dumps(body), tool_call_id=call["id"], name=call.get("name"), status="error"))
    return out


def _reviewed(record: Mapping[str, Any], review: Mapping[str, Any], outcome: str,
              edited: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """The guard record plus what the reviewer decided, so the checkpoint holds the whole audit trail."""
    entry: dict[str, Any] = {"outcome": outcome, "note": review.get("note")}
    if review.get("action") != outcome:
        entry["requested"] = review.get("action")   # an unrecognized answer, refused (fail closed)
    if edited is not None:
        entry["executed_calls"] = [{"id": c.get("id"), "tool": c.get("name"), "arguments": c.get("args") or {}}
                                   for c in edited]
    return {**record, "review": _jsonable(entry)}


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def human_review_node(*, key: str = GUARD_KEY, tools: str = "tools", agent: str = "agent",
                      messages_key: str = "messages"):
    """A node that pauses with ``interrupt()`` and acts on the reviewer's answer.

    Nothing before ``interrupt()`` has side effects, because LangGraph re-runs
    this node from its first line on resume.
    """

    def cli_human_review(state: Mapping[str, Any]) -> Command:
        message = state[messages_key][-1]
        record = state.get(key) or {}
        review = interrupt({
            "kind": "cli_human_review",
            "question": "Approve these actions?",
            "calls": record.get("calls", []),
            "resume_with": {"action": "approve | edit | reject", "args": "edited arguments (edit)",
                            "note": "optional reviewer note"},
        })
        if not isinstance(review, Mapping):
            review = {"action": review}
        action = review.get("action")
        if action == "approve":
            return Command(goto=tools, update={key: _reviewed(record, review, "approve")})
        if action == "edit":
            per_call = review.get("calls") or {}
            shared = review.get("args") or {}
            edited = [{**c, "args": {**(c.get("args") or {}), **shared, **per_call.get(c.get("id"), {})}}
                      for c in message.tool_calls]
            # Same id, so add_messages replaces the proposal; model_copy keeps its
            # response_metadata and usage_metadata for the audit trail.
            return Command(goto=tools, update={
                key: _reviewed(record, review, "edit", edited),
                messages_key: [message.model_copy(update={"tool_calls": edited})],
            })
        # Reject, or any unrecognized answer: fail closed.
        return Command(goto=agent, update={
            key: _reviewed(record, review, "reject"),
            messages_key: _refusals(message, record, "rejected_by_reviewer", review.get("note")),
        })

    return cli_human_review


def blocked_node(*, key: str = GUARD_KEY, messages_key: str = "messages"):
    """A node that answers every tool call of a blocked turn with an error ``ToolMessage``."""

    def cli_blocked(state: Mapping[str, Any]) -> dict[str, Any]:
        return {messages_key: _refusals(state[messages_key][-1], state.get(key) or {}, "blocked")}

    return cli_blocked


def add_cli_guard(builder: Any, guard: ToolGuard, *, agent: str = "agent", tools: str = "tools",
                  key: str = GUARD_KEY, messages_key: str = "messages") -> Any:
    """Wire the guard between an existing ``agent`` node and ``tools`` node.

    Adds the ``cli_guard``, ``cli_human_review`` and ``cli_blocked`` nodes and
    the conditional edges out of ``agent``. You still add ``START -> agent``
    and ``tools -> agent``. Returns the builder.
    """
    builder.add_node(GUARD_NODE, guard_node(guard, key=key, messages_key=messages_key))
    builder.add_node(REVIEW_NODE, human_review_node(key=key, tools=tools, agent=agent, messages_key=messages_key),
                     destinations=(tools, agent))
    builder.add_node(BLOCKED_NODE, blocked_node(key=key, messages_key=messages_key))
    builder.add_conditional_edges(agent, lambda s: tools_condition(s, messages_key=messages_key),
                                  {"tools": GUARD_NODE, END: END})
    builder.add_conditional_edges(GUARD_NODE, route_after_guard(key),
                                  {ALLOW: tools, ESCALATE: REVIEW_NODE, BLOCK: BLOCKED_NODE})
    builder.add_edge(BLOCKED_NODE, agent)
    return builder


def pending_review(result: Any) -> Optional[dict[str, Any]]:
    """The review request in an ``invoke`` result (v1 dict or v2 ``GraphOutput``), or ``None``."""
    interrupts = getattr(result, "interrupts", None)
    if interrupts is None and isinstance(result, dict):
        interrupts = result.get("__interrupt__", ())
    for item in interrupts or ():
        value = getattr(item, "value", None)
        if isinstance(value, Mapping) and value.get("kind") == "cli_human_review":
            return {"interrupt_id": getattr(item, "id", None), **value}
    return None


__all__ = ["CLIGuardState", "add_cli_guard", "guard_node", "human_review_node", "blocked_node",
           "route_after_guard", "pending_review", "GUARD_KEY"]
