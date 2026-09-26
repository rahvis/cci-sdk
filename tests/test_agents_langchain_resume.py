"""Regression: a reviewer's decision survives a resume on a worker whose guard would now answer differently.

LangChain re-evaluates the human-in-the-loop ``when`` predicate on resume. If
the guard were consulted live, a changed answer (another worker, a restart, a
recalibrated profile, sampling noise) would skip the interrupt, ignore the
reviewer, and run a call the reviewer rejected. The adapter stores each
decision in checkpointed state and only reads it afterwards.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from cli_sdk import Gate  # noqa: E402
from cli_sdk.evidence import MockEvidenceBackend  # noqa: E402
from cli_sdk.integrations import GuardRule, ToolGuard  # noqa: E402
from cli_sdk.integrations.langchain import cli_middleware, guard_decisions, pending_reviews  # noqa: E402
from cli_sdk.local import LocalCLIClient  # noqa: E402

GATE = Gate(instructions="Is approving this refund correct under policy?", calibration_profile="refunds",
            guarantee="risk", target=0.05)


class FakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _guard(tmp_path, name, p_true, label=lambda amount: amount <= 100):
    def scorer(context, instructions, options):
        p = p_true(context["amount"])
        return {"true": p, "false": 1 - p}

    client = LocalCLIClient(MockEvidenceBackend(scorer=scorer), store=tmp_path / name)
    examples = [{"context": {"amount": a}, "label": label(a)} for a in (20.0, 50.0, 80.0, 600.0, 900.0) * 40]
    client.calibrate(GATE, examples)
    return ToolGuard(client, [GuardRule(tool="issue_refund", query=GATE, context=lambda a: {"amount": float(a["amount"])})])


def _proposal(amount):
    return AIMessage("", tool_calls=[{"name": "issue_refund", "args": {"order_id": "A1", "amount": amount},
                                      "id": "call_1", "type": "tool_call"}])


@pytest.mark.parametrize("decision", ["reject", "approve"])
def test_resume_on_a_drifted_worker_honours_the_reviewer(tmp_path, decision):
    ran = []

    @tool
    def issue_refund(order_id: str, amount: float) -> str:
        """Issue a refund."""
        ran.append(amount)
        return f"refunded {amount}"

    strict = _guard(tmp_path, "strict", lambda amount: 0.97 if amount <= 100 else 0.6)
    # A worker whose recalibrated guard would now auto-approve everything.
    lenient = _guard(tmp_path, "lenient", lambda amount: 0.99, label=lambda amount: True)
    assert lenient.check("issue_refund", {"order_id": "A1", "amount": 600}).allowed

    saver = InMemorySaver()
    config = {"configurable": {"thread_id": f"thread-{decision}"}}
    worker1 = create_agent(FakeModel(responses=[_proposal(600), AIMessage("done")]), tools=[issue_refund],
                           middleware=cli_middleware(strict), checkpointer=saver)
    paused = worker1.invoke({"messages": [HumanMessage("refund 600")]}, config)
    assert [p["name"] for p in pending_reviews(paused)] == ["issue_refund"]
    assert guard_decisions(paused)["call_1"]["action"] == "escalate"

    evaluations_before = len(lenient.log)
    worker2 = create_agent(FakeModel(responses=[AIMessage("finished")]), tools=[issue_refund],
                           middleware=cli_middleware(lenient), checkpointer=saver)
    resume = {"type": "reject", "message": "possible fraud"} if decision == "reject" else {"type": "approve"}
    worker2.invoke(Command(resume={"decisions": [resume]}), config)

    assert ran == ([] if decision == "reject" else [600.0])
    assert len(lenient.log) == evaluations_before  # the stored decision was read, not recomputed


def test_edited_arguments_run_after_review(tmp_path):
    ran = []

    @tool
    def issue_refund(order_id: str, amount: float) -> str:
        """Issue a refund."""
        ran.append(amount)
        return "ok"

    guard = _guard(tmp_path, "g", lambda amount: 0.97 if amount <= 100 else 0.6)
    agent = create_agent(FakeModel(responses=[_proposal(600), AIMessage("done")]), tools=[issue_refund],
                         middleware=cli_middleware(guard), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "edit"}}
    agent.invoke({"messages": [HumanMessage("refund")]}, config)
    edit = {"type": "edit", "edited_action": {"name": "issue_refund", "args": {"order_id": "A1", "amount": 75.0}}}
    agent.invoke(Command(resume={"decisions": [edit]}), config)
    assert ran == [75.0]


def test_without_human_review_escalations_are_refused(tmp_path):
    ran = []

    @tool
    def issue_refund(order_id: str, amount: float) -> str:
        """Issue a refund."""
        ran.append(amount)
        return "ok"

    guard = _guard(tmp_path, "g", lambda amount: 0.97 if amount <= 100 else 0.6)
    agent = create_agent(FakeModel(responses=[_proposal(600), AIMessage("done")]), tools=[issue_refund],
                         middleware=cli_middleware(guard, human_review=False))
    out = agent.invoke({"messages": [HumanMessage("refund")]})
    tool_messages = [m for m in out["messages"] if m.type == "tool"]
    assert ran == [] and tool_messages[0].status == "error"
    assert "needs_human_review" in tool_messages[0].content
