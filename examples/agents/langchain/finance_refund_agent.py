"""LangChain refunds agent whose issue_refund tool is guarded by a calibrated Gate (synthetic data).

What the agent does
    A card and e-commerce refunds assistant built with
    ``langchain.agents.create_agent``. It reads the customer's message, looks
    the order up with ``lookup_order`` (read-only, unguarded) and proposes
    ``issue_refund(order_id, amount, reason)``.

The decision being guarded
    ``issue_refund`` moves money, so every proposed call is checked before it
    runs. The guard scores the call against the order record from the order
    system (not the agent's summary of it) and the written refund policy, with
    the same context builder that produced the calibration set.

Primitive and guarantee
    ``Gate(guarantee="risk", target=0.05)``, calibrated with conformal risk
    control (CRC) on 320 labelled synthetic refund decisions
    (``data/refund_requests.jsonl``). The guarantee card reads: "Expected rate
    of decisions that are auto-approved and wrong is at most 0.05". That is a
    bound over requests like the calibration set, counted over all requests;
    it does not make any single auto-approved refund correct. Every refund the
    Gate does not auto-approve pauses for a human.

How it maps onto LangChain
    ``middleware=cli_middleware(guard)`` installs ``CLIGuardMiddleware``, which
    scores each proposed call once, right after the model turn, and stores the
    decision in the checkpointed agent state (``cli_guard``), and
    ``HumanInTheLoopMiddleware``, whose ``when`` predicate reads that decision
    so only escalated calls interrupt. A reviewer can approve, edit or reject;
    the run resumes with ``Command(resume=...)`` on the same ``thread_id``. An
    edit is an explicit human approval: the edited call runs as written.
    ``--no-human-review`` shows batch-job mode: there is no interrupt, and
    escalated calls come back to the agent as an error ``ToolMessage`` so they
    can be queued.

Run it
    Offline and keyless (default; a scripted LangChain fake model proposes the
    tool calls, and a deterministic mock evidence model scores them)::

        python examples/agents/langchain/finance_refund_agent.py
        python examples/agents/langchain/finance_refund_agent.py --no-human-review
        python examples/agents/langchain/finance_refund_agent.py --interactive

    With a real model (keys come from environment variables; see
    ``examples/agents/.env.example``)::

        export OPENAI_API_KEY=...        # then: --provider openai
        export AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_DEPLOYMENT=...
                                         # then: --provider azure
        export ANTHROPIC_API_KEY=...     # then: --provider anthropic
        export GOOGLE_API_KEY=...        # then: --provider gemini

    With open-weight Gemma served locally (tool calling must be enabled in the
    server, or the agent never calls a tool and the guard never runs)::

        vllm serve google/gemma-4-12B-it --max-logprobs 20 --generation-config vllm \\
            --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
        python examples/agents/langchain/finance_refund_agent.py --provider vllm

        python -m sglang.launch_server --model-path google/gemma-4-12B-it --port 30000 \\
            --tool-call-parser gemma4 --reasoning-parser gemma4
        python examples/agents/langchain/finance_refund_agent.py --provider sglang

    The agent model and the evidence model can differ, for example Claude as
    the agent and logprob evidence from Gemma on vLLM::

        python examples/agents/langchain/finance_refund_agent.py --provider anthropic --evidence-provider vllm

    Install: ``pip install "cci-sdk[langchain,openai]" langchain-openai`` (use
    ``langchain-anthropic`` or ``langchain-google-genai`` for those agents).

Cost of calibrating with a real evidence model
    The first run scores the 320 calibration examples once and caches the
    profile in ``.cli_profiles/refund-approvals-v1.json`` next to this file:
    320 requests at access level L1 (OpenAI, Azure OpenAI, vLLM, SGLang: one
    logprob request per example), or 320 x 8 = 2,560 requests at L0
    (Anthropic, Gemini: 8 samples per example). Each guarded call then costs
    one evidence request at L1 (8 at L0). A different evidence model, or an
    edited Gate prompt, is detected and recalibrated automatically.

What to expect (mock mode)
    R-1001  small damaged-item refund, well inside the window: allow; the tool runs.
    R-1002  change-of-mind return that also refunds the shipping fee: escalate;
            the simulated reviewer edits the amount to the item price
            (181.99 -> 169.99, shipping is not refundable), then the tool runs.
    R-1003  second refund on an order that was already refunded: escalate; the
            reviewer rejects and the tool never runs.
    With ``--no-human-review`` R-1002 and R-1003 are refused with an error
    ``ToolMessage`` and listed for the review queue. The whole run takes a few
    seconds; a second run reuses the cached profile and makes the same decisions.

All data is synthetic. This is not financial, credit or legal advice.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Optional

AGENTS_DIR = Path(__file__).resolve().parents[1]
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

from _shared import calibration, display, providers  # noqa: E402
from _shared.review import Reviewer, format_request  # noqa: E402

try:
    from langchain.agents import create_agent
    from langchain.messages import AIMessage, HumanMessage, ToolMessage
    from langchain.tools import tool
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
except ImportError as exc:  # pragma: no cover - exercised only without LangChain
    raise SystemExit('This example needs LangChain 1.x: pip install "cci-sdk[langchain]"') from exc

from cli_sdk import Gate  # noqa: E402
from cli_sdk.exceptions import BackendError  # noqa: E402
from cli_sdk.integrations import GuardRule, ToolGuard  # noqa: E402
from cli_sdk.integrations.langchain import cli_middleware, guard_decisions, pending_reviews  # noqa: E402
from cli_sdk.local import LocalCLIClient  # noqa: E402


def _load(path: Path) -> ModuleType:
    """Import a data generator by path (it holds the policy text and the context builder)."""
    spec = importlib.util.spec_from_file_location(f"_cli_example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# One definition of the policy, the order record and the evaluation context,
# shared with the generator of the calibration set.
refunds = _load(AGENTS_DIR / "data" / "generate_refund_requests.py")

Reason = Literal["damaged_item", "wrong_item", "not_delivered", "changed_mind", "duplicate_charge"]

# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

REFUND_GATE = Gate(
    instructions=(
        "Under the refund policy in the context, is issuing the proposed refund for this order correct? "
        "Answer true only if every rule of the policy holds for the order record and the proposed refund."
    ),
    calibration_profile="refund-approvals-v1",
    guarantee="risk",
    target=0.05,
)

# The order system of record for the demo requests (synthetic).
ORDERS = {
    order["order_id"]: order
    for order in (
        refunds.make_order("ORD-90101", category="home_goods", item_price=14.99, shipping_fee=4.99,
                           purchased_days_ago=7, delivery_status="delivered", delivered_days_ago=3,
                           photo_on_file=True),
        refunds.make_order("ORD-90106", category="apparel", item_price=169.99, shipping_fee=12.00,
                           purchased_days_ago=32, delivery_status="delivered", delivered_days_ago=27),
        refunds.make_order("ORD-90103", category="electronics", item_price=59.99, shipping_fee=4.99,
                           purchased_days_ago=12, delivery_status="delivered", delivered_days_ago=8,
                           refunded_to_date=64.98),
    )
}


def refund_guard_context(args: dict[str, Any]) -> dict[str, Any]:
    """What the guard evaluates for a proposed issue_refund call.

    The order comes from the order system, not from the agent, and the
    context is built by the same function as every calibration example.
    An unknown order raises, and the guard then escalates (fails closed).
    """
    order = ORDERS.get(args["order_id"])
    if order is None:
        raise LookupError(f"no order {args['order_id']!r} in the order system")
    return refunds.refund_context(order, args["amount"], args["reason"])


def build_guard(client: LocalCLIClient) -> ToolGuard:
    return ToolGuard(client, [GuardRule(tool="issue_refund", query=REFUND_GATE, context=refund_guard_context)])


# ---------------------------------------------------------------------------
# Mock evidence model: reads the case like a fallible reviewer, never the label
# ---------------------------------------------------------------------------


def _unit(*parts: Any) -> float:
    """Deterministic pseudo-random number in [0, 1) keyed on the case."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def refund_scorer(context: Any, instructions: Any, options: Any) -> dict[str, float]:
    """P(issuing this refund is correct) as a fallible model would estimate it (``--provider mock`` only).

    It checks the order against the policy, overlooks each problem now and
    then (keyed on the order id, so runs are reproducible), is cautious near
    the window's edge and on large amounts, and adds per-case noise. It only
    sees the evaluation context; it never sees a label.
    """
    order, proposal = context["order"], context["proposed_refund"]
    case = order["order_id"]
    amount, reason = float(proposal["amount"]), proposal["reason"]
    remaining = order["amount_paid"] - order["refunded_to_date"]
    window = 45 if order["customer_tier"] == "plus" else 30
    delivered = order["delivered_days_ago"]
    item_reason = reason in ("damaged_item", "wrong_item", "changed_mind")
    overdue = delivered - window if item_reason and delivered is not None else 0

    logit = 4.0
    checks = (  # (problem, present?, how much it lowers the score, chance the model overlooks it)
        ("window", overdue > 0, min(6.0, 1.5 + 0.35 * overdue), 0.05),
        ("not_delivered_yet", item_reason and order["delivery_status"] != "delivered", 4.0, 0.03),
        ("carrier_confirmed", reason == "not_delivered" and order["delivery_status"] == "delivered", 3.8, 0.08),
        ("claim_timing", reason == "not_delivered" and not 10 <= order["purchased_days_ago"] <= 90, 3.5, 0.08),
        ("duplicate_window", reason == "duplicate_charge" and order["purchased_days_ago"] > 120, 3.5, 0.05),
        ("over_remaining", amount > remaining + 0.005, 4.5, 0.04),
        ("shipping_refunded", reason == "changed_mind" and order["item_price"] + 0.005 < amount <= remaining + 0.005,
         2.8, 0.20),
        ("category", reason != "duplicate_charge"
         and (order["category"] in ("gift_card", "digital_download") or order["final_sale"]), 5.0, 0.04),
        ("prior_refund", order["refunded_to_date"] > 0, 5.0, 0.06),
        ("hold", order["open_fraud_flag"] or order["open_chargeback"], 6.0, 0.03),
        ("no_photo", reason in ("damaged_item", "wrong_item") and not order["photo_on_file"], 3.2, 0.10),
    )
    for name, present, weight, overlook in checks:
        if present and _unit(case, name) >= overlook:
            logit -= weight
    if item_reason and delivered is not None and 0 <= window - delivered <= 4:
        logit -= 1.0  # close to the window's edge
    logit -= 1.2 if amount > 500 else 0.6 if amount > 150 else 0.0
    logit += 0.9 * (2.0 * _unit(case, "noise") - 1.0)
    p_true = 1.0 / (1.0 + math.exp(-logit))
    return {"true": p_true, "false": 1.0 - p_true}


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the refunds assistant of an online store. For every refund request, first call lookup_order "
    "with the order id, then call issue_refund once with the order id, the amount and the reason code that "
    "fit the request. Never say a refund was issued unless issue_refund returned a success result. If a tool "
    "result says the refund needs human review or was rejected, do not retry it: tell the customer what "
    "happens next. The store's refund policy:\n\n" + refunds.REFUND_POLICY
)


def make_tools(ledger: list[dict[str, Any]]) -> list[Any]:
    """The agent's tools; ``ledger`` records every refund that was actually issued."""

    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order record by its id (read-only)."""
        order = ORDERS.get(order_id)
        return json.dumps(order, sort_keys=True) if order else f"No order {order_id} was found."

    @tool
    def issue_refund(order_id: str, amount: float, reason: Reason) -> str:
        """Refund ``amount`` USD to the card used for the order, with a policy reason code."""
        ledger.append({"order_id": order_id, "amount": round(float(amount), 2), "reason": reason})
        return f"Refund of ${float(amount):.2f} issued to the card on {order_id} (reason: {reason})."

    return [lookup_order, issue_refund]


def build_agent(model: Any, guard: ToolGuard, ledger: list[dict[str, Any]], *, human_review: bool = True,
                checkpointer: Any = None) -> Any:
    """``create_agent`` with the calibrated guard as middleware.

    Interrupts need a checkpointer. ``InMemorySaver`` is for a single
    process; in production use a durable checkpointer shared by every worker
    (the guard's decisions are stored in the checkpointed state, so a resume
    on another worker honours the decision the reviewer saw).
    """
    return create_agent(
        model,
        tools=make_tools(ledger),
        system_prompt=SYSTEM_PROMPT,
        middleware=cli_middleware(guard, human_review=human_review),
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )


class ScriptedAgentModel(FakeMessagesListChatModel):
    """The agent model for ``--provider mock``: LangChain's fake chat model, scripted per scenario.

    It emits the scenario's tool calls in order, then answers from the last
    tool result. ``bind_tools`` is a no-op because ``create_agent`` binds the
    tools and the fake does not implement it.
    """

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> "ScriptedAgentModel":
        return self

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        if self.i < len(self.responses):
            message = self.responses[self.i]
            self.i += 1
        else:
            message = AIMessage(content=mock_reply(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])


def mock_reply(messages: list[Any]) -> str:
    """What the scripted agent tells the customer, based on the last tool result."""
    last = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
    text = str(last.content) if last is not None else ""
    if last is None:
        return "How can I help with your order?"
    if last.status != "error":
        issued = text.split("Tool response:", 1)[-1].strip()
        if "human reviewer replaced" in text:
            return f"A refunds specialist adjusted the amount before it was issued. {issued}"
        return f"Done. {issued}"
    if text.startswith("User rejected"):
        why = text.split("with reason:", 1)[-1].strip()
        return f"A refunds specialist reviewed this request and did not issue a refund. {why}"
    return ("This refund needs a refunds specialist's review before it can be issued. I have added it to the "
            "review queue; you will get an email when it is decided.")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    case_id: str
    title: str
    message: str
    order_id: str
    refund: dict[str, Any]  # the issue_refund arguments the scripted agent proposes


SCENARIOS = (
    Scenario("R-1001", "Damaged mug, small refund, well inside the window",
             "Order ORD-90101 arrived with a cracked mug. I uploaded a photo. Can I get the $14.99 back?",
             "ORD-90101", {"order_id": "ORD-90101", "amount": 14.99, "reason": "damaged_item"}),
    Scenario("R-1002", "Change-of-mind return near the end of the window, shipping included",
             "I changed my mind about the coat from order ORD-90106. Please refund everything I paid, $181.99.",
             "ORD-90106", {"order_id": "ORD-90106", "amount": 181.99, "reason": "changed_mind"}),
    Scenario("R-1003", "Second refund on an order that was already refunded",
             "I was charged twice for the earbuds on order ORD-90103. Please refund the duplicate charge of $64.98.",
             "ORD-90103", {"order_id": "ORD-90103", "amount": 64.98, "reason": "duplicate_charge"}),
)

# The simulated reviewer (replace with your review queue). R-1002 is edited:
# rule 3 refunds the item price only on change-of-mind returns.
REVIEW_SCRIPT = {"R-1001": True, "R-1003": False}
REVIEW_EDITS = {"R-1002": {"amount": 169.99}}
REVIEW_NOTES = {
    "R-1002": "Change-of-mind return: rule 3 refunds the item price; the $12.00 shipping fee is not refundable.",
    "R-1003": "A full refund of $64.98 was already issued on this order; it reversed the duplicate charge.",
}


def scripted_model(scenario: Scenario) -> ScriptedAgentModel:
    """A fresh fake per run: the fake keeps its position in the script on the instance."""
    calls = [("lookup_order", {"order_id": scenario.order_id}), ("issue_refund", scenario.refund)]
    return ScriptedAgentModel(responses=[
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"{scenario.case_id}-{n}",
                                           "type": "tool_call"}])
        for n, (name, args) in enumerate(calls, start=1)
    ])


@dataclass
class Outcome:
    """What happened in one scenario (the tests assert on this)."""

    case_id: str
    guard_actions: list[str] = field(default_factory=list)  # one per proposed issue_refund call
    reviews: list[str] = field(default_factory=list)        # approve / edit / reject
    refunds_issued: list[dict[str, Any]] = field(default_factory=list)
    tool_statuses: list[str] = field(default_factory=list)  # status of each issue_refund ToolMessage
    final_reply: str = ""

    @property
    def executed(self) -> bool:
        return bool(self.refunds_issued)


def _wrap(prefix: str, text: str) -> None:
    lines = textwrap.wrap(" ".join(str(text).split()), 78 - len(prefix), break_on_hyphens=False) or [""]
    print(prefix + lines[0])
    for line in lines[1:]:
        print(" " * len(prefix) + line)


def _call_text(name: str, args: dict[str, Any]) -> str:
    return f"{name}({', '.join(f'{k}={json.dumps(v)}' for k, v in sorted(args.items()))})"


def _show_decision(decision: dict[str, Any]) -> None:
    _wrap("    guard:    ", f"{str(decision.get('action')).upper()}: {decision.get('reason')}")
    statement = (decision.get("guarantee") or {}).get("statement")
    if statement:
        _wrap(" " * 14, f"guarantee: {statement}")


def _tool_text(message: ToolMessage) -> str:
    """A refused call's ToolMessage is JSON (status, message, and the guard decision under "cli")."""
    try:
        body = json.loads(message.content)
    except (TypeError, ValueError):
        return str(message.content)
    if isinstance(body, dict) and "status" in body and "message" in body:
        return f"{body['status']}: {body['message']} (the guard decision is attached under \"cli\")"
    return str(message.content)


def show_messages(result: Any, start: int, outcome: Outcome) -> int:
    """Print the new part of the transcript, with the decision the guard stored for each guarded call."""
    messages = result.value["messages"]
    stored = guard_decisions(result)  # tool_call_id -> decision, from the checkpointed agent state
    for message in messages[start:]:
        if isinstance(message, HumanMessage):
            _wrap("    customer: ", message.content)
        elif isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                _wrap("    agent:    ", "proposes " + _call_text(call["name"], call["args"]))
                if call["id"] in stored:
                    outcome.guard_actions.append(stored[call["id"]]["action"])
                    _show_decision(stored[call["id"]])
        elif isinstance(message, ToolMessage):
            if message.name == "issue_refund":
                outcome.tool_statuses.append(message.status)
            _wrap(f"    tool:     [{message.status}] ", _tool_text(message))
        elif isinstance(message, AIMessage) and message.content:
            outcome.final_reply = str(message.content)
            _wrap("    agent:    ", f'"{outcome.final_reply}"')
    return len(messages)


def stored_decision_for(result: Any, action: dict[str, Any]) -> dict[str, Any]:
    """The stored decision behind a pending action (matched to the proposed tool call)."""
    stored = guard_decisions(result)
    for message in reversed(result.value["messages"]):
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                if call["name"] == action["name"] and call["args"] == action["args"] and call["id"] in stored:
                    return stored[call["id"]]
            break
    return {"tool": action["name"], "arguments": action["args"], "reason": action.get("description", "")}


def review(scenario: Scenario, action: dict[str, Any], decision: dict[str, Any], reviewer: Reviewer,
           edits: dict[str, dict[str, Any]], outcome: Outcome) -> dict[str, Any]:
    """Turn a pending action into a HumanInTheLoopMiddleware decision (approve / edit / reject)."""
    note = REVIEW_NOTES.get(scenario.case_id, "Declined by the reviewer.")
    if not reviewer.interactive and scenario.case_id in edits:
        print(f"    review request for {scenario.case_id}:")
        for line in format_request(decision).splitlines():
            print(f"      {line}")
        new_args = {**action["args"], **edits[scenario.case_id]}
        changes = ", ".join(f"{k} {action['args'].get(k)} -> {v}" for k, v in edits[scenario.case_id].items())
        _wrap(f"    [simulated {reviewer.role}] ", f"edited {changes}. {note}")
        outcome.reviews.append("edit")
        return {"type": "edit", "edited_action": {"name": action["name"], "args": new_args}}
    if reviewer.decide(scenario.case_id, decision):
        outcome.reviews.append("approve")
        return {"type": "approve"}
    outcome.reviews.append("reject")
    return {"type": "reject", "message": note}


def run_scenario(agent: Any, scenario: Scenario, reviewer: Reviewer, edits: dict[str, dict[str, Any]],
                 ledger: list[dict[str, Any]], thread: str) -> Outcome:
    outcome = Outcome(scenario.case_id)
    already = len(ledger)
    config = {"configurable": {"thread_id": thread}}
    result = agent.invoke({"messages": [HumanMessage(scenario.message)]}, config, version="v2")
    shown = show_messages(result, 0, outcome)
    for _ in range(5):  # a real agent may propose more than one guarded call
        pending = pending_reviews(result)
        if not pending:
            break
        print("    paused:   HumanInTheLoopMiddleware interrupt; the run waits for a reviewer")
        decisions = [review(scenario, action, stored_decision_for(result, action), reviewer, edits, outcome)
                     for action in pending]
        _wrap("    resume:   ", f"Command(resume={json.dumps({'decisions': decisions})})")
        if any(d["type"] == "edit" for d in decisions):
            _wrap("    note:     ", "a reviewer's edit is an explicit human approval: the edited call runs as "
                  "written and is not scored again")
        result = agent.invoke(Command(resume={"decisions": decisions}), config, version="v2")
        shown = show_messages(result, shown, outcome)
    outcome.refunds_issued = ledger[already:]
    return outcome


def print_summary(outcomes: list[Outcome], guard: ToolGuard, human_review: bool) -> None:
    print("-" * 78)
    print("Summary")
    print(f"    {'case':<8} {'guard':<10} {'human review':<14} refund issued")
    for o in outcomes:
        issued = ", ".join(f"${r['amount']:.2f} on {r['order_id']}" for r in o.refunds_issued) or "none"
        queued = not human_review and "escalate" in o.guard_actions
        reviews = ", ".join(o.reviews) or ("queued" if queued else "-")
        print(f"    {o.case_id:<8} {', '.join(o.guard_actions) or '-':<10} {reviews:<14} {issued}")
    stored = "; stored in the checkpointed agent state and reused on resume" if human_review else ""
    _wrap("    ", f"guard evaluations: {len(guard.log)} (each proposed call is scored once{stored})")
    if not human_review:
        queued = [o.case_id for o in outcomes if "escalate" in o.guard_actions]
        _wrap("    ", f"batch mode: {', '.join(queued) or 'nothing'} refused with an error ToolMessage and queued "
              "for human review")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    providers.add_arguments(parser)
    parser.add_argument("--no-human-review", action="store_true",
                        help="batch-job mode: refuse escalated calls with an error ToolMessage instead of pausing")
    return parser.parse_args(argv)


def real_agent_model(provider: str) -> Any:
    from _shared.langchain_models import chat_model

    try:
        return chat_model(provider)
    except ImportError as exc:
        package = {"anthropic": "langchain-anthropic", "gemini": "langchain-google-genai"}.get(provider,
                                                                                               "langchain-openai")
        raise SystemExit(f"--provider {provider} needs {package}: pip install {package}") from exc


def run(argv: Optional[list[str]] = None, *, reviewer: Optional[Reviewer] = None,
        edits: Optional[dict[str, dict[str, Any]]] = None) -> list[Outcome]:
    """Run every scenario and return what happened (``main`` prints; tests assert)."""
    args = parse_args(argv)
    evidence_provider = args.evidence_provider or args.provider
    # Resolve both models first: a placeholder key stops here, before any network call.
    evidence = providers.evidence_backend(evidence_provider, mock_scorer=refund_scorer)
    shared_model = None if args.provider == "mock" else real_agent_model(args.provider)
    reviewer = reviewer or Reviewer(interactive=args.interactive, script=REVIEW_SCRIPT, default=False,
                                    role="refunds specialist")
    edits = REVIEW_EDITS if edits is None else edits
    human_review = not args.no_human_review

    display.banner("LangChain refunds agent: calibrated guard on issue_refund", "finance")
    print(f"agent model: {args.provider}   evidence model: {evidence_provider}   "
          f"human review: {'on (HumanInTheLoopMiddleware)' if human_review else 'off (batch-job mode)'}")

    client = LocalCLIClient(evidence, store=calibration.default_store(__file__, args.store),
                            sample_count=providers.sample_count(evidence_provider))
    cached, _ = client.calibration_status(REFUND_GATE)
    examples = calibration.load_jsonl("refund_requests.jsonl")
    try:
        (profile,) = calibration.ensure_calibrated(client, [REFUND_GATE], examples, recalibrate=args.recalibrate)
    except BackendError as exc:  # e.g. the vLLM / SGLang server is not running
        raise SystemExit(f"could not score the calibration set with --evidence-provider {evidence_provider}: "
                         f"{exc}") from exc
    source = "reused from the profile store" if cached and not args.recalibrate else "calibrated on this run"
    print(f"profile {profile.name}: {profile.method}, n={profile.n} synthetic labelled decisions, status "
          f"{profile.status} ({source})")
    guard = build_guard(client)

    outcomes = []
    for scenario in SCENARIOS:
        display.scenario(scenario.case_id, scenario.title)
        ledger: list[dict[str, Any]] = []
        model = scripted_model(scenario) if shared_model is None else shared_model
        agent = build_agent(model, guard, ledger, human_review=human_review)
        thread = f"{scenario.case_id}-{'review' if human_review else 'batch'}"
        outcomes.append(run_scenario(agent, scenario, reviewer, edits, ledger, thread))

    print_summary(outcomes, guard, human_review)
    display.footer()
    return outcomes


def main(argv: Optional[list[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
