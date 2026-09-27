"""LangChain patient-portal triage agent; send_triage_advice is guarded by a calibrated Set (synthetic data).

What the agent does
    A clinic's patient-portal assistant built with
    ``langchain.agents.create_agent``. It reads a new portal message (intake
    form plus the patient's own words) and proposes
    ``send_triage_advice(message_id, level)`` with one of four levels:
    ``self_care``, ``routine_appointment``, ``urgent_care``, ``emergency``.

The decision being guarded
    Sending triage advice to a patient. Automated replies are allowed only for
    the two low-acuity levels, and only when the calibrated prediction set for
    the message is exactly the level the agent proposed. Everything else
    (a set with two or more levels, a confident set for a different level, and
    every urgent_care or emergency proposal) goes to a nurse.

Primitive and guarantee
    ``Set(method="APS", alpha=0.05)`` over the four levels, calibrated on 320
    labelled synthetic portal messages (``data/triage_messages.jsonl``). The
    guarantee card reads: "Contains the correct answer at least 95% of the
    time", over messages exchangeable with the calibration set. It is a
    coverage rate over many messages; it does not make any single automated
    reply correct. ``GuardRule(allow_labels=["self_care",
    "routine_appointment"], match_argument="level")`` turns the set into the
    allow / escalate decision.

How it maps onto LangChain
    ``middleware=cli_middleware(guard)``: ``CLIGuardMiddleware`` scores each
    proposed call once and stores the decision in the checkpointed agent
    state; ``HumanInTheLoopMiddleware`` pauses only the calls the guard
    escalated; a nurse approves, edits (for example changes the level) or
    rejects, and the run resumes with ``Command(resume=...)``. A nurse's edit
    is an explicit human approval, so the edited call runs as written.
    ``--no-human-review`` refuses escalated calls with an error
    ``ToolMessage`` instead (batch mode).

Emergency pathway
    In this example an emergency level is never sent automatically: it always
    waits for a nurse. A production portal must not let a review queue delay
    an emergency: it should show fixed emergency instructions (call emergency
    services) to the patient immediately, whatever the agent or the guard
    does, and route the message to a nurse at the same time.

Run it
    Offline and keyless (default; a scripted LangChain fake model proposes the
    tool calls, and a deterministic mock evidence model scores them)::

        python examples/agents/langchain/healthcare_patient_triage.py
        python examples/agents/langchain/healthcare_patient_triage.py --no-human-review
        python examples/agents/langchain/healthcare_patient_triage.py --interactive

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
        python examples/agents/langchain/healthcare_patient_triage.py --provider vllm

        python -m sglang.launch_server --model-path google/gemma-4-12B-it --port 30000 \\
            --tool-call-parser gemma4 --reasoning-parser gemma4
        python examples/agents/langchain/healthcare_patient_triage.py --provider sglang

    Mix models with ``--evidence-provider``, for example
    ``--provider anthropic --evidence-provider vllm``.

    Install: ``pip install "cci-sdk[langchain,openai]" langchain-openai`` (use
    ``langchain-anthropic`` or ``langchain-google-genai`` for those agents).

Cost of calibrating with a real evidence model
    The first run scores the 320 calibration messages once and caches the
    profile in ``.cli_profiles/portal-triage-v1.json`` next to this file: 320
    requests at access level L1 (OpenAI, Azure OpenAI, vLLM, SGLang: one
    logprob request per message over the four option letters), or
    320 x 8 = 2,560 requests at L0 (Anthropic, Gemini: 8 samples per message).
    Each guarded call then costs one evidence request at L1 (8 at L0).

What to expect (mock mode)
    M-2001  mild cold, agent proposes self_care: the set is {self_care}; allow, advice sent.
    M-2002  chest tightness, agent proposes routine_appointment: the set is
            {routine_appointment, urgent_care}; escalate; the nurse approves.
    M-2003  stroke warning signs, agent proposes emergency: escalate (emergency
            is never automated); the nurse approves.
    M-2004  cough for 25 days, agent proposes self_care: the set is the
            singleton {routine_appointment}, which does not match the proposal;
            escalate; the nurse edits the level to routine_appointment, and the
            nurse's level is sent (a human edit is an explicit approval).
    A second run reuses the cached profile and makes the same decisions.

All data is synthetic. This is not medical advice and not a medical device.
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

from cli_sdk import Set  # noqa: E402
from cli_sdk.exceptions import BackendError  # noqa: E402
from cli_sdk.integrations import GuardRule, ToolGuard  # noqa: E402
from cli_sdk.integrations.langchain import cli_middleware, guard_decisions, pending_reviews  # noqa: E402
from cli_sdk.local import LocalCLIClient  # noqa: E402


def _load(path: Path) -> ModuleType:
    """Import a data generator by path (it holds the protocol text and the context builder)."""
    spec = importlib.util.spec_from_file_location(f"_cli_example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# One definition of the protocol, the message record and the evaluation
# context, shared with the generator of the calibration set.
triage = _load(AGENTS_DIR / "data" / "generate_triage_messages.py")

Level = Literal["self_care", "routine_appointment", "urgent_care", "emergency"]

# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

TRIAGE_SET = Set(
    instructions=(
        "Under the triage protocol in the context, which triage level does this patient-portal message "
        "require? Apply the protocol's rules in order."
    ),
    options=dict(triage.TRIAGE_LEVELS),
    calibration_profile="portal-triage-v1",
    alpha=0.05,
    method="APS",
)

# The portal's message store for the demo messages (synthetic).
MESSAGES = {
    message["message_id"]: message
    for message in (
        triage.portal_message("M-2001", age_band="18-39", complaint="cold_symptoms", duration_days=3,
                              self_rated_severity="mild", temperature_c=37.2),
        triage.portal_message("M-2002", age_band="18-39", complaint="chest_discomfort", duration_days=3,
                              self_rated_severity="mild"),
        triage.portal_message("M-2003", age_band="65+", complaint="headache", duration_days=1,
                              self_rated_severity="moderate", red_flags=["face_drooping", "slurred_speech"]),
        triage.portal_message("M-2004", age_band="40-64", complaint="cough", duration_days=25,
                              self_rated_severity="mild", temperature_c=37.1),
    )
}


def triage_guard_context(args: dict[str, Any]) -> dict[str, Any]:
    """What the guard evaluates for a proposed send_triage_advice call.

    The message comes from the portal's store, not from the agent, and the
    context is built by the same function as every calibration example. The
    proposed level is not part of the context: the set is computed from the
    message alone and then compared with the proposal (``match_argument``).
    """
    message = MESSAGES.get(args["message_id"])
    if message is None:
        raise LookupError(f"no portal message {args['message_id']!r}")
    return triage.triage_context(message)


def build_guard(client: LocalCLIClient) -> ToolGuard:
    rule = GuardRule(
        tool="send_triage_advice",
        query=TRIAGE_SET,
        context=triage_guard_context,
        allow_labels=["self_care", "routine_appointment"],  # urgent_care and emergency always go to a nurse
        match_argument="level",  # the set must be exactly the level the agent proposed
    )
    return ToolGuard(client, [rule])


# ---------------------------------------------------------------------------
# Mock evidence model: reads the message like a fallible model, never the label
# ---------------------------------------------------------------------------


def _unit(*parts: Any) -> float:
    """Deterministic pseudo-random number in [0, 1) keyed on the case."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def triage_scorer(context: Any, instructions: Any, options: Any) -> dict[str, float]:
    """Probabilities of the four levels as a fallible model would give them (``--provider mock`` only).

    It applies the protocol to the intake fields, now and then overlooks a
    red flag or an urgent sign (keyed on the message id, so runs are
    reproducible), is unsure near duration and temperature cut-offs, on chest
    discomfort and on vague messages, and adds per-case noise. It only sees
    the evaluation context; it never sees a label.
    """
    msg = context["message"]
    case, complaint, temp = msg["message_id"], msg["complaint"], msg["temperature_c"]
    older = msg["age_band"] in ("40-64", "65+")
    limit = triage.SELF_CARE_LIMIT_DAYS.get(complaint, 7)
    red = bool(msg["red_flags"]) and _unit(case, "red") >= 0.02
    urgent_sign = bool(msg["urgent_signs"]) and _unit(case, "urgent") >= 0.04
    fever = temp is not None and temp >= 38.0

    if red:
        level: Optional[str] = "emergency"
    elif complaint == "unclear":
        level = None
    elif (urgent_sign or (temp is not None and temp >= 39.0) or msg["self_rated_severity"] == "severe"
          or (msg["age_band"] == "65+" and fever) or (complaint == "chest_discomfort" and older)):
        level = "urgent_care"
    elif (msg["self_rated_severity"] == "moderate" or msg["duration_days"] > limit
          or complaint in ("urinary_symptoms", "chest_discomfort") or (fever and msg["duration_days"] >= 3)):
        level = "routine_appointment"
    else:
        level = "self_care"

    order = triage.LEVEL_ORDER
    logits = {key: 0.0 for key in order}
    noise = 0.7 * (2.0 * _unit(case, "noise") - 1.0)
    if level is None:  # a vague message: the protocol cannot settle it
        logits.update(self_care=4.5 + noise, routine_appointment=4.5 - noise, urgent_care=1.5)
    else:
        i = order.index(level)
        logits[level] = 6.5 + noise
        for j in (i - 1, i + 1):
            if 0 <= j < len(order):
                logits[order[j]] = 1.0
        up = order[min(i + 1, len(order) - 1)]
        down = order[max(i - 1, 0)]
        if complaint == "chest_discomfort" and not red:
            logits[level], logits[up] = 5.0, 4.4 + noise  # torn between this level and the next one up
        elif level in ("self_care", "routine_appointment") and limit and abs(msg["duration_days"] - limit) <= 1:
            other = up if level == "self_care" else down
            logits[other] = logits[level] - 1.2  # right at the self-care limit
        elif temp is not None and (abs(temp - 38.0) <= 0.2 or abs(temp - 39.0) <= 0.2):
            logits[up] = logits[level] - 1.5  # right at a temperature cut-off
    peak = max(logits.values())
    weights = {key: math.exp(value - peak) for key, value in logits.items()}
    total = sum(weights.values())
    return {key: weights[key] / total for key in options}


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You triage new messages in a clinic's patient portal. For each message, decide the triage level under "
    "the protocol below and call send_triage_advice exactly once with the message id and that level. Do not "
    "give medical advice in your own words. If a tool result says the advice needs a nurse's review or was "
    "declined, do not retry: say that a nurse will follow up.\n\n" + triage.TRIAGE_PROTOCOL
)


def make_tools(ledger: list[dict[str, Any]]) -> list[Any]:
    """The agent's tool; ``ledger`` records every piece of advice that was actually sent."""

    @tool
    def send_triage_advice(message_id: str, level: Level) -> str:
        """Send the patient the clinic's standard advice for a triage level."""
        ledger.append({"message_id": message_id, "level": level})
        return f"Standard {level} advice sent to the patient for message {message_id}."

    return [send_triage_advice]


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

    It emits the scenario's tool call, then answers from the tool result.
    ``bind_tools`` is a no-op because ``create_agent`` binds the tools and
    the fake does not implement it.
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
    """What the scripted agent writes back, based on the last tool result."""
    last = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
    if last is None:
        return "A nurse will review your message."
    text = str(last.content)
    if last.status != "error":
        sent = text.split("Tool response:", 1)[-1].strip()
        if "human reviewer replaced" in text:
            return f"A nurse changed the triage level before anything was sent. {sent}"
        return f"{sent} A nurse can see this conversation."
    if text.startswith("User rejected"):
        return "A nurse reviewed this message and did not send the automated advice; the nurse will follow up."
    return "This message needs a nurse's review, so no automated advice was sent. It is in the nurse queue."


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    case_id: str
    title: str
    proposed_level: str  # what the scripted agent proposes in --provider mock
    note: str = ""


SCENARIOS = (
    Scenario("M-2001", "Mild cold for 3 days; the agent proposes self_care", "self_care"),
    Scenario("M-2002", "Chest tightness, no red flags; the agent proposes routine_appointment",
             "routine_appointment"),
    Scenario("M-2003", "Stroke warning signs; the agent proposes emergency", "emergency",
             note=("Emergency advice is never automated here. A production portal also shows fixed emergency "
                   "instructions to the patient at once, without waiting for the nurse queue.")),
    Scenario("M-2004", "Cough for 25 days; the agent proposes self_care", "self_care"),
)

# The simulated nurse (replace with your nurse queue).
REVIEW_SCRIPT = {"M-2002": True, "M-2003": True}
REVIEW_EDITS = {"M-2004": {"level": "routine_appointment"}}
REVIEW_NOTES = {
    "M-2004": "A cough lasting more than 21 days calls for a routine appointment under rule 3.",
}


def patient_prompt(message_id: str) -> str:
    message = MESSAGES[message_id]
    return f"New portal message {message_id} (age band {message['age_band']}): {message['text']}"


def scripted_model(scenario: Scenario) -> ScriptedAgentModel:
    """A fresh fake per run: the fake keeps its position in the script on the instance."""
    args = {"message_id": scenario.case_id, "level": scenario.proposed_level}
    return ScriptedAgentModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "send_triage_advice", "args": args,
                                           "id": f"{scenario.case_id}-1", "type": "tool_call"}]),
    ])


@dataclass
class Outcome:
    """What happened in one scenario (the tests assert on this)."""

    case_id: str
    guard_actions: list[str] = field(default_factory=list)   # one per proposed send_triage_advice call
    prediction_sets: list[list[str]] = field(default_factory=list)
    reviews: list[str] = field(default_factory=list)         # approve / edit / reject
    advice_sent: list[dict[str, Any]] = field(default_factory=list)
    tool_statuses: list[str] = field(default_factory=list)
    final_reply: str = ""

    @property
    def executed(self) -> bool:
        return bool(self.advice_sent)


def _wrap(prefix: str, text: str) -> None:
    lines = textwrap.wrap(" ".join(str(text).split()), 78 - len(prefix), break_on_hyphens=False) or [""]
    print(prefix + lines[0])
    for line in lines[1:]:
        print(" " * len(prefix) + line)


def _call_text(name: str, args: dict[str, Any]) -> str:
    return f"{name}({', '.join(f'{k}={json.dumps(v)}' for k, v in sorted(args.items()))})"


def _show_decision(decision: dict[str, Any]) -> None:
    probabilities = (decision.get("evidence") or {}).get("probabilities") or {}
    if probabilities:
        shown = ", ".join(f"{k} {v:.3f}" for k, v in sorted(probabilities.items(), key=lambda kv: -kv[1]))
        _wrap("    evidence: ", shown)
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
            _wrap("    portal:   ", message.content)
        elif isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                _wrap("    agent:    ", "proposes " + _call_text(call["name"], call["args"]))
                if call["id"] in stored:
                    decision = stored[call["id"]]
                    outcome.guard_actions.append(decision["action"])
                    outcome.prediction_sets.append(list((decision.get("evidence") or {}).get("set", [])))
                    _show_decision(decision)
        elif isinstance(message, ToolMessage):
            if message.name == "send_triage_advice":
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
    note = REVIEW_NOTES.get(scenario.case_id, "Declined by the nurse; the nurse will contact the patient.")
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
    result = agent.invoke({"messages": [HumanMessage(patient_prompt(scenario.case_id))]}, config, version="v2")
    shown = show_messages(result, 0, outcome)
    for _ in range(5):  # a real agent may propose more than one guarded call
        pending = pending_reviews(result)
        if not pending:
            break
        print("    paused:   HumanInTheLoopMiddleware interrupt; the run waits for a nurse")
        decisions = [review(scenario, action, stored_decision_for(result, action), reviewer, edits, outcome)
                     for action in pending]
        _wrap("    resume:   ", f"Command(resume={json.dumps({'decisions': decisions})})")
        if any(d["type"] == "edit" for d in decisions):
            _wrap("    note:     ", "a nurse's edit is an explicit human approval: the edited call runs as "
                  "written and is not scored again")
        result = agent.invoke(Command(resume={"decisions": decisions}), config, version="v2")
        shown = show_messages(result, shown, outcome)
    if scenario.note:
        _wrap("    note:     ", scenario.note)
    outcome.advice_sent = ledger[already:]
    return outcome


def print_summary(outcomes: list[Outcome], guard: ToolGuard, human_review: bool) -> None:
    print("-" * 78)
    print("Summary")
    print(f"    {'case':<8} {'guard':<10} {'nurse':<9} advice sent")
    for o in outcomes:
        sent = ", ".join(a["level"] for a in o.advice_sent) or "none"
        queued = not human_review and "escalate" in o.guard_actions
        reviews = ", ".join(o.reviews) or ("queued" if queued else "-")
        print(f"    {o.case_id:<8} {', '.join(o.guard_actions) or '-':<10} {reviews:<9} {sent}")
    stored = "; stored in the checkpointed agent state and reused on resume" if human_review else ""
    _wrap("    ", f"guard evaluations: {len(guard.log)} (each proposed call is scored once{stored})")
    if not human_review:
        queued_cases = [o.case_id for o in outcomes if "escalate" in o.guard_actions]
        _wrap("    ", f"batch mode: {', '.join(queued_cases) or 'nothing'} refused with an error ToolMessage and "
              "queued for a nurse")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    providers.add_arguments(parser)
    parser.add_argument("--no-human-review", action="store_true",
                        help="batch mode: refuse escalated calls with an error ToolMessage instead of pausing")
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
    evidence = providers.evidence_backend(evidence_provider, mock_scorer=triage_scorer)
    shared_model = None if args.provider == "mock" else real_agent_model(args.provider)
    reviewer = reviewer or Reviewer(interactive=args.interactive, script=REVIEW_SCRIPT, default=False, role="nurse")
    edits = REVIEW_EDITS if edits is None else edits
    human_review = not args.no_human_review

    display.banner("LangChain patient-portal triage: calibrated guard on send_triage_advice", "healthcare")
    print(f"agent model: {args.provider}   evidence model: {evidence_provider}   "
          f"nurse review: {'on (HumanInTheLoopMiddleware)' if human_review else 'off (batch mode)'}")

    client = LocalCLIClient(evidence, store=calibration.default_store(__file__, args.store),
                            sample_count=providers.sample_count(evidence_provider))
    cached, _ = client.calibration_status(TRIAGE_SET)
    examples = calibration.load_jsonl("triage_messages.jsonl")
    try:
        (profile,) = calibration.ensure_calibrated(client, [TRIAGE_SET], examples, recalibrate=args.recalibrate)
    except BackendError as exc:  # e.g. the vLLM / SGLang server is not running
        raise SystemExit(f"could not score the calibration set with --evidence-provider {evidence_provider}: "
                         f"{exc}") from exc
    source = "reused from the profile store" if cached and not args.recalibrate else "calibrated on this run"
    print(f"profile {profile.name}: {profile.method}, alpha={profile.alpha:g}, n={profile.n} synthetic labelled "
          f"messages, status {profile.status} ({source})")
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
