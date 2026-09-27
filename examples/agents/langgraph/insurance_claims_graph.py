"""LangGraph claims agent whose payments pass through a calibrated guard node.

What the agent does
    A claims-processing agent built as a LangGraph ``StateGraph``: an
    ``agent`` node (a chat model bound to two tools) and a ``tools`` node
    (``ToolNode``). For each claim it calls ``lookup_claim(claim_id)`` to read
    the claim file, works out the payable amount under the synthetic claims
    guideline CPG-7, and proposes ``approve_claim_payment(claim_id, amount)``.

The decision being guarded
    Whether a proposed payment runs without an adjuster. ``add_cli_guard``
    inserts a guard node between the agent and the tools. The guard scores
    the proposal with an evidence model and a calibrated ``Gate``:

    - allow: the payment runs immediately;
    - escalate: the graph pauses with ``interrupt()`` in the review node, and
      an adjuster approves, edits the amount, or rejects it;
    - block: the call is refused with an error ``ToolMessage`` (a local-mode
      Gate never blocks: it auto-approves or escalates).

    ``lookup_claim`` is read-only and has no rule, so it always runs.

Primitive and guarantee
    ``Gate(guarantee="risk_high_probability", target=0.05, delta=0.10)``,
    calibrated with RCPS on 320 labelled historical decisions. The guarantee
    card reads: "With 90% confidence, the rate of decisions that are
    auto-approved and wrong is at most 0.05 (RCPS, Hoeffding-Bentkus,
    n=320)." That is, at most 5% of payment requests like the calibration
    set end up auto-approved and wrong, with 90% confidence over the draw of
    the calibration set. It is a statement about the rate over many
    requests, not about any single payment, which is why every escalation
    goes to a person.

Data
    ``data/insurance_claims.jsonl`` (synthetic), written by
    ``data/generate_insurance_claims.py``. The guard's context is built by
    ``claim_context`` from that module, the same function that built the
    calibration contexts, so live requests and calibration examples are
    rendered identically.

Running it
    Mock mode is the default: keyless, offline, deterministic, a few seconds::

        python examples/agents/langgraph/insurance_claims_graph.py

    Real providers (keys come from environment variables, see
    ``examples/agents/.env.example``)::

        pip install "cci-sdk[langgraph,openai]" langchain-openai
        export OPENAI_API_KEY=...          # never commit keys
        python examples/agents/langgraph/insurance_claims_graph.py --provider openai

        python ... --provider azure        # AZURE_OPENAI_API_KEY, _ENDPOINT, _DEPLOYMENT
        python ... --provider anthropic    # ANTHROPIC_API_KEY; pip install langchain-anthropic "cci-sdk[anthropic]"
        python ... --provider gemini       # GOOGLE_API_KEY; pip install langchain-google-genai

    Gemma on vLLM (tool calling needs the parser flags and the Gemma 4 tool
    chat template from the vLLM repository, or the agent never calls a tool
    and the guard never runs; one server can be the agent and the evidence
    model)::

        vllm serve google/gemma-4-12B-it --max-logprobs 20 --generation-config vllm \\
            --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \\
            --chat-template examples/tool_chat_template_gemma4.jinja
        python ... --provider vllm         # VLLM_BASE_URL, VLLM_MODEL (default google/gemma-4-12B-it)

    Gemma on SGLang::

        python -m sglang.launch_server --model-path google/gemma-4-12B-it --port 30000 \\
            --tool-call-parser gemma4 --reasoning-parser gemma4
        python ... --provider sglang       # SGLANG_BASE_URL, SGLANG_MODEL

    ``--evidence-provider`` scores payments with a different model from the
    agent, for example ``--provider anthropic --evidence-provider vllm``.
    ``--interactive`` makes you the adjuster. ``--recalibrate`` rebuilds the
    cached profile.

Calibration cost
    One-time and cached in ``--store`` (default ``.cli_profiles`` next to
    this file): 320 requests with a logprob (L1) evidence model (OpenAI
    gpt-4.1-mini, Azure, vLLM, SGLang), or 2,560 with a sampling (L0)
    evidence model (Claude, Gemini; 8 samples per example). Each live
    payment decision then costs 1 request (L1) or 8 (L0).

Expected output (mock mode)
    C-3001 (burst pipe, complete file): ALLOW, the payment of 1,350.00 runs.
    C-3002 (jewelry theft, the agent ignored the sub-limit): ESCALATE, the
    graph pauses, the adjuster edits the amount to 1,000.00, and it runs.
    C-3003 (theft 9 days after inception, fraud indicators): ESCALATE, the
    adjuster rejects, and no payment runs. Each case prints the audit record
    the checkpoint keeps in ``state["cli_guard"]``. A second run reuses the
    cached profile and prints identical decisions.

All data is synthetic. Not claims-handling or legal advice.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional

AGENTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS_DIR))
sys.path.insert(0, str(AGENTS_DIR / "data"))

from _shared import calibration, display, providers  # noqa: E402
from _shared.review import format_request  # noqa: E402

import generate_insurance_claims as claims_data  # noqa: E402  (the shared context builder)

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import START, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.types import Command  # noqa: E402

from cli_sdk import Gate  # noqa: E402
from cli_sdk.integrations import GuardRule, ToolGuard  # noqa: E402
from cli_sdk.integrations.langgraph import CLIGuardState, add_cli_guard, pending_review  # noqa: E402
from cli_sdk.local import LocalCLIClient  # noqa: E402

DATASET = "insurance_claims.jsonl"

PAYMENT_GATE = Gate(
    instructions=(
        "Under claims payment guideline CPG-7, is paying the proposed amount on this claim correct? "
        "Answer true only if the policy covers the loss, no referral trigger applies, the required "
        "documents are present, and the amount equals the payable amount."
    ),
    calibration_profile="claim-payments-v1",
    guarantee="risk_high_probability",
    target=0.05,
    delta=0.10,
)

SYSTEM_PROMPT = (
    "You are a claims-processing assistant for a homeowners insurer. For the claim you are given: "
    "1) call lookup_claim to read the claim file; 2) work out the payable amount under the guideline "
    "below; 3) call approve_claim_payment with the claim id and that amount; 4) tell the adjuster in "
    "one or two sentences what happened. A calibrated check decides whether a payment runs "
    "automatically or goes to an adjuster; if a payment does not run, do not retry it.\n\n"
    + claims_data.GUIDELINE
)

# The claims system (synthetic). The guard's context builder and the lookup_claim
# tool read the same records.
CLAIMS: dict[str, dict[str, Any]] = {
    "C-3001": {
        "claim_id": "C-3001",
        "policy_form": "HO-3 homeowners (synthetic)",
        "peril": "burst pipe (sudden water discharge)",
        "loss_description": "Supply line under the kitchen sink burst; water damage to flooring and cabinets.",
        "loss_date": "2026-08-19",
        "reported_date": "2026-08-20",
        "policy_inception": "2025-11-03",
        "policy_expiry": "2026-11-03",
        "endorsements": ["water backup"],
        "coverage_limit": 350000.0,
        "sub_limit": None,
        "deductible": 500.0,
        "assessed_loss": 1850.0,
        "prior_claims_36m": 0,
        "documents": ["proof of loss", "photos", "repair estimate or invoice"],
        "fraud_indicators": [],
        "adjuster_note": "Field inspection completed; damage consistent with the reported cause.",
    },
    "C-3002": {
        "claim_id": "C-3002",
        "policy_form": "HO-3 homeowners (synthetic)",
        "peril": "theft of jewelry",
        "loss_description": "Rings and a watch taken from the bedroom during a break-in.",
        "loss_date": "2026-07-02",
        "reported_date": "2026-07-03",
        "policy_inception": "2025-09-15",
        "policy_expiry": "2026-09-15",
        "endorsements": [],
        "coverage_limit": 250000.0,
        "sub_limit": {"applies_to": "jewelry and watches (theft)", "amount": 1500.0},
        "deductible": 500.0,
        "assessed_loss": 4200.0,
        "prior_claims_36m": 0,
        "documents": ["proof of loss", "photos", "repair estimate or invoice", "police report"],
        "fraud_indicators": [],
        "adjuster_note": "Desk review; estimate reviewed against regional pricing.",
    },
    "C-3003": {
        "claim_id": "C-3003",
        "policy_form": "HO-3 homeowners (synthetic)",
        "peril": "theft of electronics",
        "loss_description": "Laptop, tablet and camera taken during a daytime break-in.",
        "loss_date": "2026-08-26",
        "reported_date": "2026-08-27",
        "policy_inception": "2026-08-17",
        "policy_expiry": "2027-08-17",
        "endorsements": [],
        "coverage_limit": 250000.0,
        "sub_limit": None,
        "deductible": 500.0,
        "assessed_loss": 4980.0,
        "prior_claims_36m": 3,
        "documents": ["proof of loss", "photos", "repair estimate or invoice"],
        "fraud_indicators": ["receipts dated after the date of loss",
                             "policy limits increased shortly before the loss"],
        "adjuster_note": "",
    },
}

SCENARIOS = (
    ("C-3001", "burst pipe, complete file, no referral triggers"),
    ("C-3002", "jewelry theft; the agent proposes the full loss and misses the sub-limit"),
    ("C-3003", "electronics theft 9 days after inception, prior claims, fraud indicators"),
)

# What the scripted mock agent proposes for each claim (--provider mock).
MOCK_PROPOSALS = {"C-3001": 1350.00, "C-3002": 3700.00, "C-3003": 4480.00}

# What the simulated adjuster decides for each escalation (the resume values).
ADJUSTER_SCRIPT: dict[str, dict[str, Any]] = {
    "C-3002": {"action": "edit", "args": {"amount": 1000.00},
               "note": "Jewelry theft sub-limit applies: 1,500.00 less the 500.00 deductible."},
    "C-3003": {"action": "reject",
               "note": "Loss 9 days after inception with two fraud indicators: refer to SIU, no payment."},
}


# ---------------------------------------------------------------------------
# mock evidence model: reads the claim file like a fallible reviewer, never a label
# ---------------------------------------------------------------------------


def _noise(case_id: str, salt: str) -> float:
    """Deterministic standard-normal noise keyed on the case id."""
    return random.Random(f"{case_id}:{salt}").gauss(0.0, 1.0)


def payment_evidence_scorer(context: Mapping[str, Any], instructions: Any,
                            options: Mapping[str, Any]) -> dict[str, float]:
    """P(paying the proposed amount is correct), read from the claim file with realistic blind spots.

    It sees only the context the guard evaluates (claim file, proposed amount,
    guideline text); calibration labels never reach it.
    """
    claim = context["claim"]
    amount = float(context["proposed_payment"]["amount"])
    peril = claim["peril"].lower()
    loss = date.fromisoformat(claim["loss_date"])
    inception = date.fromisoformat(claim["policy_inception"])
    expiry = date.fromisoformat(claim["policy_expiry"])
    logit = 2.4
    if not (inception <= loss <= expiry):
        logit -= 3.5
    days_in_force = (loss - inception).days
    if 0 <= days_in_force < 30:
        logit -= 2.6
    elif 0 <= days_in_force < 60:
        logit -= 0.6                      # over-cautious just past the cut-off
    if any(word in peril for word in ("mold", "wear and tear", "earth movement")):
        logit -= 3.0
    elif "gradual" in peril:
        logit -= 1.4                      # blind spot: a leak reads like a covered water loss
    if "sewer" in peril and "water backup" not in claim["endorsements"]:
        logit -= 2.4
    if "flood" in peril and "flood" not in claim["endorsements"]:
        logit -= 3.0
    logit -= 1.3 * len(claim["fraud_indicators"])
    logit -= 0.55 * claim["prior_claims_36m"] + (1.0 if claim["prior_claims_36m"] >= 3 else 0.0)
    missing = [d for d in claims_data.REQUIRED_DOCUMENTS if d not in claim["documents"]]
    logit -= 2.2 * len(missing)
    if "theft" in peril and claims_data.THEFT_DOCUMENT not in claim["documents"]:
        logit -= 0.9                      # under-weights the police-report rule
    note = (claim.get("adjuster_note") or "").lower()
    if any(word in note for word in ("unsigned", "low resolution", "approximate", "pending")):
        logit -= 1.0

    def amount_term(apply_sub_limit: bool) -> float:
        cap = min(float(claim["assessed_loss"]), float(claim["coverage_limit"]))
        if apply_sub_limit and claim.get("sub_limit"):
            cap = min(cap, float(claim["sub_limit"]["amount"]))
        expected = max(0.0, cap - float(claim["deductible"]))
        error = abs(amount - expected) / max(expected, 100.0)
        return 0.5 if error <= 0.002 else -min(3.2, 1.2 + 10.0 * error)

    noise = 0.9 * _noise(claim["claim_id"], "payment-evidence")
    if claim.get("sub_limit"):
        # Unsure whether the sub-limit applies: the model splits its belief.
        p_true = (0.55 / (1.0 + math.exp(-(logit + amount_term(True) + noise)))
                  + 0.45 / (1.0 + math.exp(-(logit + amount_term(False) + noise))))
    else:
        p_true = 1.0 / (1.0 + math.exp(-(logit + amount_term(False) + noise)))
    return {"true": p_true, "false": 1.0 - p_true}


# ---------------------------------------------------------------------------
# tools, guard and graph
# ---------------------------------------------------------------------------


def make_tools(ledger: list[dict[str, Any]]) -> list[Any]:
    """The agent's tools; ``ledger`` records every payment that actually runs."""

    @tool
    def lookup_claim(claim_id: str) -> str:
        """Read the claim file for a claim id (read-only)."""
        claim = CLAIMS.get(claim_id)
        return json.dumps(claim if claim else {"error": f"no claim {claim_id}"}, sort_keys=True)

    @tool
    def approve_claim_payment(claim_id: str, amount: float) -> str:
        """Issue a claim payment of `amount` dollars on `claim_id`."""
        ledger.append({"claim_id": claim_id, "amount": round(float(amount), 2)})
        return f"Payment of {float(amount):,.2f} issued on claim {claim_id} (ledger entry {len(ledger)})."

    return [lookup_claim, approve_claim_payment]


def build_guard(client: LocalCLIClient) -> ToolGuard:
    """One rule: payments are decided by the calibrated Gate; other tools run freely."""
    return ToolGuard(client, [
        GuardRule(
            tool="approve_claim_payment",
            query=PAYMENT_GATE,
            # Same builder as the calibration examples: same keys, same guideline text.
            context=lambda args: claims_data.claim_context(CLAIMS[args["claim_id"]], args["amount"]),
        ),
    ])


def build_graph(model: Any, guard: ToolGuard, tools: list[Any], checkpointer: Any = None) -> Any:
    """agent -> cli_guard -> tools | cli_human_review | cli_blocked, with a checkpointer for interrupts."""
    llm = model.bind_tools(tools)

    def agent(state: CLIGuardState) -> dict[str, Any]:
        return {"messages": [llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}

    builder = StateGraph(CLIGuardState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_edge("tools", "agent")
    add_cli_guard(builder, guard)            # agent -> cli_guard -> tools / review / blocked
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


class ScriptedClaimsAgent(GenericFakeChatModel):
    """LangChain's fake chat model, scripted per claim for --provider mock.

    It proposes the scripted tool calls in order, then writes its closing
    note from the payment tool's result, as a real model would.
    """

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> "ScriptedClaimsAgent":
        return self  # GenericFakeChatModel.bind_tools raises NotImplementedError

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        last = messages[-1] if messages else None
        if isinstance(last, ToolMessage) and last.name == "approve_claim_payment":
            if last.status == "error":
                body = json.loads(last.content)
                text = ("No payment was made: the adjuster declined the proposal. "
                        f"Adjuster note: {body.get('reviewer_note') or body.get('reason')} The claim stays open.")
            else:
                text = f"{last.content} The claim can move to closure review."
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def agent_model(provider: str, claim_id: str) -> Any:
    """The chat model that drives the agent node."""
    if provider != "mock":
        from _shared.langchain_models import chat_model

        return chat_model(provider)
    script = [
        AIMessage(content="", tool_calls=[{"name": "lookup_claim", "args": {"claim_id": claim_id},
                                           "id": f"call-{claim_id}-lookup"}]),
        AIMessage(content="", tool_calls=[{"name": "approve_claim_payment",
                                           "args": {"claim_id": claim_id, "amount": MOCK_PROPOSALS[claim_id]},
                                           "id": f"call-{claim_id}-pay"}]),
    ]
    return ScriptedClaimsAgent(messages=iter(script), disable_streaming=True)


# ---------------------------------------------------------------------------
# the adjusters' review queue (stand-in)
# ---------------------------------------------------------------------------


def _ask_amount() -> float:
    while True:
        try:
            return round(float(input("    amount to pay: ").replace(",", "")), 2)
        except ValueError:
            print("    enter a number, for example 1000.00")


class AdjusterDesk:
    """Scripted adjuster decisions, or yours with --interactive. Returns LangGraph resume values."""

    def __init__(self, *, interactive: bool = False, script: Optional[Mapping[str, Mapping[str, Any]]] = None):
        self.interactive = interactive
        self.script = dict(script or {})

    def decide(self, case_id: str, review: Mapping[str, Any]) -> dict[str, Any]:
        print(f"    review request for {case_id}:")
        for call in review["calls"]:
            for line in format_request(call).splitlines():
                print(f"      {line}")
        if self.interactive:
            choice = input("    adjuster: [a]pprove, [e]dit amount, or [r]eject? ").strip().lower()
            note = input("    note for the audit trail: ").strip() or None
            if choice.startswith("a"):
                decision: dict[str, Any] = {"action": "approve", "note": note}
            elif choice.startswith("e"):
                decision = {"action": "edit", "args": {"amount": _ask_amount()}, "note": note}
            else:
                decision = {"action": "reject", "note": note}
        else:
            decision = dict(self.script.get(case_id, {"action": "reject", "note": "no scripted decision"}))
            shown = decision["action"]
            if decision["action"] == "edit":
                shown += f" amount to {decision['args']['amount']:,.2f}"
            print(f"    [simulated adjuster] {shown}: {decision.get('note')}")
        return decision


# ---------------------------------------------------------------------------
# running the scenarios
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph claims agent with a calibrated payment guard.")
    providers.add_arguments(parser)
    return parser.parse_args(argv)


def build_client(args: argparse.Namespace) -> LocalCLIClient:
    evidence_provider = args.evidence_provider or args.provider
    evidence = providers.evidence_backend(evidence_provider, mock_scorer=payment_evidence_scorer)
    return LocalCLIClient(evidence, store=calibration.default_store(__file__, args.store),
                          sample_count=providers.sample_count(evidence_provider))


def _text(content: Any) -> str:
    """Message content as text (some providers return a list of content blocks)."""
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()


def process_claim(graph: Any, guard: ToolGuard, claim_id: str, desk: AdjusterDesk,
                  ledger: list[dict[str, Any]], max_reviews: int = 3) -> dict[str, Any]:
    """Run one claim through the graph, answering each review interrupt the guard raises."""
    config = {"configurable": {"thread_id": f"claim-{claim_id}"}}
    before, seen = len(ledger), len(guard.log)
    request = HumanMessage(f"Process claim {claim_id}: look up the file and propose the payment under CPG-7.")
    out = graph.invoke({"messages": [request]}, config)
    decisions = []
    while True:
        for proposal in guard.log[seen:]:
            print(f"    agent proposes: {proposal.tool}({json.dumps(dict(proposal.arguments), sort_keys=True)})")
        display.decisions(guard.log[seen:])
        seen = len(guard.log)
        review = pending_review(out)
        if review is None:
            break
        if len(decisions) == max_reviews:     # a model that keeps re-proposing: leave the thread paused
            print(f"    still paused after {max_reviews} reviews; leaving the thread for the adjusters' queue")
            break
        print(f"    graph paused at {graph.get_state(config).next[0]} (thread {config['configurable']['thread_id']})")
        decisions.append(desk.decide(claim_id, review))
        out = graph.invoke(Command(resume=decisions[-1]), config)
    state = graph.get_state(config).values
    return {
        "case_id": claim_id,
        "audit": state.get("cli_guard"),
        "review": decisions[-1] if decisions else None,
        "executed": ledger[before:],
        "final": _text(state["messages"][-1].content),
    }


def audit_lines(audit: Optional[Mapping[str, Any]]) -> list[str]:
    """A readable rendering of the JSON record the checkpoint keeps in state["cli_guard"]."""
    if not audit:
        return ["none: the agent proposed no tool call, so the guard never ran"]
    lines = [f"action: {audit['action']}"]
    for call in audit["calls"]:
        lines.append(f"call {call['id']}: {call['tool']}({json.dumps(call['arguments'], sort_keys=True)}) "
                     f"-> {call['action']}")
        evidence, card = call.get("evidence") or {}, call.get("guarantee") or {}
        if evidence:
            lo, hi = evidence.get("venn_abers") or (None, None)
            va = f", Venn-Abers [{lo:.3f}, {hi:.3f}]" if lo is not None else ""
            lines.append(f"  evidence: confidence {evidence['confidence']:.3f}, "
                         f"threshold {evidence['threshold']:.3f}{va}")
        if card:
            lines.append(f"  guarantee: {card['type']} via {card['method']}, profile "
                         f"{card['calibration_profile']} (n={card['calibration_n']})")
    review = audit.get("review")
    if review:
        lines.append(f"review: {review['outcome']}; note: {review.get('note')}")
        for call in review.get("executed_calls") or []:
            lines.append(f"  executed as: {call['tool']}({json.dumps(call['arguments'], sort_keys=True)})")
    return lines


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    client = build_client(args)
    agent_provider = args.provider
    if agent_provider != "mock":
        providers.settings(agent_provider).require_key()
    examples = calibration.load_jsonl(DATASET)
    [profile] = calibration.ensure_calibrated(client, [PAYMENT_GATE], examples, recalibrate=args.recalibrate)
    print(f"calibration profile {profile.name}: {profile.method}, n={profile.n} labelled decisions "
          f"(at least {profile.minimum_n} needed for target={PAYMENT_GATE.target:.2f}, "
          f"delta={PAYMENT_GATE.delta:.2f}), status={profile.status}")
    print(f"evidence model: {client.backend.name} (access level {client.backend.access_level})")

    guard = build_guard(client)
    desk = AdjusterDesk(interactive=args.interactive, script=ADJUSTER_SCRIPT)
    ledger: list[dict[str, Any]] = []
    results = []
    for claim_id, description in SCENARIOS:
        display.scenario(claim_id, description)
        tools = make_tools(ledger)
        graph = build_graph(agent_model(agent_provider, claim_id), guard, tools)
        result = process_claim(graph, guard, claim_id, desk, ledger)
        for entry in result["executed"]:
            print(f"    payment executed: {entry['amount']:,.2f} on {entry['claim_id']}")
        if not result["executed"]:
            print("    payment executed: none")
        print(f"    agent: {result['final']}")
        print('    audit record in the checkpoint (state["cli_guard"]):')
        for line in audit_lines(result["audit"]):
            print(f"      {line}")
        results.append(result)
    return results


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    display.banner("LangGraph: claim payments behind a calibrated guard node (RCPS Gate)", "insurance")
    run(args)
    display.footer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
