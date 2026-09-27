"""Credit underwriting assistant on Microsoft Agent Framework, with a calibrated risk-tier guard.

What the agent does
    An ``agent_framework.Agent`` reads a consumer-loan application file and
    calls ``assign_risk_tier(application_id, tier)`` to record a tier from A
    (lowest risk) to E (highest risk) in the loan origination system. All
    data is synthetic, and the tiering policy is invented for the example.

The decision being guarded
    Recording the tier. ``CLIGuardMiddleware`` (an Agent Framework
    ``FunctionMiddleware``) checks every proposed ``assign_risk_tier`` call
    before it runs, against the application file from the system of record
    (never the agent's own summary of it).

Primitive and guarantee
    ``Set`` over tiers A to E, method APS, ``alpha=0.10``, ``group_by="channel"``
    (Mondrian calibration). For each application channel (branch, online,
    broker) with enough calibration data, the set contains the tier the
    policy assigns for at least 90% of applications like that channel's
    calibration files. ``GuardRule(allow_labels=["A", "B", "C"],
    match_argument="tier")`` lets the call run only when the set is exactly
    the tier the agent proposed and that tier is A, B or C. Any other set,
    and every D or E proposal, pauses the run for an underwriter (the
    framework's native tool approval). A tier recorded automatically is
    wrong only when its set missed, so at most 10% of applications like the
    calibration files are recorded automatically with a wrong tier. That
    is a rate over all applications, not over the recorded ones, and it
    does not make any single tier decision correct.

Fair lending
    The features exclude protected characteristics and obvious proxies.
    Per-channel calibration shows the coverage guarantee holds within each
    channel, not only on average; it is not a fair-lending analysis, and a
    fair-lending review of the policy and of the guard's escalation rates is
    still required.

Install
    ``pip install "cci-sdk[agent-framework,openai]" agent-framework-openai``,
    plus ``agent-framework-anthropic`` or ``agent-framework-gemini`` for those
    agent models (both are beta connectors; Python 3.10 or later).

How to run (from sdk/python)
    Mock mode, keyless and offline (the default; a few seconds)::

        python examples/agents/agent_framework/credit_underwriting_agent.py

    Real providers read keys from environment variables (see
    examples/agents/.env.example); a placeholder key stops the script with a
    message before any request::

        OPENAI_API_KEY=... python examples/agents/agent_framework/credit_underwriting_agent.py --provider openai
        python examples/agents/agent_framework/credit_underwriting_agent.py --provider azure
        python examples/agents/agent_framework/credit_underwriting_agent.py --provider anthropic --evidence-provider openai
        python examples/agents/agent_framework/credit_underwriting_agent.py --provider gemini

    Gemma on vLLM or SGLang (open weights; tool calling must be switched on
    for the agent, and logprobs capped at 20 for the evidence model)::

        vllm serve google/gemma-4-12B-it --max-logprobs 20 --generation-config vllm \\
            --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \\
            --chat-template examples/tool_chat_template_gemma4.jinja
        python examples/agents/agent_framework/credit_underwriting_agent.py --provider vllm

        python -m sglang.launch_server --model-path google/gemma-4-12B-it --port 30000 \\
            --tool-call-parser gemma4 --reasoning-parser gemma4
        python examples/agents/agent_framework/credit_underwriting_agent.py --provider sglang

    Other flags: ``--evidence-provider`` (a different model scores the
    guard), ``--store DIR`` (profile directory), ``--recalibrate``, and
    ``--interactive`` (you approve or reject escalations).

Calibration cost
    One-time, then cached in ``.cli_profiles``: the 320 labelled files in
    data/credit_applications.jsonl cost 320 requests to a logprob (L1)
    evidence model (OpenAI, Azure OpenAI, vLLM, SGLang) or 2,560 requests to a
    sampling (L0) model (Anthropic, Gemini; 8 samples per file). Each guard
    check afterwards costs one request (L1) or eight (L0).

Expected output (mock mode)
    The per-channel calibration status (broker is the smallest channel,
    so its realized coverage is the least certain), then:
    L-7001 strong file, set {A}, ALLOW, tier recorded;
    L-7002 thin file, set {B, C}, ESCALATE, session saved to JSON, resumed,
    underwriter approves, tier recorded;
    L-7003 weak broker file proposed D, set {D, E}, ESCALATE, underwriter
    rejects, nothing recorded. A second run reuses the stored profile and
    prints the same decisions. With a real model the proposals, and so the
    decisions, can differ.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Optional

AGENTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS_DIR))
sys.path.insert(0, str(AGENTS_DIR / "data"))

from agent_framework import Agent, AgentSession, Content, tool  # noqa: E402
from pydantic import Field  # noqa: E402

import generate_credit_applications as credit_data  # noqa: E402
from _shared import calibration, display, providers  # noqa: E402
from _shared.maf_models import chat_client  # noqa: E402
from _shared.review import Reviewer  # noqa: E402
from cli_sdk import Set  # noqa: E402
from cli_sdk.evidence import EvidenceError  # noqa: E402
from cli_sdk.integrations import GuardDecision, GuardRule, ToolGuard  # noqa: E402
from cli_sdk.integrations.agent_framework import CLIGuardMiddleware, pending_reviews, review_message  # noqa: E402
from cli_sdk.local import LocalCLIClient  # noqa: E402
from cli_sdk.stats.conformal import coverage_confidence_interval  # noqa: E402

DATASET = "credit_applications.jsonl"
TOOL = "assign_risk_tier"

TIER_OPTIONS = {
    "A": "lowest risk: prime pricing",
    "B": "low risk: near-prime pricing",
    "C": "moderate risk: standard pricing",
    "D": "elevated risk: an underwriter must approve",
    "E": "high risk: decline unless an underwriter documents compensating factors",
}

TIER_QUERY = Set(
    instructions={
        "question": "Which risk tier does the tiering policy assign to this consumer-loan application?",
        "tiering_policy": list(credit_data.TIERING_POLICY),
    },
    options=TIER_OPTIONS,
    calibration_profile="credit-tiers-v1",
    alpha=0.10,
    method="APS",
    group_by="channel",
)

# The system of record the guard reads (a stand-in for your loan origination system).
APPLICATIONS: dict[str, dict[str, Any]] = {
    "L-7001": {"application_id": "L-7001", "channel": "branch", "credit_score_band": "800+",
               "debt_to_income_pct": 14, "payment_history": "no late payments", "thin_file": False,
               "income_verification": "verified with payroll or tax documents"},
    "L-7002": {"application_id": "L-7002", "channel": "online", "credit_score_band": "800+",
               "debt_to_income_pct": 24, "payment_history": "no late payments", "thin_file": True,
               "income_verification": "verified with bank statements"},
    "L-7003": {"application_id": "L-7003", "channel": "broker", "credit_score_band": "580-669",
               "debt_to_income_pct": 41,
               "payment_history": "two or more late payments, or one 60-day delinquency",
               "thin_file": False, "income_verification": "verified with bank statements"},
}
SCENARIOS = (
    ("L-7001", "strong file, branch channel"),
    ("L-7002", "thin credit file, online channel"),
    ("L-7003", "weak file submitted by a broker"),
)
# The tier the scripted (mock) model proposes for each file. A real model reads the file itself.
SCRIPTED_PROPOSALS = {"L-7001": "A", "L-7002": "B", "L-7003": "D"}
# The stand-in underwriter's decisions for escalated files (use --interactive to decide yourself).
REVIEW_SCRIPT = {"L-7002": True, "L-7003": False}

INSTRUCTIONS = (
    "You are an underwriting assistant for a consumer lender. Read the application file in the "
    "user's message, apply the tiering policy below, and call assign_risk_tier exactly once with the "
    "tier the policy assigns. If the tool says the call was sent for review or was not carried out, "
    "do not call it again; tell the user in one or two sentences. Never invent application data.\n\n"
    + "\n".join(credit_data.TIERING_POLICY)
)

FAIR_LENDING_NOTE = (
    "Fair-lending note: the features exclude protected characteristics (age, sex, race, ethnicity, "
    "religion, national origin, marital status) and obvious proxies such as ZIP code. Per-channel "
    "calibration shows the coverage guarantee holds within each channel, not only on average. It is "
    "not a fair-lending analysis: review the policy, and the guard's escalation rates by segment, "
    "with your fair-lending and model-risk teams."
)

# Stand-in for the loan origination system: application id -> recorded tier.
TIER_ASSIGNMENTS: dict[str, str] = {}


@tool
def assign_risk_tier(
    application_id: Annotated[str, Field(description="The application id, for example L-7001.")],
    tier: Annotated[Literal["A", "B", "C", "D", "E"],
                    Field(description="Risk tier from A (lowest risk) to E (highest risk).")],
) -> str:
    """Record the risk tier for a consumer-loan application in the loan origination system."""
    TIER_ASSIGNMENTS[application_id] = tier
    return f"Tier {tier} recorded for application {application_id}."


def guard_context(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The evaluation context for a proposed call.

    It is the application file from the system of record, rendered by the
    same ``application_context`` function that built the calibration data,
    so inference and calibration contexts have identical keys and wording.
    """
    return credit_data.application_context(APPLICATIONS[arguments["application_id"]])


def build_guard(client: LocalCLIClient) -> ToolGuard:
    return ToolGuard(client, [GuardRule(
        tool=TOOL,
        query=TIER_QUERY,
        context=guard_context,
        allow_labels=["A", "B", "C"],   # D and E always go to an underwriter
        match_argument="tier",          # the set must be exactly the tier the agent proposed
    )])


# ---------------------------------------------------------------------------
# Mock mode: a scripted agent model and a fallible, keyless evidence model
# ---------------------------------------------------------------------------


def _deterministic_normal(*parts: Any) -> float:
    """A standard-normal draw fixed by ``parts`` (Box-Muller on a hash)."""
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()
    u1 = (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 2)
    u2 = int.from_bytes(digest[8:16], "big") / 2**64
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _interpolate(x: float, points: tuple[tuple[float, float], ...]) -> float:
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return points[-1][1]


def tier_scorer(context: Mapping[str, Any], instructions: Any, options: Mapping[str, Any]) -> dict[str, float]:
    """How a fallible underwriting model reads the file (mock evidence). It never sees the label.

    It applies the policy's points, but reads debt-to-income on a smooth
    scale (so files near a band edge come out between two tiers), is less
    certain on thin files and on broker-submitted files, and carries a
    fixed per-file error keyed on the application id.
    """
    dti = float(str(context["debt_to_income"]).rstrip("%"))
    thin = context["credit_file"] != credit_data.ESTABLISHED_FILE
    stated = context["income_verification"] == credit_data.INCOME_VERIFICATION[2]
    points = (credit_data.BAND_POINTS[context["credit_score_band"]]
              + _interpolate(dti, ((12, 0), (28, 1), (39.5, 2), (47, 3), (58, 5)))
              + credit_data.HISTORY_POINTS[context["payment_history_24_months"]]
              + (2 if thin else 0) + (1 if stated else 0))
    position = _interpolate(points, ((0.5, 0), (2.5, 1), (4.5, 2), (7.0, 3), (10.0, 4)))
    if thin:
        position = max(position, 1.0)
    if stated:
        position = max(position, 2.0)
    if context["payment_history_24_months"] == credit_data.PAYMENT_HISTORY[3]:
        position = max(position, 3.0)
    broker = context["channel"] == "broker"
    error_sd = 0.15 + (0.25 if thin else 0.0) + (0.20 if broker else 0.0)
    spread = 0.50 if thin else (0.42 if broker else 0.33)
    position += error_sd * _deterministic_normal("credit-reader", context["application_id"])
    return {key: math.exp(-((i - position) ** 2) / (2 * spread**2)) for i, key in enumerate(options)}


def scripted_proposal(request: str) -> Optional[tuple[str, dict[str, Any]]]:
    """The scripted agent model: propose the scripted tier for the application named in the request."""
    match = re.search(r"\bL-\d{4}\b", request)
    if match is None or match.group(0) not in SCRIPTED_PROPOSALS:
        return None
    application_id = match.group(0)
    return TOOL, {"application_id": application_id, "tier": SCRIPTED_PROPOSALS[application_id]}


# ---------------------------------------------------------------------------
# Running the scenarios
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    case_id: str
    proposed: dict[str, Any]
    action: str
    calibrated_set: list[str]
    executed: bool
    review: Optional[bool]
    agent_text: str


def user_message(application_id: str) -> str:
    file = guard_context({"application_id": application_id})
    return (f"Assign a risk tier to application {application_id}.\n"
            f"Application file:\n{json.dumps(file, indent=2, sort_keys=True)}")


def latest_decision(guard: ToolGuard, application_id: str) -> Optional[GuardDecision]:
    for decision in reversed(guard.log):
        if decision.tool == TOOL and decision.arguments.get("application_id") == application_id:
            return decision
    return None


def print_channel_status(client: LocalCLIClient) -> int:
    """Print the profile and its per-channel status; return the profile's size."""
    profile = client.get_profile(TIER_QUERY.calibration_profile)
    print(f"calibration: {calibration.describe_profile(profile)}")
    print(f"per-channel (Mondrian) status, alpha={profile.alpha:g}:")
    for channel, status in profile.groups.items():
        lo, hi = coverage_confidence_interval(status.n, profile.alpha)
        print(f"    {channel:<7} n={status.n:<4} {status.status:<12} "
              f"realized coverage 90% interval [{lo:.3f}, {hi:.3f}]")
    print()
    return profile.n


def save_pending(path: Path, session: AgentSession, review: Mapping[str, Any]) -> None:
    """Persist everything needed to resume after the review: session (history and ticket) and request."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"session": session.to_dict(), "request": review["request"].to_dict(), "cli": review["cli"]}
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


async def run_case(make_agent, guard: ToolGuard, reviewer: Reviewer, case_id: str, pending_dir: Path,
                   profile_n: int) -> Outcome:
    agent = make_agent()
    session = agent.create_session()
    result = await agent.run(user_message(case_id), session=session)
    decision = latest_decision(guard, case_id)
    if decision is None:
        print("    the agent did not propose assign_risk_tier for this file")
        return Outcome(case_id, {}, "none", [], case_id in TIER_ASSIGNMENTS, None, result.text)
    print(f"    agent proposed: {TOOL}({json.dumps(decision.arguments, sort_keys=True)})")
    display.decisions([decision])
    record = decision.to_dict()
    card = record.get("guarantee", {})
    if card.get("group"):
        print(f"           group-conditional: channel '{card['group']}', n={card['calibration_n']} "
              f"of the profile's {profile_n} files")
    calibrated_set = list(record.get("evidence", {}).get("set", []))
    approved: Optional[bool] = None

    reviews = pending_reviews(result)
    if reviews:
        # The run is paused. Persist the session and the approval request; the process could exit here.
        path = pending_dir / f"{case_id}.json"
        save_pending(path, session, reviews[0])
        print(f"    run paused for an underwriter; session and approval request saved to "
              f"{path.parent.name}/{path.name}")

        # Later, possibly in another process: load, decide, and resume with a fresh agent.
        saved = json.loads(path.read_text(encoding="utf-8"))
        approved = reviewer.decide(case_id, saved)
        resumed_agent = make_agent()
        resumed_session = AgentSession.from_dict(saved["session"])
        request = Content.from_dict(saved["request"])
        result = await resumed_agent.run(review_message(request, approved), session=resumed_session)
        path.unlink()
        print(f"    resumed from {path.name} with a new agent instance; pending file removed")

    executed = case_id in TIER_ASSIGNMENTS
    recorded = f"yes, tier {TIER_ASSIGNMENTS[case_id]} recorded" if executed else "no, nothing recorded"
    print(f"    tool executed: {recorded}")
    print(f"    agent: {result.text}")
    return Outcome(case_id, dict(decision.arguments), decision.action, calibrated_set, executed, approved,
                   result.text)


def parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    providers.add_arguments(parser)
    return parser.parse_args(argv)


def run(argv: Optional[list[str]] = None, *, reviewer: Optional[Reviewer] = None) -> list[Outcome]:
    """Run every scenario and return what happened (``main`` prints the same)."""
    args = parse_args(argv)
    evidence_provider = args.evidence_provider or args.provider
    # Build both models first: a placeholder key stops here, before any request is sent.
    evidence = providers.evidence_backend(evidence_provider, mock_scorer=tier_scorer)
    agent_model = chat_client(args.provider, planner=scripted_proposal)

    display.banner("Credit underwriting assistant (Microsoft Agent Framework) with a calibrated "
                   "tier guard", "finance")
    print(f"agent model: {args.provider}; evidence model: {evidence_provider}")
    store = calibration.default_store(__file__, args.store)
    client = LocalCLIClient(evidence, store=store, sample_count=providers.sample_count(evidence_provider))
    examples = calibration.load_jsonl(DATASET)
    try:
        calibration.ensure_calibrated(client, [TIER_QUERY], examples, recalibrate=args.recalibrate)
    except EvidenceError as exc:
        raise SystemExit(f"calibration stopped: the {evidence_provider} evidence model failed ({exc})") from exc
    profile_n = print_channel_status(client)

    guard = build_guard(client)
    reviewer = reviewer or Reviewer(interactive=args.interactive, script=REVIEW_SCRIPT, role="underwriter")

    def make_agent() -> Agent:
        return Agent(client=agent_model, name="underwriting-assistant", instructions=INSTRUCTIONS,
                     tools=[assign_risk_tier], middleware=[CLIGuardMiddleware(guard)])

    TIER_ASSIGNMENTS.clear()
    outcomes = []
    for case_id, description in SCENARIOS:
        display.scenario(case_id, description)
        outcomes.append(asyncio.run(run_case(make_agent, guard, reviewer, case_id,
                                             Path(store) / "pending_reviews", profile_n)))
    display.footer()
    for line in textwrap.wrap(FAIR_LENDING_NOTE, 78):
        print(line)
    return outcomes


def main(argv: Optional[list[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
