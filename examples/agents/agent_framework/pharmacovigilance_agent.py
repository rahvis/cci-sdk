"""Drug-safety case processing assistant on Microsoft Agent Framework, with a calibrated seriousness guard.

What the agent does
    An ``agent_framework.Agent`` reads an adverse-event case and calls
    ``submit_expedited_report(case_id, seriousness_grade)`` when it judges the
    case may qualify for expedited reporting. All data is synthetic, the
    product ``CLX-101`` is fictitious, and the grading guide is invented for
    the example; nothing here is medical or regulatory advice.

The decision being guarded
    Submitting an expedited report. ``CLIGuardMiddleware`` (an Agent
    Framework ``FunctionMiddleware``) checks every proposed
    ``submit_expedited_report`` call before it runs, against the case as
    recorded in the safety database (never the agent's own summary of it).

Primitive and guarantee
    ``Interval`` over the ordered levels grade_1 to grade_5 (mild, moderate,
    severe, life-threatening, fatal), ordinal APS, ``alpha=0.10``. For cases
    like the calibration set, the interval contains the grade an assessor
    records at least 90% of the time. It is a rate over many cases; it does
    not make any single grading correct. The rule maps the whole interval:

    - entirely grade_3 to grade_5 (serious under the synthetic guide): the
      report is submitted automatically. Over-reporting is the conservative
      direction.
    - entirely grade_1 to grade_2: the call is blocked; the case is not
      expedited automatically and stays in routine case processing and
      periodic reporting.
    - crossing the boundary: the run pauses for a safety physician (the
      framework's native tool approval), in the same process and session.

    Because the interval misses the recorded grade for at most 10% of cases
    like the calibration set, at most 10% of all such cases can be blocked
    while their recorded grade is serious. That is a demonstration setting:
    a real deployment would choose a far smaller alpha for this direction,
    or route blocked cases to routine physician review as well.

    The guard decides whether the report is expedited. It does not check the
    grade the agent writes into the report, so the calibrated interval should
    travel with the case for the physician's routine review.

Install
    ``pip install "cci-sdk[agent-framework,openai]" agent-framework-openai``,
    plus ``agent-framework-anthropic`` or ``agent-framework-gemini`` for those
    agent models (both are beta connectors; Python 3.10 or later).

How to run (from sdk/python)
    Mock mode, keyless and offline (the default; a few seconds)::

        python examples/agents/agent_framework/pharmacovigilance_agent.py

    Real providers read keys from environment variables (see
    examples/agents/.env.example); a placeholder key stops the script with a
    message before any request::

        OPENAI_API_KEY=... python examples/agents/agent_framework/pharmacovigilance_agent.py --provider openai
        python examples/agents/agent_framework/pharmacovigilance_agent.py --provider azure
        python examples/agents/agent_framework/pharmacovigilance_agent.py --provider anthropic --evidence-provider openai
        python examples/agents/agent_framework/pharmacovigilance_agent.py --provider gemini

    Gemma on vLLM or SGLang (open weights; tool calling must be switched on
    for the agent, and logprobs capped at 20 for the evidence model)::

        vllm serve google/gemma-4-12B-it --max-logprobs 20 --generation-config vllm \\
            --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \\
            --chat-template examples/tool_chat_template_gemma4.jinja
        python examples/agents/agent_framework/pharmacovigilance_agent.py --provider vllm

        python -m sglang.launch_server --model-path google/gemma-4-12B-it --port 30000 \\
            --tool-call-parser gemma4 --reasoning-parser gemma4
        python examples/agents/agent_framework/pharmacovigilance_agent.py --provider sglang

    Other flags: ``--evidence-provider``, ``--store DIR``, ``--recalibrate``
    and ``--interactive`` (you act as the safety physician).

Calibration cost
    One-time, then cached in ``.cli_profiles``: the 300 labelled cases in
    data/adverse_events.jsonl cost 300 requests to a logprob (L1) evidence
    model (OpenAI, Azure OpenAI, vLLM, SGLang) or 2,400 requests to a
    sampling (L0) model (Anthropic, Gemini; 8 samples per case). Each guard
    check afterwards costs one request (L1) or eight (L0).

Expected output (mock mode)
    AE-8001 hospital admission, interval [grade_3, grade_3], ALLOW, report
    submitted; AE-8002 overnight observation, interval [grade_2, grade_3],
    ESCALATE, physician approves, report submitted; AE-8003 mild rash,
    interval [grade_1, grade_1], BLOCK, no expedited report. The scripted
    agent is deliberately over-cautious and proposes a report for every case,
    which is the failure the block rule exists for. With a real model the
    proposals, and so the decisions, can differ (a real model may not
    propose a report for AE-8003 at all).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Optional

AGENTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS_DIR))
sys.path.insert(0, str(AGENTS_DIR / "data"))

from agent_framework import Agent, tool  # noqa: E402
from pydantic import Field  # noqa: E402

import generate_adverse_events as ae_data  # noqa: E402
from _shared import calibration, display, providers  # noqa: E402
from _shared.maf_models import chat_client  # noqa: E402
from _shared.review import Reviewer  # noqa: E402
from cli_sdk import Interval  # noqa: E402
from cli_sdk.evidence import EvidenceError  # noqa: E402
from cli_sdk.integrations import GuardDecision, GuardRule, ToolGuard  # noqa: E402
from cli_sdk.integrations.agent_framework import CLIGuardMiddleware, pending_reviews, review_message  # noqa: E402
from cli_sdk.local import LocalCLIClient  # noqa: E402

DATASET = "adverse_events.jsonl"
TOOL = "submit_expedited_report"
SERIOUS = ["grade_3", "grade_4", "grade_5"]
NON_SERIOUS = ["grade_1", "grade_2"]

SERIOUSNESS_QUERY = Interval(
    instructions={
        "question": "What seriousness grade does the grading guide assign to this adverse-event case?",
        "grading_guide": list(ae_data.GRADING_GUIDE),
    },
    levels=list(ae_data.GRADES),
    calibration_profile="ae-seriousness-v1",
    alpha=0.10,
)

# The safety database the guard reads (a stand-in for your case-management system).
CASES: dict[str, dict[str, Any]] = {
    "AE-8001": {
        "case_id": "AE-8001", "event_term": "acute kidney injury", "onset_days_after_dose": 6,
        "hospitalization": "admitted to hospital", "treatment": "intravenous therapy or procedure",
        "lab_changes": "severe abnormality", "daily_activities": "unable to perform self-care",
        "outcome": "recovering",
        "narrative": "Acute kidney injury reported 6 days after a dose of CLX-101 (synthetic). Hospital "
                     "course: admitted to hospital. Treated with intravenous therapy or procedure. Outcome at "
                     "last follow-up: recovering.",
    },
    "AE-8002": {
        "case_id": "AE-8002", "event_term": "dizziness", "onset_days_after_dose": 2,
        "hospitalization": "observation stay under 24 hours", "treatment": "prescription medicine",
        "lab_changes": "moderate abnormality", "daily_activities": "some activities limited",
        "outcome": "recovering",
        "narrative": "Dizziness reported 2 days after a dose of CLX-101 (synthetic). Kept overnight for "
                     "monitoring. Treated with prescription medicine. Outcome at last follow-up: recovering.",
    },
    "AE-8003": {
        "case_id": "AE-8003", "event_term": "rash", "onset_days_after_dose": 3,
        "hospitalization": "none", "treatment": "over-the-counter medicine", "lab_changes": "none",
        "daily_activities": "not affected", "outcome": "recovered",
        "narrative": "Rash reported 3 days after a dose of CLX-101 (synthetic). Treated with over-the-counter "
                     "medicine. Outcome at last follow-up: recovered.",
    },
}
SCENARIOS = (
    ("AE-8001", "hospital admission with intravenous therapy"),
    ("AE-8002", "overnight observation stay, borderline"),
    ("AE-8003", "mild rash treated over the counter"),
)
# The scripted (mock) agent is deliberately over-cautious: it proposes a report for every case.
SCRIPTED_PROPOSALS = {"AE-8001": "grade_3", "AE-8002": "grade_3", "AE-8003": "grade_2"}
# The stand-in safety physician's decisions for escalated cases (use --interactive to decide yourself).
REVIEW_SCRIPT = {"AE-8002": True}

INSTRUCTIONS = (
    "You are a drug-safety case processing assistant. Read the adverse-event case in the user's "
    "message and grade it with the grading guide below. Call submit_expedited_report once when the case "
    "may qualify for expedited reporting (grade_3 to grade_5); when in doubt, submit, because "
    "over-reporting is the conservative direction. If the tool says the report was sent for physician "
    "review or was not submitted, do not call it again; tell the user in one or two sentences. Never "
    "invent case facts.\n\n" + "\n".join(ae_data.GRADING_GUIDE)
)

# Stand-in for the regulatory submission gateway: case id -> grade in the expedited report.
SUBMITTED_REPORTS: dict[str, str] = {}


@tool
def submit_expedited_report(
    case_id: Annotated[str, Field(description="The adverse-event case id, for example AE-8001.")],
    seriousness_grade: Annotated[Literal["grade_1", "grade_2", "grade_3", "grade_4", "grade_5"],
                                 Field(description="Seriousness grade under the grading guide.")],
) -> str:
    """Submit an expedited safety report for an adverse-event case."""
    SUBMITTED_REPORTS[case_id] = seriousness_grade
    return f"Expedited report for case {case_id} submitted with {seriousness_grade}."


def guard_context(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The evaluation context for a proposed call.

    It is the case from the safety database, rendered by the same
    ``case_context`` function that built the calibration data, so inference
    and calibration contexts have identical keys and wording.
    """
    return ae_data.case_context(CASES[arguments["case_id"]])


def build_guard(client: LocalCLIClient) -> ToolGuard:
    return ToolGuard(client, [GuardRule(
        tool=TOOL,
        query=SERIOUSNESS_QUERY,
        context=guard_context,
        allow_levels=SERIOUS,        # the whole interval is serious: submit automatically
        block_levels=NON_SERIOUS,    # the whole interval is non-serious: routine periodic reporting
    )])                              # anything that crosses the boundary: a safety physician decides


# ---------------------------------------------------------------------------
# Mock mode: a scripted agent model and a fallible, keyless evidence model
# ---------------------------------------------------------------------------

# Event terms that make the mock reader lean towards a higher grade, whatever the findings say.
ALARMING_TERMS = frozenset({"anaphylaxis", "ventricular arrhythmia", "neutropenic sepsis", "status epilepticus",
                            "cardiac arrest", "respiratory failure", "syncope", "pancreatitis"})


def _deterministic_normal(*parts: Any) -> float:
    """A standard-normal draw fixed by ``parts`` (Box-Muller on a hash)."""
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()
    u1 = (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 2)
    u2 = int.from_bytes(digest[8:16], "big") / 2**64
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def seriousness_scorer(context: Mapping[str, Any], instructions: Any, options: Mapping[str, Any]) -> dict[str, float]:
    """How a fallible safety-coding model reads the case (mock evidence). It never sees the label.

    It applies the guide to the structured findings, reads the narrative to
    place observation stays between grade_2 and grade_3 (and is less sure
    about them), leans higher for alarming event terms and unresolved
    outcomes, and carries a fixed per-case error keyed on the case id.
    """
    observation = context["hospitalization"] == ae_data.HOSPITALIZATION[2]
    narrative = context["narrative"]
    if "precaution" in narrative:
        observation_position = 1.25
    elif "not settled" in narrative:
        observation_position = 1.75
    else:
        observation_position = 1.5
    position = float(max(
        ae_data.OUTCOME_GRADE[context["outcome"]],
        observation_position if observation else ae_data.HOSPITALIZATION_GRADE[context["hospitalization"]],
        ae_data.TREATMENT_GRADE[context["treatment"]],
        ae_data.LAB_GRADE[context["lab_changes"]],
        ae_data.ACTIVITY_GRADE[context["daily_activities"]],
    ))
    if context["outcome"] == "not yet recovered":
        position += 0.15
    if context["event_term"] in ALARMING_TERMS:
        position += 0.2
    position += (0.25 if observation else 0.15) * _deterministic_normal("ae-reader", context["case_id"])
    spread = 0.55 if observation else 0.33
    return {key: math.exp(-((i - position) ** 2) / (2 * spread**2)) for i, key in enumerate(options)}


def scripted_proposal(request: str) -> Optional[tuple[str, dict[str, Any]]]:
    """The scripted agent model: propose a report for the case named in the request."""
    match = re.search(r"\bAE-\d{4}\b", request)
    if match is None or match.group(0) not in SCRIPTED_PROPOSALS:
        return None
    case_id = match.group(0)
    return TOOL, {"case_id": case_id, "seriousness_grade": SCRIPTED_PROPOSALS[case_id]}


# ---------------------------------------------------------------------------
# Running the scenarios
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    case_id: str
    proposed: dict[str, Any]
    action: str
    interval: list[str]
    executed: bool
    review: Optional[bool]
    agent_text: str


def user_message(case_id: str) -> str:
    case = guard_context({"case_id": case_id})
    return (f"Process adverse-event case {case_id}.\n"
            f"Case record:\n{json.dumps(case, indent=2, sort_keys=True)}")


def latest_decision(guard: ToolGuard, case_id: str) -> Optional[GuardDecision]:
    for decision in reversed(guard.log):
        if decision.tool == TOOL and decision.arguments.get("case_id") == case_id:
            return decision
    return None


def describe_interval(levels: list[str]) -> str:
    if len(levels) != 2:
        return "unavailable"
    lo, hi = levels
    names = ae_data.GRADE_NAMES
    span = names[lo] if lo == hi else f"{names[lo]} to {names[hi]}"
    return f"[{lo}, {hi}] ({span})"


async def run_case(agent: Agent, guard: ToolGuard, reviewer: Reviewer, case_id: str) -> Outcome:
    session = agent.create_session()
    result = await agent.run(user_message(case_id), session=session)
    decision = latest_decision(guard, case_id)
    if decision is None:
        print("    the agent did not propose an expedited report for this case")
        return Outcome(case_id, {}, "none", [], case_id in SUBMITTED_REPORTS, None, result.text)
    print(f"    agent proposed: {TOOL}({json.dumps(decision.arguments, sort_keys=True)})")
    display.decisions([decision])
    levels = list(decision.to_dict().get("evidence", {}).get("levels", []))
    print(f"           calibrated interval: {describe_interval(levels)}")
    approved: Optional[bool] = None

    reviews = pending_reviews(result)
    if reviews:
        # Paused for a safety physician. Here the session stays in memory and the
        # run resumes in the same process; see credit_underwriting_agent.py for
        # persisting the session to JSON and resuming elsewhere.
        approved = reviewer.decide(case_id, reviews[0])
        result = await agent.run(review_message(reviews[0], approved), session=session)

    executed = case_id in SUBMITTED_REPORTS
    if executed:
        print(f"    tool executed: yes, expedited report submitted ({SUBMITTED_REPORTS[case_id]})")
    elif decision.blocked:
        print("    tool executed: no; not expedited, the case stays in routine periodic reporting")
    else:
        print("    tool executed: no, nothing submitted")
    print(f"    agent: {result.text}")
    return Outcome(case_id, dict(decision.arguments), decision.action, levels, executed, approved, result.text)


def parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    providers.add_arguments(parser)
    return parser.parse_args(argv)


def run(argv: Optional[list[str]] = None, *, reviewer: Optional[Reviewer] = None) -> list[Outcome]:
    """Run every scenario and return what happened (``main`` prints the same)."""
    args = parse_args(argv)
    evidence_provider = args.evidence_provider or args.provider
    # Build both models first: a placeholder key stops here, before any request is sent.
    evidence = providers.evidence_backend(evidence_provider, mock_scorer=seriousness_scorer)
    agent_model = chat_client(args.provider, planner=scripted_proposal)

    display.banner("Drug-safety case processing assistant (Microsoft Agent Framework) with a "
                   "calibrated seriousness guard", "healthcare")
    print(f"agent model: {args.provider}; evidence model: {evidence_provider}")
    store = calibration.default_store(__file__, args.store)
    client = LocalCLIClient(evidence, store=store, sample_count=providers.sample_count(evidence_provider))
    examples = calibration.load_jsonl(DATASET)
    try:
        (profile,) = calibration.ensure_calibrated(client, [SERIOUSNESS_QUERY], examples,
                                                   recalibrate=args.recalibrate)
    except EvidenceError as exc:
        raise SystemExit(f"calibration stopped: the {evidence_provider} evidence model failed ({exc})") from exc
    print(f"calibration: {calibration.describe_profile(profile)}")
    print()

    guard = build_guard(client)
    reviewer = reviewer or Reviewer(interactive=args.interactive, script=REVIEW_SCRIPT, role="safety physician")
    agent = Agent(client=agent_model, name="drug-safety-assistant", instructions=INSTRUCTIONS,
                  tools=[submit_expedited_report], middleware=[CLIGuardMiddleware(guard)])

    SUBMITTED_REPORTS.clear()
    outcomes = []
    for case_id, description in SCENARIOS:
        display.scenario(case_id, description)
        outcomes.append(asyncio.run(run_case(agent, guard, reviewer, case_id)))
    display.footer()
    return outcomes


def main(argv: Optional[list[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
