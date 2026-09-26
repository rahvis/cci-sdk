"""LangGraph discharge-summary workflow with a calibrated claim-verification node.

What the graph does
    A ``StateGraph`` that turns a (synthetic) discharge chart into a
    patient-friendly summary for a patient portal:

    ``draft_summary`` -> ``verify_claims`` -> ``publish``, or
    ``verify_claims`` -> ``clinician_review`` -> ``publish`` or END.

    ``draft_summary`` asks a chat model to write the summary from the chart.
    ``verify_claims`` splits the draft into atomic claims and keeps only the
    ones the chart supports, using the ``Claim`` primitive in local mode.
    When nothing was dropped the verified text is published. When any claim
    was dropped (or no calibrated guarantee is available), the
    ``clinician_review`` node pauses the graph with ``interrupt()`` and shows
    the retained and dropped claims; the clinician approves publishing the
    verified-only text, or rejects it and nothing is published. The portal
    only ever receives retained claims.

The decision being guarded
    Which sentences of a machine-written summary may reach a patient without
    a clinician rewriting them. There is no tool call here, so there is no
    ``ToolGuard``: the check is its own graph node.

Primitive and guarantee
    ``Claim(alpha=0.10)``, calibrated with conformal factuality on 240
    labelled drafts (1,537 labelled claims). The guarantee card reads: "With
    probability at least 90%, every retained claim is supported (conformal
    factuality, n=240)." That is a statement about drafts like the
    calibration set: in at most about 10% of them does an unsupported claim
    survive the filter. It does not make any single summary correct, so the
    clinician stays in the loop and the portal text is still reviewed under
    your clinical governance.

Data
    ``data/discharge_claims.jsonl`` (synthetic), written by
    ``data/generate_discharge_claims.py``, which also holds the written
    labelling protocol. The check's context, ``{"chart": ..., "answer":
    <draft>}``, is built by ``summary_context`` from that module, the same
    function that built the calibration contexts.

Running it
    Mock mode is the default: keyless, offline, deterministic, a few seconds::

        python examples/agents/langgraph/clinical_summary_graph.py

    Real providers (keys come from environment variables, see
    ``examples/agents/.env.example``)::

        pip install "cli-sdk[langgraph,openai]" langchain-openai
        export OPENAI_API_KEY=...          # never commit keys
        python examples/agents/langgraph/clinical_summary_graph.py --provider openai

        python ... --provider azure        # AZURE_OPENAI_API_KEY, _ENDPOINT, _DEPLOYMENT
        python ... --provider anthropic    # ANTHROPIC_API_KEY; pip install langchain-anthropic "cli-sdk[anthropic]"
        python ... --provider gemini       # GOOGLE_API_KEY; pip install langchain-google-genai

    Gemma on vLLM or SGLang (the draft is plain text, so no tool-call parser
    is needed; ``--max-logprobs 20`` lets the claim scores use logprobs)::

        vllm serve google/gemma-4-12B-it --max-logprobs 20 --generation-config vllm
        python ... --provider vllm         # VLLM_BASE_URL, VLLM_MODEL (default google/gemma-4-12B-it)

        python -m sglang.launch_server --model-path google/gemma-4-12B-it --port 30000
        python ... --provider sglang       # SGLANG_BASE_URL, SGLANG_MODEL

    ``--evidence-provider`` scores claims with a different model from the
    drafting model. ``--interactive`` makes you the clinician.
    ``--recalibrate`` rebuilds the cached profile.

Calibration cost
    One-time and cached in ``--store`` (default ``.cli_profiles`` next to
    this file): one support score per labelled claim, so 1,537 requests with
    a logprob (L1) evidence model (OpenAI gpt-4.1-mini, Azure, vLLM,
    SGLang), or 12,296 with a sampling (L0) evidence model (Claude, Gemini;
    8 samples per claim). Prefer an L1 evidence model for Claim. Each live
    summary then costs one request to split the draft plus one (L1) or
    eight (L0) per claim.

Expected output (mock mode)
    D-4001 (faithful pneumonia summary): all 7 claims retained, published
    without review. D-4002 (heart-failure summary with an invented
    metoprolol dose increase and a wrong cardiology date): those 2 claims
    are dropped, the graph pauses for the clinician, who approves
    publishing the 5 verified claims. A second run reuses the cached profile
    and prints identical results.

All data is synthetic. Not medical advice and not a medical device.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, TypedDict

AGENTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS_DIR))
sys.path.insert(0, str(AGENTS_DIR / "data"))

from _shared import calibration, display, providers  # noqa: E402
from _shared.review import Reviewer  # noqa: E402

import generate_discharge_claims as charts_data  # noqa: E402  (the shared context builder)

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Command, interrupt  # noqa: E402

from cli_sdk import Claim  # noqa: E402
from cli_sdk.local import LocalCLIClient  # noqa: E402

DATASET = "discharge_claims.jsonl"

CLAIM_CHECK = Claim(
    instructions=(
        "Keep only the statements in this patient-friendly discharge summary that the discharge "
        "chart supports."
    ),
    calibration_profile="discharge-claims-v1",
    alpha=0.10,
)

DRAFT_PROMPT = (
    "You write patient-friendly hospital discharge summaries from the chart you are given. Write short, "
    "plain sentences in the second person, one fact per sentence. Cover the diagnosis, each medication "
    "with its dose, frequency and whether it is new, continued, changed, held or stopped, the follow-up "
    "appointments, pending results, one home-care instruction and when to come back. Use only facts in "
    "the chart. Plain text only: no headings, lists or markdown."
)

# The chart system (synthetic). Same fields as the calibration charts.
CHARTS: dict[str, dict[str, Any]] = {
    "D-4001": {
        "encounter_id": "D-4001",
        "discharge_date": "2026-09-18",
        "principal_diagnosis": "community-acquired pneumonia",
        "secondary_diagnoses": ["type 2 diabetes"],
        "medications": [
            {"name": "amoxicillin-clavulanate", "dose": 875, "unit": "mg", "frequency": "twice daily",
             "status": "new", "duration": "5 more days"},
            {"name": "metformin", "dose": 1000, "unit": "mg", "frequency": "twice daily", "status": "continued"},
        ],
        "follow_up": [{"with": "primary care", "when": "within 7 days"},
                      {"with": "pulmonology clinic", "when": "2026-10-09"}],
        "pending_results": ["blood cultures"],
        "instructions": ["Finish every dose of the antibiotic.", "Rest and drink plenty of fluids."],
        "return_precautions": ["a temperature above 38.5 C", "worsening shortness of breath", "chest pain"],
    },
    "D-4002": {
        "encounter_id": "D-4002",
        "discharge_date": "2026-09-21",
        "principal_diagnosis": "acute decompensated heart failure",
        "secondary_diagnoses": ["hypertension"],
        "medications": [
            {"name": "furosemide", "dose": 40, "unit": "mg", "frequency": "twice daily", "status": "changed",
             "previous": "40 mg once daily"},
            {"name": "metoprolol succinate", "dose": 50, "unit": "mg", "frequency": "once daily",
             "status": "continued"},
            {"name": "lisinopril", "dose": 10, "unit": "mg", "frequency": "once daily", "status": "held",
             "reason": "held until the follow-up visit"},
        ],
        "follow_up": [{"with": "primary care", "when": "within 7 days"},
                      {"with": "cardiology clinic", "when": "2026-10-08"}],
        "pending_results": ["echocardiogram"],
        "instructions": ["Weigh yourself every morning and write it down.", "Limit salt to 2 grams a day."],
        "return_precautions": ["weight gain of more than 1 kg in a day", "swelling in the legs",
                               "worsening shortness of breath"],
    },
}

SCENARIOS = (
    ("D-4001", "pneumonia; the drafting model writes a faithful summary"),
    ("D-4002", "heart failure; the draft invents a dose increase and a wrong appointment date"),
)

# What the scripted mock drafting model writes (--provider mock).
MOCK_DRAFTS = {
    "D-4001": (
        "You were in the hospital for community-acquired pneumonia. "
        "Start amoxicillin-clavulanate 875 mg twice daily for 5 more days. "
        "Continue metformin 1000 mg twice daily as before. "
        "See your primary care doctor within 7 days. "
        "You have an appointment at the pulmonology clinic on 2026-10-09. "
        "The results of your blood cultures are still pending; the team will call you with them. "
        "Come back to the emergency department if you notice worsening shortness of breath."
    ),
    "D-4002": (
        "You were in the hospital for acute decompensated heart failure. "
        "Your furosemide dose is now 40 mg twice daily, up from 40 mg once daily. "
        "Your metoprolol succinate dose was increased to 100 mg once daily. "
        "Do not take lisinopril until your follow-up visit. "
        "Weigh yourself every morning and write it down. "
        "You have an appointment at the cardiology clinic on 2026-10-01. "
        "See your primary care doctor within 7 days."
    ),
}

# What the simulated clinician decides for each review (True approves the verified-only text).
CLINICIAN_SCRIPT = {"D-4002": True}


# ---------------------------------------------------------------------------
# mock evidence model: does the chart support the claim? (never sees a label)
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
# Sentence scaffolding that carries no chart fact ("Continue ... as before", "Come back to ...").
_SCAFFOLDING = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "you your also appointment away back before call come continue department do doctor dose emergency "
    "follow-up get gets help hospital if increased lowered more next not notice now pending restart results "
    "right seek start still stop take taking team tell them treated until up visit when".split()
)
# A few plain-language paraphrases the mock model understands (a real model knows many more).
_PARAPHRASES = {"family": "primary", "breathing": "breath", "bluish": "blue", "peeing": "urine",
                "sugar": "blood", "scale": "weight", "pen": "marked", "red": "redness", "pause": "held"}


def _noise(case_id: str, text: str) -> float:
    """Deterministic standard-normal noise keyed on the case id and the claim."""
    return random.Random(f"{case_id}:{text}").gauss(0.0, 1.0)


def _chart_values(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in _chart_values(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _chart_values(item)]
    return [str(value)] if value is not None else []


def claim_support_scorer(context: Mapping[str, Any], instructions: Any,
                         options: Mapping[str, Any]) -> dict[str, float]:
    """P(the chart supports the claim), from fact overlap with the chart, with realistic blind spots.

    ``context`` is ``{"chart": ...}`` (the engine removes the draft before
    scoring support) and ``instructions`` ends with ``Claim: <text>``.
    Calibration labels never reach this function.
    """
    chart = context["chart"]
    claim = str(instructions).split("Claim:", 1)[1].strip()
    lowered = claim.lower()
    chart_tokens = set(_TOKEN.findall(" ".join(_chart_values(chart)).lower()))
    stems = {t[:5] for t in chart_tokens if len(t) >= 5}
    tokens = [t for t in _TOKEN.findall(lowered) if t not in _SCAFFOLDING]
    numbers = [t for t in tokens if any(ch.isdigit() for ch in t)]
    facts = [_PARAPHRASES.get(t, t) for t in tokens if t not in numbers and len(t) > 2]

    def known(token: str) -> bool:
        return token in chart_tokens or (len(token) >= 5 and token[:5] in stems)

    fact_overlap = sum(known(t) for t in facts) / len(facts) if facts else 0.5
    logit = -2.5 + 5.0 * fact_overlap
    if numbers:
        missing = sum(t not in chart_tokens for t in numbers)
        logit += 1.5 if missing == 0 else -3.0     # a dose, date or interval the chart does not contain
    else:
        logit += 0.8
    for med in chart.get("medications") or []:
        if med["name"] not in lowered:
            continue
        # The model notices a contradicted medication status only some of the time.
        catch = random.Random(f"{chart['encounter_id']}:{claim}:status").random()
        if "stop" in lowered and med["status"] != "stopped" and catch < 0.8:
            logit -= 3.0
        if "continue" in lowered and med["status"] in ("held", "stopped") and catch < 0.75:
            logit -= 3.0
        if ("increased" in lowered or "lowered" in lowered) and med["status"] != "changed":
            logit -= 1.5
    logit += 0.7 * _noise(chart["encounter_id"], claim)
    p_true = 1.0 / (1.0 + math.exp(-logit))
    return {"true": p_true, "false": 1.0 - p_true}


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


class SummaryState(TypedDict, total=False):
    encounter_id: str
    chart: dict
    draft: str
    verification: dict
    review: dict
    published: str
    status: str


def build_graph(model: Any, client: LocalCLIClient, portal: list[dict[str, Any]], checkpointer: Any = None) -> Any:
    """draft_summary -> verify_claims -> publish | clinician_review -> publish | END."""

    def draft_summary(state: SummaryState) -> dict[str, Any]:
        chart = json.dumps(state["chart"], indent=2, sort_keys=True)
        reply = model.invoke([SystemMessage(DRAFT_PROMPT), HumanMessage(f"Discharge chart:\n{chart}")])
        content = reply.content
        if not isinstance(content, str):   # some providers return a list of content blocks
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        return {"draft": content.strip()}

    def verify_claims(state: SummaryState) -> dict[str, Any]:
        # Same builder as the calibration examples: {"chart": ..., "answer": <draft>}.
        context = charts_data.summary_context(state["chart"], state["draft"])
        answer = client.evaluate(context, {"claims": CLAIM_CHECK}).answers["claims"]
        return {"verification": {
            "retained_claims": list(answer.retained_claims),
            "dropped_claims": [{"text": d.text, "score": d.score, "reason": d.reason} for d in answer.dropped_claims],
            "threshold": answer.raw.get("threshold"),
            "guarantee": answer.guarantee.describe(),
            "heuristic": answer.is_heuristic,
            "verified_text": answer.as_text(),
        }}

    def route_after_verification(state: SummaryState) -> str:
        v = state["verification"]
        # Publish unreviewed only when every claim was verified; anything else goes to a clinician.
        verified = v["retained_claims"] and not v["dropped_claims"] and not v["heuristic"]
        return "publish" if verified else "clinician_review"

    def clinician_review(state: SummaryState) -> Command[Literal["publish", "__end__"]]:
        # Nothing above interrupt() has side effects: this node re-runs from the top on resume.
        v = state["verification"]
        decision = interrupt({
            "kind": "clinician_review",
            "encounter_id": state["encounter_id"],
            "retained_claims": v["retained_claims"],
            "dropped_claims": v["dropped_claims"],
            "verified_text": v["verified_text"],
            "guarantee": v["guarantee"],
            "heuristic": v["heuristic"],
            "resume_with": {"action": "approve | reject", "note": "optional clinician note"},
        })
        if not isinstance(decision, Mapping):
            decision = {"action": decision}
        review = {"action": decision.get("action"), "note": decision.get("note")}
        if decision.get("action") == "approve" and v["verified_text"]:
            return Command(goto="publish", update={"review": review})
        # Reject, an unrecognized answer, or nothing verified to publish: fail closed.
        return Command(goto=END, update={"review": review,
                                         "status": "returned to the clinician; nothing was published"})

    def publish(state: SummaryState) -> dict[str, Any]:
        text = state["verification"]["verified_text"]      # only retained claims are ever published
        portal.append({"encounter_id": state["encounter_id"], "text": text})
        return {"published": text, "status": "published to the patient portal"}

    builder = StateGraph(SummaryState)
    builder.add_node("draft_summary", draft_summary)
    builder.add_node("verify_claims", verify_claims)
    builder.add_node("clinician_review", clinician_review, destinations=("publish", END))
    builder.add_node("publish", publish)
    builder.add_edge(START, "draft_summary")
    builder.add_edge("draft_summary", "verify_claims")
    builder.add_conditional_edges("verify_claims", route_after_verification,
                                  {"publish": "publish", "clinician_review": "clinician_review"})
    builder.add_edge("publish", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def drafting_model(provider: str, encounter_id: str) -> Any:
    """The chat model that writes the draft."""
    if provider != "mock":
        from _shared.langchain_models import chat_model

        return chat_model(provider)
    # LangChain's fake chat model, scripted per encounter (a fresh one per thread).
    return GenericFakeChatModel(messages=iter([AIMessage(content=MOCK_DRAFTS[encounter_id])]),
                                disable_streaming=True)


# ---------------------------------------------------------------------------
# running the scenarios
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph discharge summaries with calibrated claim checks.")
    providers.add_arguments(parser)
    return parser.parse_args(argv)


def build_client(args: argparse.Namespace) -> LocalCLIClient:
    evidence_provider = args.evidence_provider or args.provider
    evidence = providers.evidence_backend(evidence_provider, mock_scorer=claim_support_scorer)
    return LocalCLIClient(evidence, store=calibration.default_store(__file__, args.store),
                          sample_count=providers.sample_count(evidence_provider))


def review_request(review: Mapping[str, Any]) -> dict[str, Any]:
    """The review card in the shape the shared reviewer prints."""
    kept, dropped = len(review["retained_claims"]), len(review["dropped_claims"])
    if review.get("heuristic"):
        reason = "no calibrated guarantee is available, so no claim counts as verified; nothing can be published."
    elif not kept:
        reason = "no claim in the draft was verified; nothing can be published."
    else:
        reason = (f"{dropped} of {kept + dropped} claims are at or below the calibrated support threshold; "
                  f"approving publishes only the {kept} retained claims.")
    return {
        "tool": "publish_discharge_summary",
        "arguments": {"encounter_id": review["encounter_id"], "claims_retained": kept, "claims_dropped": dropped},
        "reason": reason,
        "guarantee": {"statement": review["guarantee"]},
    }


def process_encounter(graph: Any, encounter_id: str, reviewer: Reviewer) -> dict[str, Any]:
    config = {"configurable": {"thread_id": f"encounter-{encounter_id}"}}
    out = graph.invoke({"encounter_id": encounter_id, "chart": CHARTS[encounter_id]}, config)
    state = graph.get_state(config).values
    v = state["verification"]
    n_claims = len(v["retained_claims"]) + len(v["dropped_claims"])
    if v["threshold"] is None:
        print(f"    draft: {n_claims} claims; no calibrated support threshold (no guarantee available)")
    else:
        print(f"    draft: {n_claims} claims; calibrated support threshold {v['threshold']:.3f}")
    for text in v["retained_claims"]:
        print(f"      retained  {text}")
    for item in v["dropped_claims"]:
        print(f"      DROPPED   {item['text']}  (support score {item['score']:.3f})")
    print(f"    guarantee: {v['guarantee']}")
    decision = None
    interrupts = out.get("__interrupt__") or ()
    if interrupts:
        review = interrupts[0].value
        print(f"    graph paused at {graph.get_state(config).next[0]} (thread {config['configurable']['thread_id']})")
        approved = reviewer.decide(encounter_id, review_request(review))
        decision = ({"action": "approve", "note": "Publish the verified claims; dropped claims stay out."}
                    if approved else {"action": "reject", "note": "Rewrite from the chart before release."})
        print(f"    resume value: {json.dumps(decision, sort_keys=True)}")
        graph.invoke(Command(resume=decision), config)
        state = graph.get_state(config).values
        print(f'    review recorded in the checkpoint (state["review"]): {json.dumps(state["review"], sort_keys=True)}')
    return {"case_id": encounter_id, "verification": v, "review": decision,
            "published": state.get("published"), "status": state.get("status")}


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    client = build_client(args)
    if args.provider != "mock":
        providers.settings(args.provider).require_key()
    examples = calibration.load_jsonl(DATASET)
    [profile] = calibration.ensure_calibrated(client, [CLAIM_CHECK], examples, recalibrate=args.recalibrate)
    print(f"calibration profile {calibration.describe_profile(profile)}")
    print(f"evidence model: {client.backend.name} (access level {client.backend.access_level})")

    reviewer = Reviewer(interactive=args.interactive, script=CLINICIAN_SCRIPT, role="clinician")
    portal: list[dict[str, Any]] = []
    results = []
    for encounter_id, description in SCENARIOS:
        display.scenario(encounter_id, description)
        graph = build_graph(drafting_model(args.provider, encounter_id), client, portal)
        result = process_encounter(graph, encounter_id, reviewer)
        print(f"    outcome: {result['status']}")
        if result["published"]:
            print(textwrap.fill(result["published"], 78, initial_indent="    portal text: ",
                                subsequent_indent="      "))
        results.append(result)
    return results


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    display.banner("LangGraph: discharge summaries with a calibrated claim-verification node", "healthcare")
    run(args)
    display.footer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
