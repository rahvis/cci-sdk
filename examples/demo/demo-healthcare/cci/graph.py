"""CCI-guarded pipeline: the same triage task, with a calibrated guard in front
of what gets auto-sent.

    START -> assess_urgency -> draft -> [ambiguous or high-acuity] -> escalate -> END
                                      -> gate_check -> [gate escalates] -> escalate -> END
                                                     -> verify_claims -> [claim dropped] -> escalate -> END
                                                                       -> auto_send -> END

Set gives a calibrated candidate set (coverage guarantee); auto-send is
only even considered when that set is a single, low-acuity label. Gate
then checks the specific proposal against a real risk bound. Claim checks
the drafted reply against the patient's own intake facts before it can
ever be sent. Any of the three saying no sends the case to a human nurse
review queue instead of guessing.
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from cci.queries import LOW_ACUITY, URGENCY_LEVELS, claim_query, urgency_gate_query, urgency_set_query
from common.models import chat_model

BACKEND = {"provider": "anthropic", "model": "claude-sonnet-5"}

DRAFT_SYSTEM_PROMPT = (
    "You are a patient-portal assistant drafting a short reply for a patient, given the "
    "urgency level a clinical triage step has already assigned. Write only what the intake "
    "facts support; do not invent details not present in the facts. SYNTHETIC DEMO DATA ONLY."
)


class CCIState(TypedDict, total=False):
    message: str
    facts: dict[str, Any]
    set_top: str
    set_full: list[str]
    set_is_singleton: bool
    set_guarantee: str
    gate_decision: str | None
    gate_guarantee: str | None
    explanation: str
    retained_claims: list[str]
    dropped_claims: list[dict[str, Any]]
    claim_guarantee: str | None
    claim_heuristic: bool
    auto_sent: bool
    escalation_reason: str
    final_text: str


def build_graph(client):
    model = chat_model()

    def assess_urgency(state: CCIState) -> dict[str, Any]:
        result = client.evaluate(
            context={"message": state["message"], "facts": state["facts"]},
            backend=BACKEND,
            queries={"urgency": urgency_set_query()},
        )
        answer = result.answers["urgency"]
        return {
            "set_top": answer.top or "",
            "set_full": list(answer.set),
            "set_is_singleton": answer.is_singleton,
            "set_guarantee": answer.guarantee.describe(),
        }

    def draft(state: CCIState) -> dict[str, Any]:
        prompt = (
            f"Patient message: {state['message']}\n\n"
            f"Intake facts: {json.dumps(state['facts'])}\n\n"
            f"Assigned urgency level: {state['set_top']}\n\n"
            "Write a short reply (2-4 sentences) explaining the urgency level and what the "
            "patient should do."
        )
        reply = model.invoke([("system", DRAFT_SYSTEM_PROMPT), ("human", prompt)])
        content = reply.content if isinstance(reply.content, str) else str(reply.content)
        return {"explanation": content.strip()}

    def route_after_draft(state: CCIState) -> Literal["gate_check", "escalate"]:
        if state["set_is_singleton"] and state["set_top"] in LOW_ACUITY:
            return "gate_check"
        return "escalate"

    def gate_check(state: CCIState) -> dict[str, Any]:
        result = client.evaluate(
            context={"message": state["message"], "facts": state["facts"], "proposed_urgency": state["set_top"]},
            backend=BACKEND,
            queries={"route": urgency_gate_query()},
        )
        answer = result.answers["route"]
        return {"gate_decision": answer.decision, "gate_guarantee": answer.guarantee.describe()}

    def route_after_gate(state: CCIState) -> Literal["verify_claims", "escalate"]:
        return "verify_claims" if state["gate_decision"] == "auto_approve" else "escalate"

    def verify_claims(state: CCIState) -> dict[str, Any]:
        result = client.evaluate(
            context={
                "facts": state["facts"],
                "guidance": URGENCY_LEVELS.get(state["set_top"], ""),
                "answer": state["explanation"],
            },
            backend=BACKEND,
            queries={"claims": claim_query()},
        )
        answer = result.answers["claims"]
        return {
            "retained_claims": list(answer.retained_claims),
            "dropped_claims": [
                {"text": d.text, "reason": d.reason, "score": d.score} for d in answer.dropped_claims
            ],
            "claim_guarantee": answer.guarantee.describe(),
            "claim_heuristic": answer.is_heuristic,
        }

    def route_after_claims(state: CCIState) -> Literal["auto_send", "escalate"]:
        if state["claim_heuristic"] or state["dropped_claims"] or not state["retained_claims"]:
            return "escalate"
        return "auto_send"

    def auto_send(state: CCIState) -> dict[str, Any]:
        return {"auto_sent": True, "final_text": " ".join(state["retained_claims"]),
                "escalation_reason": ""}

    def escalate(state: CCIState) -> dict[str, Any]:
        if not state["set_is_singleton"]:
            reason = f"urgency candidate set is not a single label: guarantee {state['set_guarantee']}"
        elif state["set_top"] not in LOW_ACUITY:
            reason = f"proposed urgency '{state['set_top']}' is not low-acuity: always routed to a human"
        elif state.get("gate_decision") not in (None, "auto_approve"):
            reason = f"gate declined to auto-approve: {state.get('gate_guarantee', '')}"
        elif state.get("claim_heuristic"):
            reason = "claim check has no guarantee yet (profile still collecting): fails closed"
        else:
            reason = f"reply included unsupported claim(s): {state.get('dropped_claims')}"
        return {"auto_sent": False, "final_text": "", "escalation_reason": reason}

    builder = StateGraph(CCIState)
    builder.add_node("assess_urgency", assess_urgency)
    builder.add_node("draft", draft)
    builder.add_node("gate_check", gate_check)
    builder.add_node("verify_claims", verify_claims)
    builder.add_node("auto_send", auto_send)
    builder.add_node("escalate", escalate)

    builder.add_edge(START, "assess_urgency")
    builder.add_edge("assess_urgency", "draft")
    builder.add_conditional_edges("draft", route_after_draft, {"gate_check": "gate_check", "escalate": "escalate"})
    builder.add_conditional_edges("gate_check", route_after_gate, {"verify_claims": "verify_claims", "escalate": "escalate"})
    builder.add_conditional_edges("verify_claims", route_after_claims, {"auto_send": "auto_send", "escalate": "escalate"})
    builder.add_edge("auto_send", END)
    builder.add_edge("escalate", END)
    return builder.compile()
