"""Baseline: a LangGraph agent using Claude directly, no CCI involved.

Asks the model for an urgency label, a self-reported confidence, and a
patient-facing explanation, then auto-sends whenever that self-reported
number clears a hand-picked threshold. Nothing checks the explanation
against the patient's actual intake facts.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.models import chat_model

AUTO_SEND_CONFIDENCE_THRESHOLD = 0.8

SYSTEM_PROMPT = (
    "You are a patient-portal triage assistant. Given a patient's message and their "
    "intake facts, decide the urgency level and draft a short reply. "
    "SYNTHETIC DEMO DATA ONLY."
)

LEVELS = ("self_care", "primary_care", "urgent_care", "emergency")


class TriageAssessment(BaseModel):
    urgency: str = Field(description=f"one of: {', '.join(LEVELS)}")
    confidence: float = Field(description="your own confidence, 0 to 1, that this urgency label is correct")
    explanation: str = Field(description="a short, patient-facing explanation and recommendation")


class BaselineState(TypedDict):
    message: str
    facts: dict[str, Any]
    assessment: dict[str, Any]
    auto_sent: bool


def build_graph():
    model = chat_model().with_structured_output(TriageAssessment)

    def triage(state: BaselineState) -> dict[str, Any]:
        prompt = (
            f"Patient message: {state['message']}\n\n"
            f"Intake facts (structured): {json.dumps(state['facts'])}"
        )
        result = model.invoke([("system", SYSTEM_PROMPT), ("human", prompt)])
        return {"assessment": result.model_dump()}

    def decide(state: BaselineState) -> dict[str, Any]:
        confidence = state["assessment"]["confidence"]
        return {"auto_sent": confidence >= AUTO_SEND_CONFIDENCE_THRESHOLD}

    builder = StateGraph(BaselineState)
    builder.add_node("triage", triage)
    builder.add_node("decide", decide)
    builder.add_edge(START, "triage")
    builder.add_edge("triage", "decide")
    builder.add_edge("decide", END)
    return builder.compile()
