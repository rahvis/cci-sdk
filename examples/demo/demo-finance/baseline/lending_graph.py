"""Baseline: LangGraph + Claude only, no CCI, for credit tiering."""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.models import chat_model

AUTO_APPROVE_CONFIDENCE_THRESHOLD = 0.8
TIERS = ("prime", "near_prime", "subprime", "decline")

SYSTEM_PROMPT = (
    "You are a credit underwriting assistant. Given an applicant's channel and financials, "
    "assign a credit tier and decide whether to auto-approve at that tier. SYNTHETIC DEMO DATA ONLY."
)


class TierAssessment(BaseModel):
    tier: str = Field(description=f"one of: {', '.join(TIERS)}")
    confidence: float = Field(description="your own confidence, 0 to 1, that this tier is correct")
    rationale: str = Field(description="one short sentence on why")


class LendingState(TypedDict):
    message: str
    facts: dict[str, Any]
    assessment: dict[str, Any]
    auto_approved: bool


def build_graph():
    model = chat_model().with_structured_output(TierAssessment)

    def assess(state: LendingState) -> dict[str, Any]:
        prompt = f"{state['message']}\n\nStructured facts: {json.dumps(state['facts'])}"
        result = model.invoke([("system", SYSTEM_PROMPT), ("human", prompt)])
        return {"assessment": result.model_dump()}

    def decide(state: LendingState) -> dict[str, Any]:
        return {"auto_approved": state["assessment"]["confidence"] >= AUTO_APPROVE_CONFIDENCE_THRESHOLD}

    builder = StateGraph(LendingState)
    builder.add_node("assess", assess)
    builder.add_node("decide", decide)
    builder.add_edge(START, "assess")
    builder.add_edge("assess", "decide")
    builder.add_edge("decide", END)
    return builder.compile()
