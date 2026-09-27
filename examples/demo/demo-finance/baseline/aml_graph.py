"""Baseline: LangGraph + Claude only, no CCI, for AML account holds."""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.models import chat_model

AUTO_ACTION_CONFIDENCE_THRESHOLD = 0.8

SYSTEM_PROMPT = (
    "You are an AML compliance assistant. Given a transaction alert, decide whether the account "
    "should be held for investigation or cleared. SYNTHETIC DEMO DATA ONLY."
)


class HoldAssessment(BaseModel):
    should_hold: bool = Field(description="true if the account should be held, false if it should be cleared")
    confidence: float = Field(description="your own confidence, 0 to 1, that this decision is correct")
    rationale: str = Field(description="one short sentence on why")


class AMLState(TypedDict):
    message: str
    facts: dict[str, Any]
    assessment: dict[str, Any]
    auto_actioned: bool


def build_graph():
    model = chat_model().with_structured_output(HoldAssessment)

    def assess(state: AMLState) -> dict[str, Any]:
        prompt = f"{state['message']}\n\nStructured facts: {json.dumps(state['facts'])}"
        result = model.invoke([("system", SYSTEM_PROMPT), ("human", prompt)])
        return {"assessment": result.model_dump()}

    def decide(state: AMLState) -> dict[str, Any]:
        return {"auto_actioned": state["assessment"]["confidence"] >= AUTO_ACTION_CONFIDENCE_THRESHOLD}

    builder = StateGraph(AMLState)
    builder.add_node("assess", assess)
    builder.add_node("decide", decide)
    builder.add_edge(START, "assess")
    builder.add_edge("assess", "decide")
    builder.add_edge("decide", END)
    return builder.compile()
