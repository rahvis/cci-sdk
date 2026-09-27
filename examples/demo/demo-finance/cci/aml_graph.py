"""CCI-guarded AML pipeline: a single calibrated Gate decides whether to
auto-hold the account, with a real FDR bound on the auto-held population,
computed via Learn-then-Test (a different algorithm than the CRC/risk
Gate the lending scenario and the healthcare demo both used)."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from cci.queries import aml_hold_query

BACKEND = {"provider": "anthropic", "model": "claude-sonnet-5"}


class AMLCCIState(TypedDict, total=False):
    message: str
    facts: dict[str, Any]
    gate_decision: str
    gate_guarantee: str
    auto_held: bool


def build_graph(client):
    def gate_check(state: AMLCCIState) -> dict[str, Any]:
        result = client.evaluate(
            context={"message": state["message"], "facts": state["facts"]},
            backend=BACKEND,
            queries={"hold": aml_hold_query()},
        )
        answer = result.answers["hold"]
        return {
            "gate_decision": answer.decision,
            "gate_guarantee": answer.guarantee.describe(),
            "auto_held": answer.decision == "auto_approve",
        }

    builder = StateGraph(AMLCCIState)
    builder.add_node("gate_check", gate_check)
    builder.add_edge(START, "gate_check")
    builder.add_edge("gate_check", END)
    return builder.compile()
