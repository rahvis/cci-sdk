"""CCI-guarded lending pipeline.

    START -> assess_tier -> [not singleton] -> escalate -> END
                          -> gate_check -> [not approved] -> escalate -> END
                                        -> auto_approve -> END

Set gives a calibrated candidate tier, audited **per channel** (Mondrian
grouping) so a channel with weaker data can't hide behind a strong
blended coverage number. Gate then checks the specific proposal against a
real, calibrated risk bound before anything is auto-approved.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from cci.queries import lending_approve_query, lending_tier_query

BACKEND = {"provider": "anthropic", "model": "claude-sonnet-5"}


class LendingCCIState(TypedDict, total=False):
    message: str
    facts: dict[str, Any]
    channel: str
    set_top: str
    set_full: list[str]
    set_is_singleton: bool
    set_guarantee: str
    set_group: str | None
    gate_decision: str | None
    gate_guarantee: str | None
    auto_approved: bool
    escalation_reason: str


def build_graph(client):
    def assess_tier(state: LendingCCIState) -> dict[str, Any]:
        result = client.evaluate(
            context={"message": state["message"], "facts": state["facts"], "channel": state["channel"]},
            backend=BACKEND,
            queries={"tier": lending_tier_query()},
        )
        answer = result.answers["tier"]
        return {
            "set_top": answer.top or "",
            "set_full": list(answer.set),
            "set_is_singleton": answer.is_singleton,
            "set_guarantee": answer.guarantee.describe(),
        }

    def route_after_tier(state: LendingCCIState) -> Literal["gate_check", "escalate"]:
        return "gate_check" if state["set_is_singleton"] else "escalate"

    def gate_check(state: LendingCCIState) -> dict[str, Any]:
        result = client.evaluate(
            context={"message": state["message"], "facts": state["facts"], "proposed_tier": state["set_top"]},
            backend=BACKEND,
            queries={"approve": lending_approve_query()},
        )
        answer = result.answers["approve"]
        return {"gate_decision": answer.decision, "gate_guarantee": answer.guarantee.describe()}

    def route_after_gate(state: LendingCCIState) -> Literal["auto_approve", "escalate"]:
        return "auto_approve" if state["gate_decision"] == "auto_approve" else "escalate"

    def auto_approve(state: LendingCCIState) -> dict[str, Any]:
        return {"auto_approved": True, "escalation_reason": ""}

    def escalate(state: LendingCCIState) -> dict[str, Any]:
        if not state["set_is_singleton"]:
            reason = f"candidate tier set is not a single label ({state['set_full']}): {state['set_guarantee']}"
        else:
            reason = f"gate declined to auto-approve: {state.get('gate_guarantee', '')}"
        return {"auto_approved": False, "escalation_reason": reason}

    builder = StateGraph(LendingCCIState)
    builder.add_node("assess_tier", assess_tier)
    builder.add_node("gate_check", gate_check)
    builder.add_node("auto_approve", auto_approve)
    builder.add_node("escalate", escalate)
    builder.add_edge(START, "assess_tier")
    builder.add_conditional_edges("assess_tier", route_after_tier, {"gate_check": "gate_check", "escalate": "escalate"})
    builder.add_conditional_edges("gate_check", route_after_gate, {"auto_approve": "auto_approve", "escalate": "escalate"})
    builder.add_edge("auto_approve", END)
    builder.add_edge("escalate", END)
    return builder.compile()
