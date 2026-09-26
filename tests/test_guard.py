"""The framework-agnostic ``ToolGuard``: decision mapping, caching, and fail-closed errors."""

from __future__ import annotations

import pytest

from cli_sdk import Belief, Gate, Interval, Set
from cli_sdk.answers import parse_answer
from cli_sdk.answers.response import EvaluateResponse
from cli_sdk.integrations import ALLOW, BLOCK, ESCALATE, GuardRule, ToolGuard, decide, most_severe

GATE = Gate(instructions="Approve?", calibration_profile="g", guarantee="risk", target=0.05)
BELIEF = Belief(instructions="Safe?", calibration_profile="b")
TIERS = Set(instructions="Tier?", options={"A": None, "B": None, "C": None}, calibration_profile="s")
SEVERITY = Interval(instructions="Grade?", levels=["1", "2", "3", "4", "5"], calibration_profile="i")
CARD = {"type": "risk", "method": "CRC", "target": 0.05, "calibration_profile": "g", "calibration_n": 300}


def answer(payload):
    return parse_answer(payload)


def test_gate_mapping():
    rule = GuardRule(tool="refund", query=GATE)
    approve = answer({"type": "gate", "decision": "auto_approve", "confidence": 0.99, "threshold": 0.95, "guarantee": CARD})
    assert decide(rule, approve, "refund", {}).action == ALLOW
    assert decide(rule, answer({"type": "gate", "decision": "escalate", "guarantee": CARD}), "refund", {}).action == ESCALATE
    assert decide(rule, answer({"type": "gate", "decision": "abstain", "guarantee": CARD}), "refund", {}).action == BLOCK
    # an unknown decision is coerced to escalate by the answer parser: never allow
    assert decide(rule, answer({"type": "gate", "decision": "yolo", "guarantee": CARD}), "refund", {}).action == ESCALATE


def test_heuristic_answers_never_allow():
    rule = GuardRule(tool="refund", query=GATE)
    heuristic = answer({"type": "gate", "decision": "auto_approve", "guarantee": {"type": "heuristic"}})
    assert decide(rule, heuristic, "refund", {}).action == ESCALATE
    strict = GuardRule(tool="refund", query=GATE, on_heuristic=BLOCK)
    assert decide(strict, heuristic, "refund", {}).action == BLOCK


def test_belief_straddle_rule():
    card = {"type": "calibration", "method": "IVAP"}
    rule = GuardRule(tool="order", query=BELIEF, allow_above=0.8, block_below=0.3)
    assert decide(rule, answer({"type": "belief", "venn_abers": [0.85, 0.9], "guarantee": card}), "order", {}).action == ALLOW
    assert decide(rule, answer({"type": "belief", "venn_abers": [0.1, 0.2], "guarantee": card}), "order", {}).action == BLOCK
    assert decide(rule, answer({"type": "belief", "venn_abers": [0.5, 0.9], "guarantee": card}), "order", {}).action == ESCALATE
    never_block = GuardRule(tool="order", query=BELIEF, allow_above=0.8)
    assert decide(never_block, answer({"type": "belief", "venn_abers": [0.1, 0.2], "guarantee": card}), "order", {}).action == ESCALATE


def test_set_rule():
    card = {"type": "coverage", "method": "APS", "alpha": 0.1}
    rule = GuardRule(tool="tier", query=TIERS, allow_labels=["A", "B"], block_labels=["C"])
    assert decide(rule, answer({"type": "set", "set": ["A"], "guarantee": card}), "tier", {}).action == ALLOW
    assert decide(rule, answer({"type": "set", "set": ["A", "B"], "guarantee": card}), "tier", {}).action == ESCALATE
    assert decide(rule, answer({"type": "set", "set": ["C"], "guarantee": card}), "tier", {}).action == BLOCK
    assert decide(rule, answer({"type": "set", "set": [], "guarantee": card}), "tier", {}).action == ESCALATE
    with pytest.raises(ValueError):
        GuardRule(tool="tier", query=TIERS)


def test_set_rule_with_match_argument():
    card = {"type": "coverage", "method": "APS", "alpha": 0.1}
    rule = GuardRule(tool="tier", query=TIERS, allow_labels=["A", "B"], match_argument="tier")
    singleton_a = answer({"type": "set", "set": ["A"], "guarantee": card})
    assert decide(rule, singleton_a, "tier", {"tier": "A"}).action == ALLOW
    mismatch = decide(rule, singleton_a, "tier", {"tier": "B"})
    assert mismatch.action == ESCALATE and "proposed" in mismatch.reason


def test_interval_rule():
    card = {"type": "coverage", "method": "ordinal-aps", "alpha": 0.1}
    legend = {str(i): lvl for i, lvl in enumerate(["1", "2", "3", "4", "5"])}
    rule = GuardRule(tool="expedite", query=SEVERITY, allow_levels=["3", "4", "5"], block_levels=["1", "2"])

    def interval(lo, hi):
        return answer({"type": "interval", "interval": [lo, hi], "legend": legend, "guarantee": card})

    assert decide(rule, interval(2, 4), "expedite", {}).action == ALLOW
    assert decide(rule, interval(0, 1), "expedite", {}).action == BLOCK
    assert decide(rule, interval(1, 3), "expedite", {}).action == ESCALATE


class RecordingClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def evaluate(self, context, queries, **kwargs):
        self.calls.append(context)
        if self.error:
            raise self.error
        return EvaluateResponse.from_payload({"answers": {"guard": self.payload}})


def test_toolguard_caches_and_ignores_unguarded_tools():
    client = RecordingClient({"type": "gate", "decision": "auto_approve", "guarantee": CARD})
    guard = ToolGuard(client, [GuardRule(tool="refund", query=GATE, context=lambda a: {"amount": a["amount"]})])
    assert guard.check("lookup", {"id": 1}).allowed
    assert guard.check("refund", {"amount": 10}).allowed
    assert guard.check("refund", {"amount": 10}).allowed
    assert client.calls == [{"amount": 10}]
    guard.check("refund", {"amount": 11})
    assert len(client.calls) == 2


def test_context_builder_may_read_state():
    client = RecordingClient({"type": "gate", "decision": "escalate", "guarantee": CARD})
    rule = GuardRule(tool="refund", query=GATE, context=lambda args, state: {"amount": args["amount"], "tier": state["tier"]})
    decision = ToolGuard(client, [rule]).check("refund", {"amount": 5}, {"tier": "gold"})
    assert client.calls == [{"amount": 5, "tier": "gold"}]
    assert decision.needs_review
    summary = decision.to_dict()
    assert summary["guarantee"]["type"] == "risk" and summary["action"] == ESCALATE


def test_errors_fail_closed_or_raise():
    client = RecordingClient(error=RuntimeError("provider down"))
    guard = ToolGuard(client, [GuardRule(tool="refund", query=GATE)])
    decision = guard.check("refund", {"amount": 1})
    assert decision.action == ESCALATE and "provider down" in decision.reason
    raising = ToolGuard(client, [GuardRule(tool="refund", query=GATE)], on_error="raise")
    with pytest.raises(RuntimeError):
        raising.check("refund", {"amount": 1})
    bad_context = ToolGuard(client, [GuardRule(tool="refund", query=GATE, context=lambda a: a["missing"])])
    assert bad_context.check("refund", {}).action == ESCALATE


def test_most_severe():
    client = RecordingClient({"type": "gate", "decision": "auto_approve", "guarantee": CARD})
    guard = ToolGuard(client, [GuardRule(tool="refund", query=GATE)])
    allow = guard.check("refund", {"a": 1})
    other = guard.check("lookup", {})
    assert most_severe([allow, other]) == ALLOW
    assert most_severe([]) == ALLOW


@pytest.mark.asyncio
async def test_acheck_runs_off_the_event_loop():
    client = RecordingClient({"type": "gate", "decision": "auto_approve", "guarantee": CARD})
    guard = ToolGuard(client, [GuardRule(tool="refund", query=GATE)])
    decision = await guard.acheck("refund", {"amount": 3})
    assert decision.allowed


def test_gate_with_no_feasible_threshold_says_so():
    # Local mode returns threshold=None when no threshold meets the target on the
    # calibration set; the reason must not imply that a threshold was missed.
    rule = GuardRule(tool="refund", query=GATE)
    none = answer({"type": "gate", "decision": "escalate", "confidence": 0.91, "threshold": None, "guarantee": CARD})
    decision = decide(rule, none, "refund", {})
    assert decision.action == ESCALATE
    assert "no auto-approval threshold meets the target" in decision.reason and "0.910" in decision.reason
    missed = answer({"type": "gate", "decision": "escalate", "confidence": 0.5, "threshold": 0.8, "guarantee": CARD})
    assert "below the calibrated auto-approval threshold" in decide(rule, missed, "refund", {}).reason
    hosted = answer({"type": "gate", "decision": "escalate", "guarantee": CARD})  # no confidence or threshold
    assert decide(rule, hosted, "refund", {}).reason == "below the calibrated auto-approval threshold."
