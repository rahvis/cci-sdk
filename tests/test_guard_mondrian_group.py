"""Regression: a guard decision on a group-conditional (Mondrian) Set names its group.

The guarantee card of a Mondrian answer reports ``calibration_n`` for the
caller's group only. ``GuardDecision.to_dict()`` is what reviewer queues,
approval requests and audit records carry, so it must say which group that
count belongs to; without it, "n=40" reads as the size of the whole profile.
"""

from __future__ import annotations

from cli_sdk import Set
from cli_sdk.evidence import MockEvidenceBackend
from cli_sdk.integrations import GuardRule, ToolGuard
from cli_sdk.local import LocalCLIClient

OPTIONS = {"low": None, "high": None}


def _scorer(context, instructions, options):
    p = 0.9 if context["signal"] == "low" else 0.1
    return {"low": p, "high": 1.0 - p}


def _examples():
    rows = []
    for i in range(60):
        channel = "branch" if i < 45 else "broker"
        signal = "low" if i % 3 else "high"
        rows.append({"context": {"id": i, "channel": channel, "signal": signal}, "label": signal})
    return rows


def test_group_is_named_in_the_decision_summary(tmp_path):
    query = Set(instructions="Risk level?", options=OPTIONS, calibration_profile="mondrian-demo",
                alpha=0.2, method="LAC", group_by="channel")
    client = LocalCLIClient(MockEvidenceBackend(scorer=_scorer), store=tmp_path)
    client.calibrate(query, _examples())
    guard = ToolGuard(client, [GuardRule(tool="set_level", query=query, allow_labels=["low"])])

    decision = guard.check("set_level", {"id": 99, "channel": "broker", "signal": "low"})
    summary = decision.to_dict()["guarantee"]
    assert summary["group"] == "broker"
    assert summary["calibration_n"] == 15


def test_marginal_answers_have_no_group(tmp_path):
    query = Set(instructions="Risk level?", options=OPTIONS, calibration_profile="marginal-demo",
                alpha=0.2, method="LAC")
    client = LocalCLIClient(MockEvidenceBackend(scorer=_scorer), store=tmp_path)
    client.calibrate(query, _examples())
    guard = ToolGuard(client, [GuardRule(tool="set_level", query=query, allow_labels=["low"])])

    summary = guard.check("set_level", {"id": 99, "channel": "broker", "signal": "low"}).to_dict()["guarantee"]
    assert "group" not in summary and summary["calibration_n"] == 60
