"""The LangChain agent examples run end to end in mock mode.

Covers ``examples/agents/langchain/finance_refund_agent.py`` and
``examples/agents/langchain/healthcare_patient_triage.py``: the allow /
escalate outcomes, whether each tool really ran, the human approve / edit /
reject paths through ``HumanInTheLoopMiddleware``, batch mode, profile
reuse, the async path, exchangeability of the guard context with the
calibration data, and the friendly stop on a placeholder API key.

Skipped when LangChain 1.x is not installed, so the core suite runs without it.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _installed(distribution: str) -> bool:
    try:
        importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


# Check the installed distributions first: once examples/agents is on
# sys.path, its langchain/ folder would otherwise import as an empty
# namespace package and hide a missing LangChain.
if not (_installed("langchain") and _installed("langgraph")):
    pytest.skip('LangChain 1.x is not installed: pip install "cli-sdk[langchain]"', allow_module_level=True)
pytest.importorskip("langchain.agents")
pytest.importorskip("langgraph.checkpoint.memory")

from langchain.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from cli_sdk.integrations.langchain import guard_decisions, pending_reviews  # noqa: E402

AGENTS = Path(__file__).resolve().parents[1] / "examples" / "agents"


def _load(relative: str) -> ModuleType:
    path = AGENTS / relative
    name = "cli_test_" + relative.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses in the example resolve their module through sys.modules
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def refunds() -> ModuleType:
    return _load("langchain/finance_refund_agent.py")


@pytest.fixture(scope="module")
def triage() -> ModuleType:
    return _load("langchain/healthcare_patient_triage.py")


def _by_case(outcomes):
    return {o.case_id: o for o in outcomes}


def _calibrated_guard(module, query, scorer, dataset, store):
    from _shared import calibration  # importable once an example module is loaded

    from cli_sdk.evidence import MockEvidenceBackend
    from cli_sdk.local import LocalCLIClient

    client = LocalCLIClient(MockEvidenceBackend(scorer=scorer, name="mock"), store=store)
    calibration.ensure_calibrated(client, [query], calibration.load_jsonl(dataset), quiet=True)
    return module.build_guard(client)


# ---------------------------------------------------------------------------
# finance_refund_agent.py
# ---------------------------------------------------------------------------


def test_refund_scenarios_allow_edit_reject(refunds, tmp_path, capsys):
    outcomes = _by_case(refunds.run(["--store", str(tmp_path)]))
    out = capsys.readouterr().out

    small = outcomes["R-1001"]
    assert small.guard_actions == ["allow"] and small.reviews == []
    assert small.refunds_issued == [{"order_id": "ORD-90101", "amount": 14.99, "reason": "damaged_item"}]
    assert small.tool_statuses == ["success"]

    borderline = outcomes["R-1002"]
    assert borderline.guard_actions == ["escalate"] and borderline.reviews == ["edit"]
    # the reviewer's amount ran, not the agent's
    assert borderline.refunds_issued == [{"order_id": "ORD-90106", "amount": 169.99, "reason": "changed_mind"}]
    assert borderline.tool_statuses == ["success"]

    duplicate = outcomes["R-1003"]
    assert duplicate.guard_actions == ["escalate"] and duplicate.reviews == ["reject"]
    assert not duplicate.executed and duplicate.tool_statuses == ["error"]
    assert "did not issue a refund" in duplicate.final_reply

    assert "Synthetic data for demonstration only" in out
    assert "Expected rate of decisions that are auto-approved and wrong is at most 0.05" in out
    assert "status serving" in out
    assert out.isascii()


def test_refund_second_run_reuses_profile_and_decides_the_same(refunds, tmp_path, capsys):
    first = refunds.run(["--store", str(tmp_path)])
    profile = tmp_path / "refund-approvals-v1.json"
    stored = profile.read_text()
    capsys.readouterr()
    second = refunds.run(["--store", str(tmp_path)])
    out = capsys.readouterr().out
    assert "reused from the profile store" in out
    assert profile.read_text() == stored
    def summary(outcomes):
        return [(o.case_id, o.guard_actions, o.reviews, o.refunds_issued) for o in outcomes]

    assert summary(first) == summary(second)


def test_refund_reviewer_approves_as_proposed(refunds, tmp_path):
    from _shared.review import Reviewer

    reviewer = Reviewer(script={"R-1002": True, "R-1003": False})
    outcomes = _by_case(refunds.run(["--store", str(tmp_path)], reviewer=reviewer, edits={}))
    assert outcomes["R-1002"].reviews == ["approve"]
    assert outcomes["R-1002"].refunds_issued == [{"order_id": "ORD-90106", "amount": 181.99,
                                                  "reason": "changed_mind"}]
    assert outcomes["R-1003"].reviews == ["reject"] and not outcomes["R-1003"].executed


def test_refund_batch_mode_refuses_escalations(refunds, tmp_path, capsys):
    outcomes = _by_case(refunds.run(["--store", str(tmp_path), "--no-human-review"]))
    out = capsys.readouterr().out
    assert outcomes["R-1001"].executed and outcomes["R-1001"].guard_actions == ["allow"]
    for case in ("R-1002", "R-1003"):
        assert outcomes[case].guard_actions == ["escalate"]
        assert outcomes[case].reviews == []  # no interrupt in batch mode
        assert not outcomes[case].executed
        assert outcomes[case].tool_statuses == ["error"]
        assert "review queue" in outcomes[case].final_reply
    assert "needs_human_review" in out
    assert "R-1002, R-1003 refused with an error ToolMessage" in out


def test_refund_async_invoke_and_resume(refunds, tmp_path):
    guard = _calibrated_guard(refunds, refunds.REFUND_GATE, refunds.refund_scorer, "refund_requests.jsonl", tmp_path)

    async def scenario(index: int, decision=None):
        ledger: list = []
        case = refunds.SCENARIOS[index]
        agent = refunds.build_agent(refunds.scripted_model(case), guard, ledger)
        config = {"configurable": {"thread_id": f"async-{case.case_id}"}}
        result = await agent.ainvoke({"messages": [HumanMessage(case.message)]}, config, version="v2")
        pending = pending_reviews(result)
        if decision is not None:
            assert [p["name"] for p in pending] == ["issue_refund"]
            result = await agent.ainvoke(Command(resume={"decisions": [decision]}), config, version="v2")
        else:
            assert pending == []
        return ledger

    assert asyncio.run(scenario(0)) == [{"order_id": "ORD-90101", "amount": 14.99, "reason": "damaged_item"}]
    assert asyncio.run(scenario(2, {"type": "reject", "message": "already refunded"})) == []


def test_refund_resume_on_a_new_worker_honours_the_stored_decision(refunds, tmp_path):
    guard = _calibrated_guard(refunds, refunds.REFUND_GATE, refunds.refund_scorer, "refund_requests.jsonl", tmp_path)
    saver, ledger = InMemorySaver(), []
    case = refunds.SCENARIOS[1]  # R-1002, escalated
    config = {"configurable": {"thread_id": "restart-R-1002"}}
    first = refunds.build_agent(refunds.scripted_model(case), guard, ledger, checkpointer=saver)
    result = first.invoke({"messages": [HumanMessage(case.message)]}, config, version="v2")
    assert [p["name"] for p in pending_reviews(result)] == ["issue_refund"]
    assert [d["action"] for d in guard_decisions(result).values()] == ["escalate"]
    assert len(guard.log) == 1

    # Another worker resumes the same thread: fresh guard (empty cache), same checkpoint.
    fresh_guard = refunds.build_guard(guard.client)
    model = refunds.scripted_model(case)
    model.i = len(model.responses)  # this worker's model only writes the final reply
    second = refunds.build_agent(model, fresh_guard, ledger, checkpointer=saver)
    edit = {"type": "edit", "edited_action": {"name": "issue_refund", "args": {**case.refund, "amount": 169.99}}}
    result = second.invoke(Command(resume={"decisions": [edit]}), config, version="v2")
    assert ledger == [{"order_id": "ORD-90106", "amount": 169.99, "reason": "changed_mind"}]
    assert fresh_guard.log == []  # the stored decision was used; the edited call was not re-scored
    assert "adjusted the amount" in result.value["messages"][-1].content


def test_refund_guard_context_matches_calibration_data(refunds):
    from _shared import calibration

    example = calibration.load_jsonl("refund_requests.jsonl")[0]["context"]
    for case in refunds.SCENARIOS:
        live = refunds.refund_guard_context(case.refund)
        assert live.keys() == example.keys()
        assert live["order"].keys() == example["order"].keys()
        assert live["proposed_refund"].keys() == example["proposed_refund"].keys()
        assert live["policy"] == example["policy"] == refunds.refunds.REFUND_POLICY


def test_refund_unknown_order_fails_closed(refunds, tmp_path):
    guard = _calibrated_guard(refunds, refunds.REFUND_GATE, refunds.refund_scorer, "refund_requests.jsonl", tmp_path)
    decision = guard.check("issue_refund", {"order_id": "ORD-00000", "amount": 5.0, "reason": "damaged_item"})
    assert decision.action == "escalate"
    assert "no order 'ORD-00000'" in decision.reason
    assert guard.check("lookup_order", {"order_id": "ORD-00000"}).action == "allow"  # unguarded tool


@pytest.mark.parametrize(("flags", "variables"), [
    (["--provider", "openai"], ("OPENAI_API_KEY",)),
    (["--provider", "anthropic"], ("ANTHROPIC_API_KEY",)),
    (["--provider", "mock", "--evidence-provider", "azure"], ("AZURE_OPENAI_API_KEY",)),
    (["--provider", "gemini"], ("GOOGLE_API_KEY", "GEMINI_API_KEY")),
])
def test_refund_placeholder_key_stops_before_any_call(refunds, tmp_path, monkeypatch, flags, variables):
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(SystemExit) as stop:
        refunds.main([*flags, "--store", str(tmp_path)])
    assert variables[0] in str(stop.value.code)
    assert list(tmp_path.iterdir()) == []  # nothing was calibrated


def test_refund_dataset_is_reproducible_and_balanced(refunds):
    assert refunds.refunds.main(["--check"]) == 0
    from _shared import calibration

    records = calibration.load_jsonl("refund_requests.jsonl")
    positives = sum(r["label"] for r in records)
    assert 250 <= len(records) <= 400
    assert 0.35 < positives / len(records) < 0.65
    assert all(set(r) == {"id", "context", "label"} for r in records)


# ---------------------------------------------------------------------------
# healthcare_patient_triage.py
# ---------------------------------------------------------------------------


def test_triage_scenarios(triage, tmp_path, capsys):
    outcomes = _by_case(triage.run(["--store", str(tmp_path)]))
    out = capsys.readouterr().out

    cold = outcomes["M-2001"]
    assert cold.guard_actions == ["allow"] and cold.prediction_sets == [["self_care"]]
    assert cold.advice_sent == [{"message_id": "M-2001", "level": "self_care"}]

    chest = outcomes["M-2002"]
    assert chest.guard_actions == ["escalate"]
    assert chest.prediction_sets == [["routine_appointment", "urgent_care"]]
    assert chest.reviews == ["approve"]
    assert chest.advice_sent == [{"message_id": "M-2002", "level": "routine_appointment"}]

    stroke = outcomes["M-2003"]
    assert stroke.guard_actions == ["escalate"] and stroke.prediction_sets == [["emergency"]]
    assert stroke.reviews == ["approve"]
    assert stroke.advice_sent == [{"message_id": "M-2003", "level": "emergency"}]

    cough = outcomes["M-2004"]
    assert cough.guard_actions == ["escalate"] and cough.prediction_sets == [["routine_appointment"]]
    assert cough.reviews == ["edit"]
    # the nurse's level was sent, never the agent's self_care
    assert cough.advice_sent == [{"message_id": "M-2004", "level": "routine_appointment"}]

    assert "but the agent proposed 'self_care'" in out
    assert "Contains the correct answer at least 95% of the time" in out
    assert "Not medical advice" in out
    assert out.isascii()


def test_triage_nurse_rejects(triage, tmp_path):
    from _shared.review import Reviewer

    outcomes = _by_case(triage.run(["--store", str(tmp_path)], reviewer=Reviewer(default=False), edits={}))
    assert outcomes["M-2001"].executed
    for case in ("M-2002", "M-2003", "M-2004"):
        assert outcomes[case].reviews == ["reject"]
        assert not outcomes[case].executed
        assert outcomes[case].tool_statuses == ["error"]


def test_triage_batch_mode_and_profile_reuse(triage, tmp_path, capsys):
    first = triage.run(["--store", str(tmp_path)])
    capsys.readouterr()
    batch = _by_case(triage.run(["--store", str(tmp_path), "--no-human-review"]))
    out = capsys.readouterr().out
    assert "reused from the profile store" in out
    assert batch["M-2001"].executed
    for case in ("M-2002", "M-2003", "M-2004"):
        assert not batch[case].executed and batch[case].tool_statuses == ["error"]
    assert [o.guard_actions for o in first] == [batch[o.case_id].guard_actions for o in first]


def test_triage_urgent_and_emergency_levels_are_never_automated(triage, tmp_path, monkeypatch):
    guard = _calibrated_guard(triage, triage.TRIAGE_SET, triage.triage_scorer, "triage_messages.jsonl", tmp_path)
    from _shared import calibration

    checked = 0
    for record in calibration.load_jsonl("triage_messages.jsonl"):
        message = record["context"]["message"]
        monkeypatch.setitem(triage.MESSAGES, message["message_id"], message)
        for level in ("urgent_care", "emergency"):
            decision = guard.check("send_triage_advice", {"message_id": message["message_id"], "level": level})
            assert decision.action == "escalate"
            checked += 1
        if checked >= 60:
            break


def test_triage_guard_context_matches_calibration_data(triage):
    from _shared import calibration

    example = calibration.load_jsonl("triage_messages.jsonl")[0]["context"]
    for message_id in triage.MESSAGES:
        live = triage.triage_guard_context({"message_id": message_id, "level": "self_care"})
        assert live.keys() == example.keys()
        assert live["message"].keys() == example["message"].keys()
        assert live["protocol"] == example["protocol"] == triage.triage.TRIAGE_PROTOCOL


def test_triage_placeholder_key(triage, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as stop:
        triage.main(["--provider", "openai", "--store", str(tmp_path)])
    assert "OPENAI_API_KEY" in str(stop.value.code)


def test_triage_dataset_is_reproducible(triage):
    assert triage.triage.main(["--check"]) == 0
    from _shared import calibration

    records = calibration.load_jsonl("triage_messages.jsonl")
    assert 250 <= len(records) <= 400
    labels = {r["label"] for r in records}
    assert labels == set(triage.TRIAGE_SET.options)
    assert all(json.dumps(r).isascii() for r in records)
