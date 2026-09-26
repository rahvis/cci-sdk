"""The Microsoft Agent Framework examples, end to end in mock mode (keyless and offline).

Skipped when ``agent_framework`` (agent-framework-core) is not installed, so
the core suite runs without it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

pytest.importorskip("agent_framework")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "agents"


def _load(name: str):
    path = EXAMPLES / "agent_framework" / f"{name}.py"
    module_name = f"maf_example_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def credit():
    return _load("credit_underwriting_agent")


@pytest.fixture(scope="module")
def pv():
    return _load("pharmacovigilance_agent")


def _by_case(outcomes) -> dict[str, Any]:
    return {o.case_id: o for o in outcomes}


# ---------------------------------------------------------------------------
# Credit underwriting (Set, Mondrian by channel)
# ---------------------------------------------------------------------------


def test_credit_scenarios_in_mock_mode(credit, tmp_path, capsys):
    outcomes = _by_case(credit.run(["--store", str(tmp_path)]))

    strong = outcomes["L-7001"]
    assert (strong.action, strong.calibrated_set, strong.executed, strong.review) == ("allow", ["A"], True, None)

    thin = outcomes["L-7002"]
    assert thin.action == "escalate" and thin.calibrated_set == ["B", "C"]
    assert thin.review is True and thin.executed is True

    weak = outcomes["L-7003"]
    assert weak.action == "escalate" and weak.proposed["tier"] == "D"
    assert set(weak.calibrated_set) <= {"D", "E"}
    assert weak.review is False and weak.executed is False

    assert credit.TIER_ASSIGNMENTS == {"L-7001": "A", "L-7002": "B"}
    out = capsys.readouterr().out
    assert "Synthetic data for demonstration only" in out
    assert "per-channel (Mondrian) status" in out and "Fair-lending note" in out
    assert not any((tmp_path / "pending_reviews").glob("*.json")), "resolved reviews leave no pending file"


def test_credit_profile_is_serving_and_calibrated_per_channel(credit, tmp_path):
    credit.run(["--store", str(tmp_path)])
    from cli_sdk.local import LocalCLIClient

    profile = LocalCLIClient(store=tmp_path).get_profile(credit.TIER_QUERY.calibration_profile)
    assert profile.status == "serving" and profile.n == 320 and profile.group_by == "channel"
    assert {g: s.n for g, s in profile.groups.items()} == {"branch": 140, "broker": 40, "online": 140}
    assert all(s.status == "calibrated" for s in profile.groups.values())


def test_credit_second_run_reuses_the_profile_and_decides_the_same(credit, tmp_path, capsys):
    first = credit.run(["--store", str(tmp_path)])
    assert "calibrating 'credit-tiers-v1'" in capsys.readouterr().err
    profile_file = tmp_path / "credit-tiers-v1.json"
    stamp = profile_file.stat().st_mtime_ns

    second = credit.run(["--store", str(tmp_path)])
    assert "calibrating" not in capsys.readouterr().err
    assert profile_file.stat().st_mtime_ns == stamp
    assert second == first


def test_credit_review_is_persisted_as_json_while_pending(credit, tmp_path):
    from _shared.review import Reviewer

    seen: dict[str, Mapping[str, Any]] = {}

    class Inspecting(Reviewer):
        def decide(self, case_id, request):
            path = tmp_path / "pending_reviews" / f"{case_id}.json"
            saved = json.loads(path.read_text(encoding="utf-8"))  # exists while the review is pending
            seen[case_id] = saved
            return super().decide(case_id, request)

    credit.run(["--store", str(tmp_path)], reviewer=Inspecting(script=credit.REVIEW_SCRIPT))
    assert set(seen) == {"L-7002", "L-7003"}
    saved = seen["L-7002"]
    assert set(saved) == {"session", "request", "cli"}
    tickets = saved["session"]["state"]["cli_guard"]
    assert len(tickets) == 1 and next(iter(tickets.values()))["fingerprint"]
    assert saved["request"]["function_call"]["arguments"] == {"application_id": "L-7002", "tier": "B"}
    assert saved["cli"]["action"] == "escalate"
    assert saved["cli"]["guarantee"]["statement"].startswith("Contains the correct answer at least 90%")
    # Mondrian: the count is the online channel's, and the review payload says so.
    assert saved["cli"]["guarantee"]["group"] == "online" and saved["cli"]["guarantee"]["calibration_n"] == 140


def test_credit_underwriter_rejection_records_nothing(credit, tmp_path):
    from _shared.review import Reviewer

    outcomes = _by_case(credit.run(["--store", str(tmp_path)],
                                   reviewer=Reviewer(script={"L-7002": False, "L-7003": False})))
    assert outcomes["L-7002"].executed is False and outcomes["L-7002"].review is False
    assert "rejected" in outcomes["L-7002"].agent_text
    assert credit.TIER_ASSIGNMENTS == {"L-7001": "A"}


def _guarded_agent(module, tmp_path):
    from agent_framework import Agent

    from _shared import calibration, providers
    from cli_sdk.integrations.agent_framework import CLIGuardMiddleware
    from cli_sdk.local import LocalCLIClient

    client = LocalCLIClient(providers.evidence_backend("mock", mock_scorer=module.tier_scorer), store=tmp_path)
    calibration.ensure_calibrated(client, [module.TIER_QUERY], calibration.load_jsonl(module.DATASET), quiet=True)
    guard = module.build_guard(client)

    def make_agent():
        return Agent(client=module.chat_client("mock", planner=module.scripted_proposal),
                     tools=[module.assign_risk_tier], middleware=[CLIGuardMiddleware(guard)])

    return make_agent


def test_a_tampered_approval_cannot_change_the_call(credit, tmp_path):
    from agent_framework import AgentSession, Content

    from cli_sdk.integrations.agent_framework import pending_reviews, review_message

    make_agent = _guarded_agent(credit, tmp_path)

    async def scenario():
        credit.TIER_ASSIGNMENTS.clear()
        agent = make_agent()
        session = agent.create_session()
        result = await agent.run(credit.user_message("L-7002"), session=session)
        (review,) = pending_reviews(result)
        saved = json.loads(json.dumps({"session": session.to_dict(), "request": review["request"].to_dict()}))
        saved["request"]["function_call"]["arguments"]["tier"] = "A"  # edited while stored
        resumed = make_agent()
        await resumed.run(review_message(Content.from_dict(saved["request"]), True),
                          session=AgentSession.from_dict(saved["session"]))

    asyncio.run(scenario())
    # The framework runs the call it paused, which is the call the guard escalated.
    assert credit.TIER_ASSIGNMENTS == {"L-7002": "B"}


def test_escalation_surfaces_on_a_streamed_run(credit, tmp_path):
    from cli_sdk.integrations.agent_framework import pending_reviews, review_message

    make_agent = _guarded_agent(credit, tmp_path)

    async def scenario():
        credit.TIER_ASSIGNMENTS.clear()
        agent = make_agent()
        session = agent.create_session()
        stream = agent.run(credit.user_message("L-7002"), session=session, stream=True)
        on_stream = [request async for update in stream for request in update.user_input_requests]
        (review,) = pending_reviews(await stream.get_final_response())
        assert len(on_stream) == 1 and credit.TIER_ASSIGNMENTS == {}
        resumed = agent.run(review_message(review, True), session=session, stream=True)
        async for _ in resumed:
            pass
        return await resumed.get_final_response()

    final = asyncio.run(scenario())
    assert credit.TIER_ASSIGNMENTS == {"L-7002": "B"} and "Tier B recorded" in final.text


def test_escalation_without_a_session_fails_closed(credit, tmp_path):
    from agent_framework import MiddlewareFailure

    make_agent = _guarded_agent(credit, tmp_path)
    credit.TIER_ASSIGNMENTS.clear()
    with pytest.raises(MiddlewareFailure, match="requires an AgentSession"):
        asyncio.run(make_agent().run(credit.user_message("L-7002")))
    assert credit.TIER_ASSIGNMENTS == {}


# ---------------------------------------------------------------------------
# Pharmacovigilance (ordinal Interval)
# ---------------------------------------------------------------------------


def test_pharmacovigilance_scenarios_in_mock_mode(pv, tmp_path, capsys):
    outcomes = _by_case(pv.run(["--store", str(tmp_path)]))

    admitted = outcomes["AE-8001"]
    assert admitted.action == "allow" and admitted.executed is True and admitted.review is None
    assert set(admitted.interval) <= set(pv.SERIOUS)

    borderline = outcomes["AE-8002"]
    assert borderline.action == "escalate" and borderline.interval == ["grade_2", "grade_3"]
    assert borderline.review is True and borderline.executed is True

    rash = outcomes["AE-8003"]
    assert rash.action == "block" and rash.executed is False and rash.review is None
    assert set(rash.interval) <= set(pv.NON_SERIOUS)

    assert pv.SUBMITTED_REPORTS == {"AE-8001": "grade_3", "AE-8002": "grade_3"}
    out = capsys.readouterr().out
    assert "Not medical advice" in out and "routine periodic reporting" in out


def test_pharmacovigilance_physician_rejection_submits_nothing(pv, tmp_path):
    from _shared.review import Reviewer

    outcomes = _by_case(pv.run(["--store", str(tmp_path)], reviewer=Reviewer(script={"AE-8002": False})))
    assert outcomes["AE-8002"].action == "escalate"
    assert outcomes["AE-8002"].executed is False and outcomes["AE-8002"].review is False
    assert pv.SUBMITTED_REPORTS == {"AE-8001": "grade_3"}


def test_pharmacovigilance_second_run_is_identical(pv, tmp_path, capsys):
    first = pv.run(["--store", str(tmp_path)])
    capsys.readouterr()
    second = pv.run(["--store", str(tmp_path)])
    assert "calibrating" not in capsys.readouterr().err
    assert second == first


# ---------------------------------------------------------------------------
# Shared: keys, data and contexts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("example", ["credit_underwriting_agent", "pharmacovigilance_agent"])
@pytest.mark.parametrize("flags", [["--provider", "openai"], ["--evidence-provider", "openai"]])
def test_placeholder_openai_key_stops_before_any_request(example, flags, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    module = _load(example)
    with pytest.raises(SystemExit) as stop:
        module.main([*flags, "--store", str(tmp_path / "store")])
    assert "OPENAI_API_KEY" in str(stop.value)
    assert not (tmp_path / "store").exists(), "nothing is calibrated or stored with a placeholder key"


def test_committed_datasets_match_their_generators(credit, pv):
    from _shared.calibration import load_jsonl

    for generator, dataset in ((credit.credit_data, credit.DATASET), (pv.ae_data, pv.DATASET)):
        assert generator.generate() == load_jsonl(dataset)


def test_inference_contexts_have_the_calibration_shape(credit, pv):
    from _shared.calibration import load_jsonl

    for module, dataset, ids, key in ((credit, credit.DATASET, credit.APPLICATIONS, "application_id"),
                                      (pv, pv.DATASET, pv.CASES, "case_id")):
        rows = load_jsonl(dataset)
        calibration_keys = {tuple(sorted(row["context"])) for row in rows}
        assert len(calibration_keys) == 1, "every calibration context has the same keys"
        assert all("label" not in row["context"] for row in rows), "the label is never part of the context"
        for case_id in ids:
            context = module.guard_context({key: case_id})
            assert tuple(sorted(context)) in calibration_keys


def test_unknown_case_escalates_instead_of_running(credit, tmp_path):
    from _shared import calibration, providers
    from cli_sdk.local import LocalCLIClient

    client = LocalCLIClient(providers.evidence_backend("mock", mock_scorer=credit.tier_scorer), store=tmp_path)
    calibration.ensure_calibrated(client, [credit.TIER_QUERY], calibration.load_jsonl(credit.DATASET), quiet=True)
    decision = credit.build_guard(client).check(credit.TOOL, {"application_id": "L-9999", "tier": "A"})
    assert decision.action == "escalate" and "could not build the evaluation context" in decision.reason
