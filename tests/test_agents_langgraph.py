"""End-to-end tests for the LangGraph agent examples and the LangGraph adapter.

Everything runs in mock mode: LangChain's fake chat model drives the graph and
``MockEvidenceBackend`` scores the evidence, so no key and no network are
needed. The module skips cleanly when LangGraph is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import START, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.types import Command  # noqa: E402

from cli_sdk import Gate  # noqa: E402
from cli_sdk.integrations import GuardRule, ToolGuard  # noqa: E402
from cli_sdk.integrations.langgraph import CLIGuardState, add_cli_guard, pending_review  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "agents" / "langgraph"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"cli_langgraph_example_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def insurance() -> Any:
    return _load("insurance_claims_graph")


@pytest.fixture(scope="module")
def clinical() -> Any:
    return _load("clinical_summary_graph")


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory, insurance: Any, clinical: Any) -> Path:
    """One calibrated profile store shared by the tests that do not test calibration itself."""
    root = tmp_path_factory.mktemp("langgraph_profiles")
    for module, query in ((insurance, insurance.PAYMENT_GATE), (clinical, clinical.CLAIM_CHECK)):
        client = module.build_client(module.parse_args(["--store", str(root)]))
        module.calibration.ensure_calibrated(client, [query], module.calibration.load_jsonl(module.DATASET),
                                             quiet=True)
    return root


def _by_case(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["case_id"]: r for r in results}


# ---------------------------------------------------------------------------
# insurance_claims_graph.py
# ---------------------------------------------------------------------------


def test_insurance_mock_scenarios(insurance: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = _by_case(insurance.run(insurance.parse_args(["--store", str(tmp_path)])))

    allow = results["C-3001"]
    assert allow["audit"]["action"] == "allow"
    assert allow["review"] is None
    assert allow["executed"] == [{"claim_id": "C-3001", "amount": 1350.0}]
    card = allow["audit"]["calls"][0]["guarantee"]
    assert (card["type"], card["method"], card["calibration_n"]) == ("risk_high_probability", "RCPS", 320)
    evidence = allow["audit"]["calls"][0]["evidence"]
    assert evidence["confidence"] >= evidence["threshold"]

    edit = results["C-3002"]
    assert edit["audit"]["action"] == "escalate"
    assert edit["audit"]["calls"][0]["arguments"] == {"claim_id": "C-3002", "amount": 3700.0}  # as proposed
    assert edit["audit"]["review"]["outcome"] == "edit"
    assert edit["audit"]["review"]["executed_calls"][0]["arguments"]["amount"] == 1000.0
    assert edit["executed"] == [{"claim_id": "C-3002", "amount": 1000.0}]                  # as edited

    reject = results["C-3003"]
    assert reject["audit"]["action"] == "escalate"
    assert reject["audit"]["review"]["outcome"] == "reject"
    assert reject["executed"] == []
    assert "No payment was made" in reject["final"]

    out = capsys.readouterr().out
    assert "graph paused at cli_human_review" in out
    assert out.isascii()


def test_insurance_second_run_reuses_profile(insurance: Any, tmp_path: Path,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    args = insurance.parse_args(["--store", str(tmp_path)])
    first = insurance.run(args)
    assert "calibrating 'claim-payments-v1'" in capsys.readouterr().err
    second = insurance.run(args)
    assert "calibrating" not in capsys.readouterr().err
    assert first == second


@pytest.mark.parametrize("decision, executed", [
    ({"action": "approve", "note": "sub-limit waived by endorsement"}, [{"claim_id": "C-3002", "amount": 3700.0}]),
    ({"action": "reject", "note": "sub-limit applies"}, []),
])
def test_insurance_reviewer_approve_and_reject(insurance: Any, store: Path, decision: dict[str, Any],
                                               executed: list[dict[str, Any]]) -> None:
    client = insurance.build_client(insurance.parse_args(["--store", str(store)]))
    guard = insurance.build_guard(client)
    ledger: list[dict[str, Any]] = []
    graph = insurance.build_graph(insurance.agent_model("mock", "C-3002"), guard, insurance.make_tools(ledger))
    result = insurance.process_claim(graph, guard, "C-3002", insurance.AdjusterDesk(script={"C-3002": decision}),
                                     ledger)
    assert result["audit"]["action"] == "escalate"
    assert result["audit"]["review"]["outcome"] == decision["action"]
    assert result["executed"] == executed

    messages = graph.get_state({"configurable": {"thread_id": "claim-C-3002"}}).values["messages"]
    call_ids = {c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    answered = {m.tool_call_id for m in messages if m.type == "tool"}
    assert call_ids == answered          # every tool call has a matching ToolMessage
    if decision["action"] == "reject":
        refusal = [m for m in messages if m.type == "tool" and m.name == "approve_claim_payment"][0]
        assert refusal.status == "error"
        assert json.loads(refusal.content)["reviewer_note"] == "sub-limit applies"


def test_insurance_uncalibrated_guard_fails_closed(insurance: Any, tmp_path: Path) -> None:
    client = insurance.build_client(insurance.parse_args(["--store", str(tmp_path / "empty")]))
    guard = insurance.build_guard(client)
    ledger: list[dict[str, Any]] = []
    graph = insurance.build_graph(insurance.agent_model("mock", "C-3001"), guard, insurance.make_tools(ledger))
    result = insurance.process_claim(graph, guard, "C-3001", insurance.AdjusterDesk(script={}), ledger)
    assert result["audit"]["action"] == "escalate"            # never allowed without a guarantee
    assert result["audit"]["calls"][0]["reason"].startswith("no guarantee available")
    assert result["executed"] == []                           # the default scripted adjuster rejects


def test_insurance_guard_context_matches_calibration_contexts(insurance: Any) -> None:
    example = insurance.calibration.load_jsonl(insurance.DATASET)[0]["context"]
    rule = insurance.build_guard(client=None).rules["approve_claim_payment"]
    live = rule.build_context({"claim_id": "C-3002", "amount": 3700.0})
    # Same keys and the same guideline text; the evidence prompt renders both with sorted keys.
    assert sorted(live) == sorted(example)
    assert sorted(live["claim"]) == sorted(example["claim"])
    assert sorted(live["proposed_payment"]) == sorted(example["proposed_payment"])
    assert live["guideline"] == example["guideline"]


def test_insurance_data_is_the_generator_output(insurance: Any) -> None:
    rows = insurance.calibration.load_jsonl(insurance.DATASET)
    assert rows == json.loads(json.dumps(insurance.claims_data.build_examples()))
    assert len(rows) == 320
    assert all(set(r) == {"id", "context", "label"} and "label" not in r["context"] for r in rows)
    positives = sum(r["label"] for r in rows)
    assert 0.35 < positives / len(rows) < 0.65


def test_insurance_mock_scorer_reads_only_the_context(insurance: Any) -> None:
    context = insurance.claims_data.claim_context(insurance.CLAIMS["C-3001"], 1350.0)
    options = {"true": None, "false": None}
    first = insurance.payment_evidence_scorer(context, insurance.PAYMENT_GATE.instructions, options)
    again = insurance.payment_evidence_scorer(json.loads(json.dumps(context)), "", {})
    assert first == again                                     # deterministic, context-only
    wrong_amount = insurance.claims_data.claim_context(insurance.CLAIMS["C-3001"], 1850.0)
    assert insurance.payment_evidence_scorer(wrong_amount, "", {})["true"] < first["true"]


# ---------------------------------------------------------------------------
# clinical_summary_graph.py
# ---------------------------------------------------------------------------


def test_clinical_mock_scenarios(clinical: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results = _by_case(clinical.run(clinical.parse_args(["--store", str(tmp_path)])))

    faithful = results["D-4001"]
    assert faithful["review"] is None
    assert faithful["verification"]["dropped_claims"] == []
    assert len(faithful["verification"]["retained_claims"]) == 7
    assert faithful["published"] == clinical.MOCK_DRAFTS["D-4001"]

    flawed = results["D-4002"]
    dropped = [d["text"] for d in flawed["verification"]["dropped_claims"]]
    assert dropped == ["Your metoprolol succinate dose was increased to 100 mg once daily.",
                       "You have an appointment at the cardiology clinic on 2026-10-01."]
    assert len(flawed["verification"]["retained_claims"]) == 5
    assert flawed["review"]["action"] == "approve"
    assert flawed["published"] == flawed["verification"]["verified_text"]
    assert not any(text in flawed["published"] for text in dropped)
    assert "every retained claim is supported" in flawed["verification"]["guarantee"]

    out = capsys.readouterr().out
    assert "graph paused at clinician_review" in out
    assert out.isascii()


def test_clinical_second_run_reuses_profile(clinical: Any, tmp_path: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    args = clinical.parse_args(["--store", str(tmp_path)])
    first = clinical.run(args)
    assert "calibrating 'discharge-claims-v1'" in capsys.readouterr().err
    second = clinical.run(args)
    assert "calibrating" not in capsys.readouterr().err
    assert first == second


def test_clinical_clinician_rejects(clinical: Any, store: Path) -> None:
    client = clinical.build_client(clinical.parse_args(["--store", str(store)]))
    portal: list[dict[str, Any]] = []
    graph = clinical.build_graph(clinical.drafting_model("mock", "D-4002"), client, portal)
    reviewer = clinical.Reviewer(script={"D-4002": False}, role="clinician")
    result = clinical.process_encounter(graph, "D-4002", reviewer)
    assert result["review"]["action"] == "reject"
    assert result["published"] is None
    assert portal == []


def test_clinical_uncalibrated_check_fails_closed(clinical: Any, tmp_path: Path) -> None:
    client = clinical.build_client(clinical.parse_args(["--store", str(tmp_path / "empty")]))
    portal: list[dict[str, Any]] = []
    graph = clinical.build_graph(clinical.drafting_model("mock", "D-4001"), client, portal)
    result = clinical.process_encounter(graph, "D-4001", clinical.Reviewer(script={"D-4001": True}))
    assert result["verification"]["heuristic"] is True
    assert result["verification"]["retained_claims"] == []
    assert result["review"]["action"] == "approve"          # approving still publishes nothing unverified
    assert result["published"] is None and portal == []


def test_clinical_context_matches_calibration_contexts(clinical: Any) -> None:
    example = clinical.calibration.load_jsonl(clinical.DATASET)[0]["context"]
    live = clinical.charts_data.summary_context(clinical.CHARTS["D-4002"], clinical.MOCK_DRAFTS["D-4002"])
    assert sorted(live) == sorted(example)
    assert sorted(live["chart"]) == sorted(example["chart"])
    assert set(live["chart"]["medications"][0]) >= {"name", "dose", "unit", "frequency", "status"}


def test_clinical_data_is_the_generator_output(clinical: Any) -> None:
    rows = clinical.calibration.load_jsonl(clinical.DATASET)
    assert rows == json.loads(json.dumps(clinical.charts_data.build_examples()))
    assert len(rows) == 240
    from cli_sdk.evidence._prompts import split_sentences

    for row in rows:   # one claim per sentence, the way drafts are split at run time
        assert split_sentences(row["context"]["answer"]) == [c["text"] for c in row["label"]]


# ---------------------------------------------------------------------------
# both examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["insurance_claims_graph", "clinical_summary_graph"])
def test_openai_with_placeholder_key_exits_before_any_call(name: str, tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    module = _load(name)
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--provider", "openai", "--store", str(tmp_path / "profiles")])
    assert "OPENAI_API_KEY" in str(excinfo.value)
    assert not (tmp_path / "profiles").exists()               # nothing was calibrated


@pytest.mark.parametrize("name, disclaimer", [("insurance_claims_graph", "Not claims-handling or legal advice"),
                                              ("clinical_summary_graph", "Not medical advice")])
def test_main_mock_returns_zero(name: str, disclaimer: str, store: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _load(name).main(["--store", str(store)]) == 0
    out = capsys.readouterr().out
    assert "Synthetic data for demonstration only" in out
    assert disclaimer in out
    assert out.isascii()


# ---------------------------------------------------------------------------
# adapter regression: the reviewer's decision is kept in the checkpoint
# ---------------------------------------------------------------------------


class _FailingClient:
    """Evaluation always fails, so the guard escalates every guarded call (fail closed)."""

    def evaluate(self, context: Any, queries: Any, **kwargs: Any) -> Any:
        raise RuntimeError("evidence model unavailable")


class _ToolModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ToolModel":
        return self


@pytest.mark.parametrize("resume, outcome, amounts", [
    ({"action": "approve", "note": "ok"}, "approve", [120.0]),
    ({"action": "edit", "args": {"amount": 80.0}, "note": "partial"}, "edit", [80.0]),
    ({"action": "reject", "note": "no"}, "reject", []),
    ({"action": "maybe"}, "reject", []),
])
def test_adapter_records_review_decision_in_checkpoint(resume: dict[str, Any], outcome: str,
                                                        amounts: list[float]) -> None:
    ran: list[float] = []

    @tool
    def refund(order_id: str, amount: float) -> str:
        """Issue a refund."""
        ran.append(amount)
        return f"refunded {amount}"

    gate = Gate(instructions="Refund within policy?", calibration_profile="p", guarantee="risk", target=0.1)
    guard = ToolGuard(_FailingClient(), [GuardRule(tool="refund", query=gate)])
    proposal = AIMessage("", id="ai-1", response_metadata={"model_name": "fake-model"},
                         tool_calls=[{"name": "refund", "args": {"order_id": "A1", "amount": 120.0}, "id": "c1"}])
    model = _ToolModel(messages=iter([proposal, AIMessage("done")]), disable_streaming=True)
    builder = StateGraph(CLIGuardState)
    builder.add_node("agent", lambda state: {"messages": [model.invoke(state["messages"])]})
    builder.add_node("tools", ToolNode([refund]))
    builder.add_edge(START, "agent")
    builder.add_edge("tools", "agent")
    add_cli_guard(builder, guard)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"t-{outcome}-{resume['action']}"}}

    paused = graph.invoke({"messages": [HumanMessage("refund A1")]}, config)
    review = pending_review(paused)
    assert review is not None and review["calls"][0]["action"] == "escalate"
    assert "review" not in graph.get_state(config).values["cli_guard"]

    graph.invoke(Command(resume=resume), config)
    record = graph.get_state(config).values["cli_guard"]
    assert record["action"] == "escalate"
    assert record["calls"][0]["arguments"]["amount"] == 120.0     # the proposal is preserved
    assert record["review"]["outcome"] == outcome
    assert record["review"]["note"] == resume.get("note")
    if outcome == "edit":
        assert record["review"]["executed_calls"] == [
            {"id": "c1", "tool": "refund", "arguments": {"order_id": "A1", "amount": 80.0}}]
        [edited] = [m for m in graph.get_state(config).values["messages"] if m.id == "ai-1"]
        assert edited.tool_calls[0]["args"]["amount"] == 80.0          # replaced in place, not appended
        assert edited.response_metadata == {"model_name": "fake-model"}  # metadata kept
    if resume["action"] not in ("approve", "edit", "reject"):
        assert record["review"]["requested"] == "maybe"
    assert ran == amounts
    json.dumps(record)                                            # JSON-safe for durable checkpointers
