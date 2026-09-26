"""Regressions in local mode found while writing the agent-framework docs.

1. At access level L0 every score is a smoothed frequency over
   ``sample_count`` draws, so ``sample_count`` is part of the scoring
   function. A profile calibrated with one sample count must not be served
   by a client configured with another.
2. A profile's ``minimum_n`` and ``status`` must match what evaluation
   requires. RCPS, Learn-then-Test and Venn-Abers need more examples than
   the split-conformal floor ``ceil((1 - alpha) / alpha)``.
"""

from __future__ import annotations

import random

import pytest

from cli_sdk import Belief, Gate, Interval, Judge, Route, Set
from cli_sdk.evidence import MockEvidenceBackend
from cli_sdk.integrations import GuardRule, ToolGuard
from cli_sdk.local import LocalCLIClient, engine
from cli_sdk.local.store import ProfileRecord


def _scorer(context, instructions, options):
    if set(options) == {"true", "false"}:
        p = 0.9 if context["x"] > 0.5 else 0.2
        return {"true": p, "false": 1.0 - p}
    keys = list(options)
    return {k: (3.0 if i == context["i"] % len(keys) else 1.0) for i, k in enumerate(keys)}


def _binary_examples(n: int, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        x = rng.random()
        out.append({"context": {"x": x, "i": i}, "label": (x > 0.5) == (rng.random() < 0.95)})
    return out


# ---------------------------------------------------------------------------
# 1. sample_count is part of an L0 scoring function
# ---------------------------------------------------------------------------


def test_l0_profile_is_stale_under_a_different_sample_count(tmp_path):
    gate = Gate(instructions="Approve?", calibration_profile="g", guarantee="risk", target=0.05)
    calibrated = LocalCLIClient(MockEvidenceBackend(scorer=_scorer, access_level="L0"), store=tmp_path,
                                sample_count=20)
    calibrated.calibrate(gate, _binary_examples(200))
    assert calibrated.calibration_status(gate) == (True, "")
    assert calibrated.get_profile("g").backend_fingerprint["sample_count"] == 20

    other = LocalCLIClient(MockEvidenceBackend(scorer=_scorer, access_level="L0"), store=tmp_path,
                           sample_count=8)
    ok, reason = other.calibration_status(gate)
    assert not ok and "different backend or settings" in reason
    answer = other.evaluate({"x": 0.9, "i": 999}, {"g": gate}).answers["g"]
    assert answer.is_heuristic and answer.decision == "escalate"

    guard = ToolGuard(other, [GuardRule(tool="approve", query=gate, context=lambda a: {"x": a["x"], "i": 0})])
    assert guard.check("approve", {"x": 0.9}).action == "escalate"


def test_l1_profile_ignores_sample_count(tmp_path):
    gate = Gate(instructions="Approve?", calibration_profile="g", guarantee="risk", target=0.05)
    LocalCLIClient(MockEvidenceBackend(scorer=_scorer), store=tmp_path, sample_count=20).calibrate(
        gate, _binary_examples(200))
    other = LocalCLIClient(MockEvidenceBackend(scorer=_scorer), store=tmp_path, sample_count=8)
    assert other.calibration_status(gate) == (True, "")
    assert "sample_count" not in other.get_profile("g").backend_fingerprint


# ---------------------------------------------------------------------------
# 2. minimum_n and status match what evaluation requires
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query, expected_minimum", [
    (Gate(instructions="Approve?", calibration_profile="p", guarantee="risk", target=0.05), 19),
    (Gate(instructions="Approve?", calibration_profile="p", guarantee="risk_high_probability",
          target=0.05, delta=0.10), 47),
    (Gate(instructions="Approve?", calibration_profile="p", guarantee="fdr", target=0.10, delta=0.10), 40),
    (Belief(instructions="True?", calibration_profile="p"), 20),
    (Set(instructions="Which?", options={"a": None, "b": None, "c": None}, calibration_profile="p",
         alpha=0.05), 19),
    (Interval(instructions="How severe?", levels=["low", "mid", "high"], calibration_profile="p",
              alpha=0.10), 9),
])
def test_profile_minimum_matches_evaluation(tmp_path, query, expected_minimum):
    assert engine.minimum_examples(query.to_payload()) == expected_minimum
    client = LocalCLIClient(MockEvidenceBackend(scorer=_scorer), store=tmp_path)

    def examples(n):
        if isinstance(query, (Gate, Belief)):
            return _binary_examples(n)
        keys = list(query.options) if isinstance(query, Set) else list(query.levels)
        return [{"context": {"x": 0.0, "i": i}, "label": keys[i % len(keys)]} for i in range(n)]

    below = client.calibrate(query, examples(expected_minimum - 1))
    assert below.minimum_n == expected_minimum
    assert below.status == "collecting" and not below.can_serve_guarantees
    assert client.evaluate({"x": 0.9, "i": 1}, {"q": query}).answers["q"].is_heuristic

    at = client.calibrate(query, examples(expected_minimum))
    assert at.status == "serving" and at.can_serve_guarantees
    assert not client.evaluate({"x": 0.9, "i": 1}, {"q": query}).answers["q"].is_heuristic


def _record(n: int, evidence: dict, label: str) -> ProfileRecord:
    return ProfileRecord(name="p", query_type="x", query={}, query_fingerprint="", backend_fingerprint={},
                         records=[{"evidence": evidence, "label": label} for _ in range(n)])


def test_judge_and_route_minimums_match_their_threshold_checks():
    judge = Judge(instructions="Which is better?", calibration_profile="p", alpha=0.1,
                  cascade=[{"backend": "small"}, {"backend": "large"}, "human_queue"])
    needed = engine.minimum_examples(judge.to_payload())
    evidence = {"stages": [[0.9, 0.1], [0.9, 0.1]]}
    with pytest.raises(engine.Heuristic):
        engine.judge_thresholds(judge, _record(needed - 1, evidence, "response_a"))
    engine.judge_thresholds(judge, _record(needed, evidence, "response_a"))

    task = Set(instructions="Which?", options={"a": None, "b": None}, calibration_profile="t")
    route = Route(cascade=[{"backend": "small"}, {"backend": "large"}], calibration_profile="p",
                  guarantee="accuracy", alpha=0.1, task=task)
    needed = engine.minimum_examples(route.to_payload())
    evidence = {"tiers": [[0.8, 0.2], [0.9, 0.1]]}
    with pytest.raises(engine.Heuristic):
        engine.route_thresholds(route, _record(needed - 1, evidence, "a"))
    engine.route_thresholds(route, _record(needed, evidence, "a"))


# ---------------------------------------------------------------------------
# 3. LangChainEvidenceBackend fingerprints the wrapped chat model's settings
# ---------------------------------------------------------------------------


class _StubChatModel:
    """Duck-typed stand-in for a LangChain chat model (no langchain import needed)."""

    def __init__(self, temperature, base_url="http://localhost:8000/v1"):
        self.model_name = "google/gemma-4-12B-it"
        self.temperature = temperature
        self.openai_api_base = base_url


def test_langchain_backend_fingerprint_tracks_chat_model_settings(tmp_path):
    from cli_sdk.evidence import LangChainEvidenceBackend

    hot = LangChainEvidenceBackend(_StubChatModel(temperature=1.0)).fingerprint()
    cool = LangChainEvidenceBackend(_StubChatModel(temperature=0.7)).fingerprint()
    moved = LangChainEvidenceBackend(_StubChatModel(temperature=1.0, base_url="http://other:8000/v1")).fingerprint()
    assert hot["temperature"] == 1.0 and cool["temperature"] == 0.7
    assert hot != cool and hot != moved
    assert hot == LangChainEvidenceBackend(_StubChatModel(temperature=1.0)).fingerprint()
    assert hot["chat_model"]["class"].endswith("_StubChatModel")
