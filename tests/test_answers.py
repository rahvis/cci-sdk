"""Answer parsing, the guarantee card, and typed responses.

Payload shapes follow apps/docs/pages/primitives/*.mdx. The two safety
properties under test: anything the SDK cannot recognize as a formal
guarantee is labelled ``heuristic`` (never silently guaranteed), and a Gate
decision the SDK cannot recognize fails closed to ``escalate``.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import pytest

from cli_sdk import (
    BeliefAnswer,
    ClaimAnswer,
    EvaluateResponse,
    GateAnswer,
    Guarantee,
    IntervalAnswer,
    JudgeAnswer,
    RouteAnswer,
    SetAnswer,
    Usage,
)
from cli_sdk.answers import ANSWER_TYPES, DroppedClaim, parse_answer
from cli_sdk.answers.guarantee import GUARANTEE_TYPES
from support import PROFILE, coverage_guarantee, set_answer

# -- one payload per primitive ---------------------------------------------

BELIEF = {
    "type": "belief",
    "probability": 0.81,
    "venn_abers": [0.74, 0.88],
    "guarantee": {
        "type": "calibration",
        "method": "IVAP",
        "calibration_profile": "urgency-v1",
        "calibration_n": 850,
    },
}
INTERVAL = {
    "type": "interval",
    "point_estimate": 2.0,
    "interval": [1, 3],
    "legend": {"0": "calm", "1": "mildly frustrated", "2": "frustrated", "3": "furious"},
    "guarantee": {
        "type": "coverage",
        "alpha": 0.10,
        "method": "CQR",
        "calibration_profile": "frustration-rubric-v1",
        "calibration_n": 640,
    },
}
GATE = {
    "type": "gate",
    "decision": "auto_approve",
    "guarantee": {
        "type": "risk_high_probability",
        "target": 0.01,
        "delta": 0.05,
        "method": "RCPS",
        "calibration_profile": "high-stakes-transfers",
        "calibration_n": 2400,
    },
}
CLAIM = {
    "type": "claim",
    "retained_claims": ["The refund window is 30 days.", "Refunds go to the original card."],
    "dropped_claims": [
        {"text": "Refunds are instant.", "reason": "unsupported", "score": 0.12},
        "Store credit never expires.",
    ],
    "guarantee": {
        "type": "risk",
        "target": 0.05,
        "method": "conformal-factuality",
        "calibration_profile": "rag-factuality-v1",
        "calibration_n": 140,
    },
}
JUDGE = {
    "type": "judge",
    "winner": "response_a",
    "escalated_to": None,
    "guarantee": {
        "type": "risk_high_probability",
        "statement": "human agreement >= 0.90 on non-escalated verdicts",
        "target": 0.10,
        "delta": 0.05,
        "method": "trust-or-escalate",
        "calibration_profile": "pairwise-judge-v2",
        "calibration_n": 1800,
    },
}
ROUTE = {
    "type": "route",
    "served_by": {"provider": "vllm", "model": "llama-3.3-70b"},
    "escalated": False,
    "cost_cents": 0.03,
    "output": "Your order ships tomorrow.",
    "guarantee": {
        "type": "cost_budget",
        "target_cents": 0.4,
        "alpha": 0.10,
        "method": "calibrated-cascade",
        "calibration_profile": "cost-routing-v1",
        "calibration_n": 900,
    },
}


def test_answer_registry_covers_all_seven_primitives():
    assert set(ANSWER_TYPES) == {"belief", "set", "interval", "gate", "claim", "judge", "route"}


def test_parse_answer_rejects_unknown_types():
    with pytest.raises(ValueError, match="unknown answer type"):
        parse_answer({"type": "horoscope"})
    with pytest.raises(ValueError):
        parse_answer({})


class TestBeliefAnswer:
    def test_fields(self):
        a = parse_answer(BELIEF)
        assert isinstance(a, BeliefAnswer)
        assert a.probability == pytest.approx(0.81)
        assert a.venn_abers == (0.74, 0.88)
        assert a.interval_width == pytest.approx(0.14)
        assert a.raw is BELIEF
        assert not a.is_heuristic

    def test_straddles(self):
        a = parse_answer(BELIEF)
        assert a.straddles(0.8)
        assert not a.straddles(0.5)
        assert not a.straddles(0.95)

    def test_without_interval(self):
        a = BeliefAnswer.from_payload({"type": "belief", "probability": 0.4})
        assert a.venn_abers is None and a.interval_width is None
        assert not a.straddles(0.4)
        assert a.is_heuristic  # no guarantee block at all


class TestSetAnswer:
    def test_fields(self):
        a = parse_answer(set_answer(venn_abers={"billing": [0.55, 0.66], "technical": [0.28, 0.37]}))
        assert isinstance(a, SetAnswer)
        assert a.set == ["billing", "technical"]
        assert a.probabilities == {"billing": 0.61, "technical": 0.33, "sales": 0.06}
        assert a.venn_abers == {"billing": (0.55, 0.66), "technical": (0.28, 0.37)}
        assert a.top == "billing"
        assert not a.is_singleton and not a.is_empty

    def test_singleton_and_empty(self):
        single = parse_answer(set_answer(set=["technical"]))
        assert single.is_singleton and single.top == "technical"
        empty = parse_answer(set_answer(set=[]))
        assert empty.is_empty and empty.top is None and not empty.is_singleton

    def test_top_only_considers_members_of_the_set(self):
        a = parse_answer(set_answer(set=["technical", "sales"]))
        assert a.top == "technical"  # billing has higher probability but is outside the set


class TestIntervalAnswer:
    def test_fields(self):
        a = parse_answer(INTERVAL)
        assert isinstance(a, IntervalAnswer)
        assert a.point_estimate == 2.0
        assert a.interval == (1.0, 3.0)
        assert a.width == 2.0
        assert a.legend["3"] == "furious"
        assert a.contains(1) and a.contains(3) and not a.contains(0)


class TestGateAnswer:
    def test_approved(self):
        a = parse_answer(GATE)
        assert isinstance(a, GateAnswer)
        assert a.decision == "auto_approve" and a.approved

    @pytest.mark.parametrize("decision", ["escalate", "abstain"])
    def test_known_non_approvals(self, decision):
        a = parse_answer({**GATE, "decision": decision})
        assert a.decision == decision and not a.approved

    @pytest.mark.parametrize("decision", ["approve", "AUTO_APPROVE", "yes", "", None, 1])
    def test_unknown_decision_fails_closed(self, decision):
        a = parse_answer({**GATE, "decision": decision})
        assert a.decision == "escalate"
        assert not a.approved
        assert a.raw["decision"] == decision  # the raw payload is untouched

    def test_missing_decision_fails_closed(self):
        payload = {k: v for k, v in GATE.items() if k != "decision"}
        assert parse_answer(payload).decision == "escalate"


class TestClaimAnswer:
    def test_fields(self):
        a = parse_answer(CLAIM)
        assert isinstance(a, ClaimAnswer)
        assert a.retained_claims == CLAIM["retained_claims"]
        assert a.dropped_claims == [
            DroppedClaim(text="Refunds are instant.", reason="unsupported", score=0.12),
            DroppedClaim(text="Store credit never expires."),
        ]
        assert a.retention_rate == pytest.approx(0.5)
        assert a.as_text() == "The refund window is 30 days. Refunds go to the original card."
        assert a.as_text("\n").count("\n") == 1

    def test_no_claims(self):
        a = ClaimAnswer.from_payload({"type": "claim", "guarantee": CLAIM["guarantee"]})
        assert a.retention_rate == 0.0 and a.as_text() == ""


class TestJudgeAnswer:
    def test_fields(self):
        a = parse_answer(JUDGE)
        assert isinstance(a, JudgeAnswer)
        assert a.winner == "response_a" and a.escalated_to is None and not a.needs_human
        assert a.guarantee.statement == "human agreement >= 0.90 on non-escalated verdicts"

    def test_human_queue(self):
        a = parse_answer({**JUDGE, "winner": None, "escalated_to": "human_queue"})
        assert a.needs_human and a.winner is None


class TestRouteAnswer:
    def test_fields(self):
        a = parse_answer(ROUTE)
        assert isinstance(a, RouteAnswer)
        assert a.served_by == {"provider": "vllm", "model": "llama-3.3-70b"}
        assert a.escalated is False
        assert a.cost_cents == pytest.approx(0.03)
        assert a.output == "Your order ships tomorrow."
        assert a.guarantee.type == "cost_budget" and a.guarantee.target_cents == 0.4


# -- the guarantee card ------------------------------------------------------


class TestGuarantee:
    def test_every_field_is_parsed(self):
        g = Guarantee.from_payload(coverage_guarantee(stats_version="0.1.0", backend_fingerprint="fp-9"))
        assert g.type == "coverage"
        assert g.method == "APS"
        assert g.alpha == 0.10
        assert g.calibration_profile == PROFILE
        assert g.calibration_n == 1204
        assert g.coverage_ci == (0.884, 0.915)
        assert g.last_audited == "2026-09-18T00:00:00Z"
        assert g.stats_version == "0.1.0"
        assert g.extra == {"backend_fingerprint": "fp-9"}
        assert not g.is_heuristic

    @pytest.mark.parametrize("payload", [None, {}])
    def test_missing_card_is_heuristic(self, payload):
        g = Guarantee.from_payload(payload)
        assert g.type == "heuristic" and g.is_heuristic

    @pytest.mark.parametrize("unknown", ["per_instance", "COVERAGE", "", None])
    def test_unknown_type_is_heuristic(self, unknown):
        g = Guarantee.from_payload({"type": unknown, "alpha": 0.1, "calibration_profile": PROFILE})
        assert g.is_heuristic
        assert g.describe() == "Heuristic value: no formal statistical guarantee."

    def test_card_without_type_is_heuristic(self):
        assert Guarantee.from_payload({"alpha": 0.1}).is_heuristic

    def test_answer_is_heuristic_follows_its_card(self):
        assert parse_answer(set_answer(guarantee={"type": "heuristic"})).is_heuristic
        assert parse_answer(set_answer(guarantee={"type": "made-up"})).is_heuristic
        assert parse_answer({k: v for k, v in set_answer().items() if k != "guarantee"}).is_heuristic
        assert not parse_answer(set_answer()).is_heuristic

    @pytest.mark.parametrize(
        "card, expected",
        [
            (
                coverage_guarantee(),
                f"Contains the correct answer at least 90% of the time on profile '{PROFILE}' (n=1204).",
            ),
            (
                {
                    "type": "risk",
                    "target": 0.05,
                    "calibration_profile": "rag-factuality-v1",
                    "calibration_n": 140,
                },
                "Expected loss at most 0.05 on profile 'rag-factuality-v1' (n=140).",
            ),
            (
                GATE["guarantee"],
                "With 95% confidence, risk at most 0.01 on profile 'high-stakes-transfers' (n=2400).",
            ),
            (
                {"type": "fdr", "target": 0.05, "calibration_profile": PROFILE},
                f"Expected wrong fraction among approved decisions at most 0.05 on profile '{PROFILE}'.",
            ),
            (
                ROUTE["guarantee"],
                "Cost per request at most 0.4 cents with 90% probability on profile 'cost-routing-v1' (n=900).",
            ),
            (
                {"type": "anytime", "calibration_profile": PROFILE},
                f"False-alarm rate controlled at any stopping time on profile '{PROFILE}'.",
            ),
            ({"type": "heuristic"}, "Heuristic value: no formal statistical guarantee."),
        ],
        ids=["coverage", "risk", "risk_high_probability", "fdr", "cost_budget", "anytime", "heuristic"],
    )
    def test_describe(self, card, expected):
        assert Guarantee.from_payload(card).describe() == expected

    @pytest.mark.parametrize(
        "alpha, level",
        [(0.10, "90%"), (0.05, "95%"), (0.025, "97.5%"), (0.005, "99.5%"), (0.001, "99.9%")],
    )
    def test_describe_never_rounds_a_coverage_level_up(self, alpha, level):
        text = Guarantee(type="coverage", alpha=alpha).describe()
        assert f"at least {level} of the time" in text
        assert "100%" not in text

    def test_describe_never_rounds_a_confidence_level_up(self):
        text = Guarantee(type="risk_high_probability", target=0.01, delta=0.005).describe()
        assert text.startswith("With 99.5% confidence")

    @pytest.mark.parametrize(
        "card",
        [
            BELIEF["guarantee"],  # IVAP calibration card: no alpha, per the Belief docs
            {"type": "coverage"},
            {"type": "risk", "alpha": 0.05},
            {"type": "risk_high_probability", "alpha": 0.10, "delta": 0.05},
            {"type": "fdr"},
            {"type": "cost_budget"},
        ],
        ids=["belief-ivap", "coverage-bare", "risk-alpha", "rhp-alpha", "fdr-bare", "cost-bare"],
    )
    def test_formal_guarantee_is_never_described_as_heuristic(self, card):
        g = Guarantee.from_payload(card)
        assert not g.is_heuristic
        assert "Heuristic" not in g.describe()

    def test_risk_card_carrying_alpha_describes_its_level(self):
        assert (
            Guarantee.from_payload({"type": "risk", "alpha": 0.05}).describe()
            == "Expected loss at most 0.05."
        )
        rhp = Guarantee.from_payload({"type": "risk_high_probability", "alpha": 0.1, "delta": 0.05})
        assert rhp.describe() == "With 95% confidence, risk at most 0.1."

    @pytest.mark.parametrize("guarantee_type", sorted(GUARANTEE_TYPES - {"heuristic"}))
    def test_describe_never_claims_a_per_instance_guarantee(self, guarantee_type):
        text = Guarantee(type=guarantee_type, alpha=0.1, target=0.05, delta=0.05, target_cents=1.0).describe()
        lowered = text.lower()
        for phrase in ("this answer is", "this decision is", "guaranteed correct", "100%"):
            assert phrase not in lowered, text


# -- the evaluate response ----------------------------------------------------


def _body(**answers):
    return {
        "backend": {"provider": "openai", "model": "gpt-4.1-2025-04-14"},
        "answers": answers,
        "usage": {"backend_calls": 2, "backend_tokens": 512},
        "warnings": ["profile 'urgency-v1' was last audited 40 days ago"],
        "request_id": "req_body",
    }


class TestEvaluateResponse:
    def test_all_seven_answer_types_in_one_response(self):
        body = _body(
            urgent=BELIEF,
            department=set_answer(),
            frustration=INTERVAL,
            release=GATE,
            facts=CLAIM,
            verdict=JUDGE,
            answer=ROUTE,
        )
        r = EvaluateResponse.from_payload(body)
        assert {qid: type(a) for qid, a in r.answers.items()} == {
            "urgent": BeliefAnswer,
            "department": SetAnswer,
            "frustration": IntervalAnswer,
            "release": GateAnswer,
            "facts": ClaimAnswer,
            "verdict": JudgeAnswer,
            "answer": RouteAnswer,
        }
        assert r.heuristic_answers == []
        assert r.usage == Usage(backend_calls=2, backend_tokens=512)
        assert r.backend == {"provider": "openai", "model": "gpt-4.1-2025-04-14"}
        assert r.warnings == ["profile 'urgency-v1' was last audited 40 days ago"]
        assert r.request_id == "req_body"
        assert "department" in repr(r)

    def test_header_request_id_wins(self):
        assert EvaluateResponse.from_payload(_body(), request_id="req_header").request_id == "req_header"

    def test_heuristic_answers_lists_query_ids(self):
        r = EvaluateResponse.from_payload(
            _body(department=set_answer(), route={**GATE, "guarantee": {"type": "heuristic"}})
        )
        assert r.heuristic_answers == ["route"]

    def test_empty_payload_defaults(self):
        r = EvaluateResponse.from_payload({})
        assert r.answers == {} and r.backend == {} and r.warnings == [] and r.request_id is None
        assert r.usage == Usage()


class RoutingResponse(EvaluateResponse):
    department: SetAnswer
    route: GateAnswer


class ExtendedRoutingResponse(RoutingResponse):
    urgent: BeliefAnswer


class TestTypedResponse:
    def test_attribute_access(self):
        r = RoutingResponse.from_payload(_body(department=set_answer(), route=GATE))
        assert isinstance(r, RoutingResponse)
        assert isinstance(r.department, SetAnswer)
        assert r.department is r.answers["department"]
        assert r.route.approved

    def test_annotations_are_inherited(self):
        r = ExtendedRoutingResponse.from_payload(_body(department=set_answer(), route=GATE, urgent=BELIEF))
        assert r.urgent.venn_abers == (0.74, 0.88)
        assert r.route.decision == "auto_approve"

    def test_missing_answer_raises_value_error(self):
        with pytest.raises(ValueError, match="'route'"):
            RoutingResponse.from_payload(_body(department=set_answer()))

    def test_inherited_missing_answer_raises_value_error(self):
        with pytest.raises(ValueError, match="'urgent'"):
            ExtendedRoutingResponse.from_payload(_body(department=set_answer(), route=GATE))

    def test_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError, match="department.*GateAnswer.*SetAnswer"):
            RoutingResponse.from_payload(_body(department=GATE, route=GATE))

    def test_extra_answers_are_allowed(self):
        r = RoutingResponse.from_payload(_body(department=set_answer(), route=GATE, verdict=JUDGE))
        assert isinstance(r.answers["verdict"], JudgeAnswer)

    def test_base_fields_and_private_names_are_not_answers(self):
        class WithBaseFields(EvaluateResponse):
            usage: Usage
            _cache: Optional[dict]
            department: SetAnswer

        r = WithBaseFields.from_payload(_body(department=set_answer()))
        assert isinstance(r.usage, Usage)
        assert isinstance(r.department, SetAnswer)

    def test_class_constants_are_not_answers(self):
        class Versioned(EvaluateResponse):
            SCHEMA_VERSION: ClassVar[int] = 2
            department: SetAnswer

        r = Versioned.from_payload(_body(department=set_answer()))
        assert r.SCHEMA_VERSION == 2 and isinstance(r.department, SetAnswer)


def test_null_usage_counts_do_not_break_parsing():
    r = EvaluateResponse.from_payload(
        {"answers": {"department": set_answer()}, "usage": {"backend_calls": 0, "backend_tokens": None}}
    )
    assert r.usage == Usage(backend_calls=0, backend_tokens=0)
    assert isinstance(r.answers["department"], SetAnswer)
