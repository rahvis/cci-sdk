"""Query primitives: client-side validation and the flat wire format.

``Query.to_payload()`` is exactly what goes into ``queries[<id>]`` in the
``POST /v1/evaluate`` body: a flat object with ``type`` plus every field the
caller set. Top-level fields left as ``None`` are omitted, and there is no
nested ``calibration`` object anywhere.
"""

from __future__ import annotations

import pytest

from cli_sdk import (
    Belief,
    Claim,
    ConfigurationError,
    Gate,
    Interval,
    Judge,
    OpenAIBackend,
    Query,
    Route,
    Set,
)
from support import PROFILE

SET_OPTIONS = {"billing": None, "technical": None, "sales": None}


def _assert_flat(payload: dict) -> None:
    assert "calibration" not in payload
    assert all(value is not None for value in payload.values()), payload


def test_configuration_error_is_also_a_value_error():
    # Primitives document ConfigurationError for a missing required field;
    # subclassing ValueError keeps plain `except ValueError` callers working.
    assert issubclass(ConfigurationError, ValueError)


@pytest.mark.parametrize("cls", [Belief, Set, Interval, Gate, Claim, Judge, Route])
def test_every_primitive_is_a_query_with_a_fixed_type(cls):
    assert issubclass(cls, Query)
    with pytest.raises(TypeError):
        cls(type="something-else")  # `type` is init=False on every primitive


# -- Belief ---------------------------------------------------------------


class TestBelief:
    def test_payload(self):
        q = Belief(
            instructions="Is this ticket urgent?",
            calibration_profile="urgency-v1",
            criteria={"urgent": "customer is blocked"},
        )
        assert q.to_payload() == {
            "type": "belief",
            "instructions": "Is this ticket urgent?",
            "calibration_profile": "urgency-v1",
            "criteria": {"urgent": "customer is blocked"},
        }

    def test_omits_unset_criteria(self):
        payload = Belief(instructions="Urgent?", calibration_profile="urgency-v1").to_payload()
        assert payload == {"type": "belief", "instructions": "Urgent?", "calibration_profile": "urgency-v1"}

    def test_structured_instructions_pass_through(self):
        instructions = {"question": "Does the reply follow policy?", "policy": ["no refunds after 30 days"]}
        payload = Belief(instructions=instructions, calibration_profile="policy-v1").to_payload()
        assert payload["instructions"] == instructions

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"calibration_profile": "p"}, "instructions"),
            ({"instructions": "Urgent?"}, "calibration_profile"),
            ({"instructions": "", "calibration_profile": "p"}, "instructions"),
        ],
    )
    def test_validation(self, kwargs, message):
        with pytest.raises(ConfigurationError, match=message):
            Belief(**kwargs)


# -- Set ------------------------------------------------------------------


class TestSet:
    def test_payload_is_flat(self):
        q = Set(
            instructions="Which team should handle this ticket?",
            options={"billing": None, "technical": "API errors, outages", "sales": None},
            calibration_profile=PROFILE,
            alpha=0.1,
            method="APS",
            group_by="plan_tier",
        )
        payload = q.to_payload()
        assert payload == {
            "type": "set",
            "instructions": "Which team should handle this ticket?",
            "options": {"billing": None, "technical": "API errors, outages", "sales": None},
            "calibration_profile": PROFILE,
            "alpha": 0.1,
            "method": "APS",
            "group_by": "plan_tier",
        }
        assert "calibration" not in payload

    def test_options_without_descriptions_are_kept(self):
        # The documented form is {"billing": null, ...}: a null description
        # must not make the option itself disappear from the request.
        payload = Set(
            instructions="Which team?", options=SET_OPTIONS, calibration_profile=PROFILE
        ).to_payload()
        assert payload["options"] == {"billing": None, "technical": None, "sales": None}
        assert list(payload["options"]) == ["billing", "technical", "sales"]

    def test_unset_fields_are_omitted(self):
        payload = Set(
            instructions="Which team?", options=SET_OPTIONS, calibration_profile=PROFILE
        ).to_payload()
        assert set(payload) == {"type", "instructions", "options", "calibration_profile"}
        for key in ("alpha", "method", "group_by", "backend_access_hint"):
            assert key not in payload

    @pytest.mark.parametrize("method", ["LAC", "APS", "RAPS"])
    def test_accepts_every_set_method(self, method):
        q = Set(
            instructions="Which?", options={"a": None, "b": None}, calibration_profile=PROFILE, method=method
        )
        assert q.to_payload()["method"] == method

    def test_backend_access_hint_is_sent(self):
        q = Set(
            instructions="Which?",
            options={"a": None, "b": None},
            calibration_profile=PROFILE,
            backend_access_hint="logprobs",
        )
        assert q.to_payload()["backend_access_hint"] == "logprobs"

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"options": SET_OPTIONS, "calibration_profile": PROFILE}, "instructions"),
            (
                {"instructions": "Which?", "options": {"a": None}, "calibration_profile": PROFILE},
                "2 `options`",
            ),
            ({"instructions": "Which?", "calibration_profile": PROFILE}, "2 `options`"),
            ({"instructions": "Which?", "options": SET_OPTIONS}, "calibration_profile"),
            (
                {
                    "instructions": "Which?",
                    "options": SET_OPTIONS,
                    "calibration_profile": PROFILE,
                    "method": "CQR",
                },
                "method",
            ),
            (
                {
                    "instructions": "Which?",
                    "options": SET_OPTIONS,
                    "calibration_profile": PROFILE,
                    "method": "aps",
                },
                "method",
            ),
        ],
    )
    def test_validation(self, kwargs, message):
        with pytest.raises(ConfigurationError, match=message):
            Set(**kwargs)

    @pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
    def test_alpha_must_be_a_probability(self, alpha):
        with pytest.raises(ConfigurationError, match="alpha"):
            Set(instructions="Which?", options=SET_OPTIONS, calibration_profile=PROFILE, alpha=alpha)


# -- Interval -------------------------------------------------------------


class TestInterval:
    LEVELS = ["calm", "mildly frustrated", "frustrated", "furious"]

    def test_payload(self):
        q = Interval(
            instructions="How frustrated is the customer?",
            levels=self.LEVELS,
            calibration_profile="frustration-rubric-v1",
            alpha=0.1,
            method="CQR",
        )
        assert q.to_payload() == {
            "type": "interval",
            "instructions": "How frustrated is the customer?",
            "levels": self.LEVELS,
            "calibration_profile": "frustration-rubric-v1",
            "alpha": 0.1,
            "method": "CQR",
        }

    @pytest.mark.parametrize("n_levels", [2, 10])
    def test_level_count_bounds_are_inclusive(self, n_levels):
        levels = [f"level-{i}" for i in range(n_levels)]
        assert (
            Interval(instructions="How?", levels=levels, calibration_profile="p").to_payload()["levels"]
            == levels
        )

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"levels": ["a", "b"], "calibration_profile": "p"}, "instructions"),
            ({"instructions": "How?", "levels": ["only"], "calibration_profile": "p"}, "levels"),
            (
                {"instructions": "How?", "levels": [str(i) for i in range(11)], "calibration_profile": "p"},
                "levels",
            ),
            ({"instructions": "How?", "calibration_profile": "p"}, "levels"),
            ({"instructions": "How?", "levels": ["a", "b"]}, "calibration_profile"),
            (
                {"instructions": "How?", "levels": ["a", "b"], "calibration_profile": "p", "alpha": 1.0},
                "alpha",
            ),
            (
                {"instructions": "How?", "levels": ["a", "b"], "calibration_profile": "p", "method": "APS"},
                "method",
            ),
        ],
    )
    def test_validation(self, kwargs, message):
        with pytest.raises(ConfigurationError, match=message):
            Interval(**kwargs)

    def test_accepts_ordinal_aps(self):
        q = Interval(instructions="How?", levels=["a", "b"], calibration_profile="p", method="ordinal-aps")
        assert q.to_payload()["method"] == "ordinal-aps"


# -- Gate -----------------------------------------------------------------


class TestGate:
    def test_fdr_payload(self):
        q = Gate(
            instructions="Auto-route this ticket without human review?",
            calibration_profile=PROFILE,
            guarantee="fdr",
            target=0.05,
        )
        assert q.to_payload() == {
            "type": "gate",
            "instructions": "Auto-route this ticket without human review?",
            "calibration_profile": PROFILE,
            "guarantee": "fdr",
            "target": 0.05,
        }

    def test_default_guarantee_is_risk_and_is_sent(self):
        payload = Gate(instructions="Approve?", calibration_profile="p", target=0.02).to_payload()
        assert payload["guarantee"] == "risk"
        assert "delta" not in payload and "loss" not in payload

    def test_high_probability_payload(self):
        q = Gate(
            instructions="Release the transfer?",
            calibration_profile="high-stakes-transfers",
            guarantee="risk_high_probability",
            target=0.01,
            delta=0.05,
            loss="wrong_approval",
        )
        payload = q.to_payload()
        assert payload["guarantee"] == "risk_high_probability"
        assert payload["target"] == 0.01
        assert payload["delta"] == 0.05
        assert payload["loss"] == "wrong_approval"
        _assert_flat(payload)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"calibration_profile": "p", "target": 0.05}, "instructions"),
            ({"instructions": "Approve?", "target": 0.05}, "calibration_profile"),
            (
                {
                    "instructions": "Approve?",
                    "calibration_profile": "p",
                    "guarantee": "coverage",
                    "target": 0.05,
                },
                "guarantee",
            ),
            ({"instructions": "Approve?", "calibration_profile": "p"}, "target"),
            ({"instructions": "Approve?", "calibration_profile": "p", "target": 1.0}, "target"),
            ({"instructions": "Approve?", "calibration_profile": "p", "target": -0.1}, "target"),
            (
                {
                    "instructions": "Approve?",
                    "calibration_profile": "p",
                    "guarantee": "risk_high_probability",
                    "target": 0.01,
                },
                "delta",
            ),
            (
                {
                    "instructions": "Approve?",
                    "calibration_profile": "p",
                    "guarantee": "risk_high_probability",
                    "target": 0.01,
                    "delta": 1.5,
                },
                "delta",
            ),
        ],
    )
    def test_validation(self, kwargs, message):
        with pytest.raises(ConfigurationError, match=message):
            Gate(**kwargs)


# -- Claim ----------------------------------------------------------------


class TestClaim:
    def test_payload(self):
        q = Claim(
            instructions="Answer from the retrieved passages.",
            calibration_profile="rag-factuality-v1",
            alpha=0.05,
            support_source="retrieved_passages",
        )
        assert q.to_payload() == {
            "type": "claim",
            "instructions": "Answer from the retrieved passages.",
            "calibration_profile": "rag-factuality-v1",
            "alpha": 0.05,
            "support_source": "retrieved_passages",
        }

    def test_minimal_payload(self):
        payload = Claim(instructions="Summarize.", calibration_profile="rag-v1").to_payload()
        assert payload == {"type": "claim", "instructions": "Summarize.", "calibration_profile": "rag-v1"}

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"calibration_profile": "p"}, "instructions"),
            ({"instructions": "Summarize."}, "calibration_profile"),
            ({"instructions": "Summarize.", "calibration_profile": "p", "alpha": 0.0}, "alpha"),
        ],
    )
    def test_validation(self, kwargs, message):
        with pytest.raises(ConfigurationError, match=message):
            Claim(**kwargs)


# -- Judge ----------------------------------------------------------------


class TestJudge:
    CASCADE = [
        {"backend": {"provider": "openai", "model": "gpt-4.1-mini"}},
        {"backend": {"provider": "openai", "model": "gpt-4.1"}},
        {"backend": "human_queue"},
    ]

    def test_payload(self):
        q = Judge(
            instructions="Which response better follows the prompt?",
            calibration_profile="pairwise-judge-v2",
            alpha=0.10,
            cascade=self.CASCADE,
        )
        assert q.to_payload() == {
            "type": "judge",
            "instructions": "Which response better follows the prompt?",
            "calibration_profile": "pairwise-judge-v2",
            "alpha": 0.10,
            "cascade": self.CASCADE,
        }

    def test_backend_objects_in_cascade_serialize(self):
        q = Judge(
            instructions="Which is better?",
            calibration_profile="pairwise-judge-v2",
            cascade=[{"backend": OpenAIBackend(model="gpt-4.1-mini")}, {"backend": "human_queue"}],
        )
        assert q.to_payload()["cascade"] == [
            {"backend": {"provider": "openai", "model": "gpt-4.1-mini", "access_hint": "auto"}},
            {"backend": "human_queue"},
        ]

    def test_omits_unset_cascade(self):
        payload = Judge(instructions="Which is better?", calibration_profile="j").to_payload()
        assert "cascade" not in payload and "alpha" not in payload

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"calibration_profile": "p"}, "instructions"),
            ({"instructions": "Which?"}, "calibration_profile"),
            ({"instructions": "Which?", "calibration_profile": "p", "alpha": 2}, "alpha"),
        ],
    )
    def test_validation(self, kwargs, message):
        with pytest.raises(ConfigurationError, match=message):
            Judge(**kwargs)


# -- Route ----------------------------------------------------------------


class TestRoute:
    CASCADE = [
        {"backend": {"provider": "vllm", "model": "llama-3.3-70b", "base_url": "http://internal-vllm:8000"}},
        {"backend": {"provider": "openai", "model": "gpt-4.1"}},
    ]

    def test_cost_budget_payload(self):
        q = Route(
            cascade=self.CASCADE,
            calibration_profile="cost-routing-v1",
            guarantee="cost_budget",
            target_cents=0.4,
            alpha=0.10,
        )
        assert q.to_payload() == {
            "type": "route",
            "cascade": self.CASCADE,
            "calibration_profile": "cost-routing-v1",
            "guarantee": "cost_budget",
            "target_cents": 0.4,
            "alpha": 0.10,
        }

    def test_accuracy_mode_needs_no_cost_target(self):
        payload = Route(
            cascade=self.CASCADE, calibration_profile="acc-v1", guarantee="accuracy", alpha=0.05
        ).to_payload()
        assert payload["guarantee"] == "accuracy"
        assert "target_cents" not in payload

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"calibration_profile": "p", "target_cents": 0.4}, "cascade"),
            ({"cascade": [], "calibration_profile": "p", "target_cents": 0.4}, "cascade"),
            ({"cascade": CASCADE, "target_cents": 0.4}, "calibration_profile"),
            ({"cascade": CASCADE, "calibration_profile": "p", "guarantee": "latency"}, "guarantee"),
            ({"cascade": CASCADE, "calibration_profile": "p"}, "target_cents"),
            (
                {"cascade": CASCADE, "calibration_profile": "p", "guarantee": "accuracy", "alpha": 1.2},
                "alpha",
            ),
        ],
    )
    def test_validation(self, kwargs, message):
        with pytest.raises(ConfigurationError, match=message):
            Route(**kwargs)


@pytest.mark.parametrize(
    "query",
    [
        Belief(instructions="Urgent?", calibration_profile="p"),
        Set(instructions="Which?", options=SET_OPTIONS, calibration_profile="p"),
        Interval(instructions="How?", levels=["a", "b"], calibration_profile="p"),
        Gate(instructions="Approve?", calibration_profile="p", target=0.05),
        Claim(instructions="Summarize.", calibration_profile="p"),
        Judge(instructions="Which?", calibration_profile="p"),
        Route(
            cascade=[{"backend": {"provider": "openai", "model": "gpt-4.1"}}],
            calibration_profile="p",
            target_cents=1.0,
        ),
    ],
    ids=lambda q: q.type,
)
def test_every_payload_is_flat_and_names_its_profile(query):
    payload = query.to_payload()
    _assert_flat(payload)
    assert payload["type"] == query.type
    assert payload["calibration_profile"] == "p"
