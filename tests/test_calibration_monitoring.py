"""Calibration profiles, drift monitors, audits, and label-efficient helpers.

The hosted resources (``client.calibration_profiles`` and
``client.calibration_profiles.monitors``) are tested sync and async against
``MockAPI``; the offline helpers (``LocalMonitor``, ``audit_coverage``,
``audit_risk``, ``clopper_pearson``, ``judge_quality``,
``estimate_rate_with_judge``) are tested directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from cli_sdk import (
    Alert,
    CalibrationExample,
    CalibrationProfile,
    ConfigurationError,
    LocalMonitor,
    OpenAIBackend,
)
from cli_sdk.calibration import (
    AuditResult,
    GroupStatus,
    audit_coverage,
    audit_risk,
    clopper_pearson,
    estimate_rate_with_judge,
    judge_quality,
    recommended_size,
)
from cli_sdk.monitoring import Monitor
from cli_sdk.stats.evalues.ppi import PPIResult
from support import BASE_URL, PROFILE, MockAPI

OPENAI = {"provider": "openai", "model": "gpt-4.1-2025-04-14"}


def profile_payload(**overrides):
    payload = {
        "name": PROFILE,
        "method": "APS",
        "alpha": 0.10,
        "n": 1204,
        "version": 3,
        "realized_coverage_ci": [0.884, 0.915],
        "backend_fingerprint": {"provider": "openai", "model": "gpt-4.1-2025-04-14"},
        "status": "serving",
    }
    payload.update(overrides)
    return payload


AUDIT = {
    "result": "pass",
    "realized_coverage": 0.903,
    "ci": [0.884, 0.921],
    "sample": 500,
    "target": 0.90,
    "date": "2026-09-18",
}

# -- CalibrationProfile ---------------------------------------------------------


class TestCalibrationProfile:
    def test_from_payload(self):
        p = CalibrationProfile.from_payload(
            profile_payload(
                group_by="plan_tier",
                groups={
                    "enterprise": {"n": 420, "status": "calibrated"},
                    "free": {"n": 12, "status": "underpowered"},
                },
                last_audit={"result": "pass", "date": "2026-09-18"},
            )
        )
        assert p.name == PROFILE and p.method == "APS" and p.alpha == 0.10 and p.n == 1204 and p.version == 3
        assert p.realized_coverage_ci == (0.884, 0.915)
        assert p.backend_fingerprint == {"provider": "openai", "model": "gpt-4.1-2025-04-14"}
        assert p.groups == {
            "enterprise": GroupStatus(n=420, status="calibrated"),
            "free": GroupStatus(n=12, status="underpowered"),
        }
        assert p.group_by == "plan_tier" and p.status == "serving"
        assert p.last_audit == {"result": "pass", "date": "2026-09-18"}
        assert p.can_serve_guarantees

    def test_coverage_ci_is_computed_when_absent(self):
        p = CalibrationProfile.from_payload({"name": "p", "alpha": 0.10, "n": 1000})
        lower, upper = p.realized_coverage_ci
        assert 0.85 < lower < 0.90 < upper < 0.95

    def test_coverage_ci_narrows_as_n_grows(self):
        small = CalibrationProfile.from_payload({"name": "p", "alpha": 0.10, "n": 100}).realized_coverage_ci
        large = CalibrationProfile.from_payload({"name": "p", "alpha": 0.10, "n": 5000}).realized_coverage_ci
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_no_ci_for_an_empty_profile(self):
        p = CalibrationProfile.from_payload({"name": "new-profile", "alpha": 0.05})
        assert p.n == 0 and p.realized_coverage_ci is None
        assert p.status == "collecting" and p.method == "APS" and p.version == 1
        assert not p.can_serve_guarantees

    @pytest.mark.parametrize("alpha, minimum", [(0.10, 9), (0.05, 19), (0.01, 99)])
    def test_can_serve_guarantees_at_the_hard_minimum(self, alpha, minimum):
        below = CalibrationProfile.from_payload({"name": "p", "alpha": alpha, "n": minimum - 1})
        at = CalibrationProfile.from_payload({"name": "p", "alpha": alpha, "n": minimum})
        assert below.minimum_n == minimum
        assert not below.can_serve_guarantees
        assert at.can_serve_guarantees

    def test_server_minimum_wins(self):
        p = CalibrationProfile.from_payload({"name": "p", "alpha": 0.10, "n": 50, "minimum_n": 100})
        assert not p.can_serve_guarantees

    def test_recommended_n_defaults_from_alpha(self):
        assert CalibrationProfile.from_payload({"name": "p", "alpha": 0.05}).recommended_n == 1500
        assert (
            CalibrationProfile.from_payload({"name": "p", "alpha": 0.05, "recommended_n": 2000}).recommended_n
            == 2000
        )


@pytest.mark.parametrize(
    "alpha, expected",
    [
        (0.20, 300),
        (0.30, 300),
        (0.10, 1000),
        (0.15, 1000),
        (0.05, 1500),
        (0.07, 1500),
        (0.01, 2500),
        (0.001, 2500),
    ],
)
def test_recommended_size(alpha, expected):
    assert recommended_size(alpha) == expected


def test_recommended_size_never_decreases_as_alpha_shrinks():
    sizes = [recommended_size(a) for a in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001)]
    assert sizes == sorted(sizes)


class TestCalibrationExample:
    def test_payload(self):
        e = CalibrationExample(
            context={"ticket": "t"}, label="billing", group="enterprise", metadata={"id": 7}
        )
        assert e.to_payload() == {
            "context": {"ticket": "t"},
            "label": "billing",
            "source": "human",
            "group": "enterprise",
            "metadata": {"id": 7},
        }

    def test_minimal_payload(self):
        assert CalibrationExample(context="c", label=1, source="judge").to_payload() == {
            "context": "c",
            "label": 1,
            "source": "judge",
        }

    def test_source_validated(self):
        with pytest.raises(ValueError, match="source"):
            CalibrationExample(context="c", label=1, source="crowd")


# -- hosted calibration profiles (sync) ------------------------------------------------


class TestCalibrationProfilesSync:
    def test_create(self, api: MockAPI, client):
        api.reply(201, profile_payload(n=0, realized_coverage_ci=None, status="collecting"))
        p = client.calibration_profiles.create(
            PROFILE,
            backend=OPENAI,
            method="APS",
            alpha=0.10,
            group_by="plan_tier",
            prompt_template_hash="sha256:ab12",
        )
        assert api.last.method == "POST" and str(api.last.url) == f"{BASE_URL}/calibration-profiles"
        assert api.body() == {
            "name": PROFILE,
            "backend": OPENAI,
            "method": "APS",
            "alpha": 0.10,
            "strict_fingerprint": True,
            "group_by": "plan_tier",
            "prompt_template_hash": "sha256:ab12",
        }
        assert isinstance(p, CalibrationProfile) and p.n == 0 and not p.can_serve_guarantees

    def test_create_minimal_body_with_backend_object(self, api: MockAPI, client):
        api.reply(201, profile_payload(n=0))
        client.calibration_profiles.create(
            "p", backend=OpenAIBackend(model="gpt-4.1"), strict_fingerprint=False
        )
        assert api.body() == {
            "name": "p",
            "backend": {"provider": "openai", "model": "gpt-4.1", "access_hint": "auto"},
            "method": "APS",
            "alpha": 0.10,
            "strict_fingerprint": False,
        }

    @pytest.mark.parametrize("kwargs", [{"method": "magic"}, {"alpha": 0.0}, {"alpha": 1.0}])
    def test_create_validation_never_reaches_the_network(self, api: MockAPI, client, kwargs):
        with pytest.raises(ConfigurationError):
            client.calibration_profiles.create("p", backend=OPENAI, **kwargs)
        assert api.requests == []

    def test_create_requires_a_valid_backend(self, api: MockAPI, client):
        with pytest.raises(ConfigurationError, match="provider"):
            client.calibration_profiles.create("p", backend={"model": "gpt-4.1"})
        assert api.requests == []

    def test_get(self, api: MockAPI, client):
        api.reply(200, profile_payload())
        p = client.calibration_profiles.get(PROFILE)
        assert api.last.method == "GET" and str(api.last.url) == f"{BASE_URL}/calibration-profiles/{PROFILE}"
        assert p.n == 1204 and p.version == 3

    def test_profile_names_are_escaped_in_paths(self, api: MockAPI, client):
        api.reply(200, profile_payload(name="team a/b?x"))
        client.calibration_profiles.get("team a/b?x")
        assert api.last.url.raw_path == b"/api/v1/calibration-profiles/team%20a%2Fb%3Fx"

    def test_list(self, api: MockAPI, client):
        api.reply(
            200, {"profiles": [profile_payload(), profile_payload(name="urgency-v1", method="IVAP", n=850)]}
        )
        profiles = client.calibration_profiles.list()
        assert str(api.last.url) == f"{BASE_URL}/calibration-profiles"
        assert [p.name for p in profiles] == [PROFILE, "urgency-v1"]
        assert profiles[1].method == "IVAP"

    def test_list_empty(self, api: MockAPI, client):
        api.reply(200, {})
        assert client.calibration_profiles.list() == []

    def test_add_examples(self, api: MockAPI, client):
        api.reply(200, profile_payload(n=1206))
        examples = [
            CalibrationExample(context={"ticket": "Charged twice"}, label="billing"),
            {"context": {"ticket": "API is down"}, "label": "technical", "source": "judge"},
            {"context": {"query": "q"}, "tier_outputs": {"cheap": "a", "strong": "b"}, "label": "b"},
        ]
        p = client.calibration_profiles.add_examples(PROFILE, examples)
        assert api.last.method == "POST"
        assert str(api.last.url) == f"{BASE_URL}/calibration-profiles/{PROFILE}/examples"
        assert api.body() == {
            "examples": [
                {"context": {"ticket": "Charged twice"}, "label": "billing", "source": "human"},
                {"context": {"ticket": "API is down"}, "label": "technical", "source": "judge"},
                {"context": {"query": "q"}, "tier_outputs": {"cheap": "a", "strong": "b"}, "label": "b"},
            ]
        }
        assert p.n == 1206

    def test_add_route_examples_with_tier_outputs(self, api: MockAPI, client):
        api.reply(200, profile_payload(name="cost-routing-v1", n=1))
        example = {"context": {"query": "q"}, "tier_outputs": {"cheap": {"output": "a", "cost_cents": 0.02}}}
        client.calibration_profiles.add_examples("cost-routing-v1", [example])
        assert api.body() == {"examples": [example]}

    @pytest.mark.parametrize("bad", [{"label": "billing"}, {"context": "c"}, "just a string", 42])
    def test_add_examples_rejects_malformed_examples(self, api: MockAPI, client, bad):
        with pytest.raises(ValueError, match="example 1"):
            client.calibration_profiles.add_examples(PROFILE, [{"context": "ok", "label": "x"}, bad])
        assert api.requests == []

    def test_label_with_judge(self, api: MockAPI, client):
        job = {
            "job_id": "job_1",
            "human_tasks": [{"id": 1}],
            "judge_quality": {"agreement_with_humans": 0.87},
        }
        api.reply(202, job)
        result = client.calibration_profiles.label_with_judge(
            PROFILE,
            judge=OpenAIBackend(model="gpt-4.1"),
            unlabelled_examples=[{"ticket": "a"}, {"context": {"ticket": "b"}}, "plain text"],
            human_labelled_sample_size=150,
        )
        assert str(api.last.url) == f"{BASE_URL}/calibration-profiles/{PROFILE}/label-with-judge"
        assert api.body() == {
            "judge": {"provider": "openai", "model": "gpt-4.1", "access_hint": "auto"},
            "unlabelled_examples": [
                {"context": {"ticket": "a"}},
                {"context": {"ticket": "b"}},
                {"context": "plain text"},
            ],
            "human_labelled_sample_size": 150,
        }
        assert result == job

    def test_label_with_judge_defaults_and_validation(self, api: MockAPI, client):
        api.reply(202, {"job_id": "job_2"})
        client.calibration_profiles.label_with_judge(PROFILE, judge=OPENAI, unlabelled_examples=["x"])
        assert api.body()["human_labelled_sample_size"] == 300
        with pytest.raises(ConfigurationError, match="at least 2"):
            client.calibration_profiles.label_with_judge(
                PROFILE, judge=OPENAI, unlabelled_examples=["x"], human_labelled_sample_size=1
            )
        assert len(api.requests) == 1

    def test_audit(self, api: MockAPI, client):
        api.reply(200, AUDIT)
        result = client.calibration_profiles.audit(
            PROFILE, [CalibrationExample(context="c", label="billing")]
        )
        assert str(api.last.url) == f"{BASE_URL}/calibration-profiles/{PROFILE}/audit"
        assert api.body() == {"examples": [{"context": "c", "label": "billing", "source": "human"}]}
        assert result == AuditResult(
            result="pass",
            realized_coverage=0.903,
            ci_lower=0.884,
            ci_upper=0.921,
            sample=500,
            target=0.90,
            date="2026-09-18",
        )
        assert result.passed

    def test_resource_errors_are_mapped(self, api: MockAPI, client):
        from cli_sdk import AuthenticationError, CLIError

        api.reply(401, {"message": "invalid key"})
        with pytest.raises(AuthenticationError):
            client.calibration_profiles.get(PROFILE)
        api.reply(404, {"message": "no such profile"})
        with pytest.raises(CLIError, match="no such profile"):
            client.calibration_profiles.get("missing")


class TestAuditResult:
    def test_alternate_ci_key(self):
        r = AuditResult.from_payload(
            {
                "result": "fail",
                "realized_coverage": 0.8,
                "realized_coverage_ci": [0.75, 0.85],
                "sample": 300,
                "target": 0.9,
            }
        )
        assert (r.ci_lower, r.ci_upper) == (0.75, 0.85) and not r.passed

    def test_defaults_fail_closed(self):
        r = AuditResult.from_payload({})
        assert r.result == "fail" and not r.passed and (r.ci_lower, r.ci_upper) == (0.0, 1.0)


# -- hosted monitors (sync) ------------------------------------------------------------


def monitor_payload(mid="mon_1", **overrides):
    payload = {
        "id": mid,
        "profile": PROFILE,
        "type": "coverage",
        "target": 0.90,
        "false_alarm_rate": 0.05,
        "status": "active",
    }
    payload.update(overrides)
    return payload


class TestMonitorsSync:
    def test_create(self, api: MockAPI, client):
        api.reply(201, monitor_payload(labelled_sample_rate=0.02))
        m = client.calibration_profiles.monitors.create(
            PROFILE, type="coverage", target=0.90, false_alarm_rate=0.05, labelled_sample_rate=0.02
        )
        assert api.last.method == "POST"
        assert str(api.last.url) == f"{BASE_URL}/calibration-profiles/{PROFILE}/monitors"
        assert api.body() == {
            "type": "coverage",
            "false_alarm_rate": 0.05,
            "target": 0.90,
            "labelled_sample_rate": 0.02,
        }
        assert m == Monitor(
            id="mon_1",
            profile=PROFILE,
            type="coverage",
            target=0.90,
            false_alarm_rate=0.05,
            labelled_sample_rate=0.02,
            status="active",
        )

    def test_create_risk_with_judge_pseudo_labels(self, api: MockAPI, client):
        api.reply(201, monitor_payload(type="risk", target=0.02))
        client.calibration_profiles.monitors.create(
            PROFILE, type="risk", target=0.02, use_judge_pseudo_labels=True
        )
        assert api.body() == {
            "type": "risk",
            "false_alarm_rate": 0.05,
            "target": 0.02,
            "use_judge_pseudo_labels": True,
        }

    def test_fingerprint_monitor_needs_no_target(self, api: MockAPI, client):
        api.reply(201, monitor_payload(type="fingerprint", target=None))
        m = client.calibration_profiles.monitors.create(PROFILE, type="fingerprint")
        assert api.body() == {"type": "fingerprint", "false_alarm_rate": 0.05}
        assert m.type == "fingerprint" and m.target is None

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"type": "latency", "target": 0.9}, "type"),
            ({"type": "coverage"}, "target"),
            ({"type": "risk"}, "target"),
            ({"type": "coverage", "target": 0.9, "false_alarm_rate": 0.0}, "false_alarm_rate"),
            ({"type": "coverage", "target": 0.9, "false_alarm_rate": 1.0}, "false_alarm_rate"),
        ],
    )
    def test_create_validation(self, api: MockAPI, client, kwargs, message):
        with pytest.raises(ConfigurationError, match=message):
            client.calibration_profiles.monitors.create(PROFILE, **kwargs)
        assert api.requests == []

    def test_list(self, api: MockAPI, client):
        api.reply(
            200, {"monitors": [monitor_payload("mon_1"), monitor_payload("mon_2", type="risk", target=0.02)]}
        )
        monitors = client.calibration_profiles.monitors.list(PROFILE)
        assert api.last.method == "GET"
        assert str(api.last.url) == f"{BASE_URL}/calibration-profiles/{PROFILE}/monitors"
        assert [m.id for m in monitors] == ["mon_1", "mon_2"]
        assert monitors[1].type == "risk"

    def test_poll_all_monitors(self, api: MockAPI, client):
        api.route(
            "GET",
            f"/api/v1/calibration-profiles/{PROFILE}/monitors",
            {"monitors": [monitor_payload("mon_1"), monitor_payload("mon_2")]},
        )
        api.route(
            "GET",
            f"/api/v1/calibration-profiles/{PROFILE}/monitors/mon_1/alerts",
            {
                "alerts": [
                    {
                        "type": "coverage",
                        "n": 812,
                        "e_value": 21.4,
                        "target": 0.9,
                        "false_alarm_rate": 0.05,
                        "detected_at": "2026-09-20T10:00:00Z",
                        "window": "7d",
                    }
                ]
            },
        )
        api.route("GET", f"/api/v1/calibration-profiles/{PROFILE}/monitors/mon_2/alerts", {"alerts": []})
        alerts = client.calibration_profiles.monitors.poll(PROFILE)
        assert api.requests == []  # sync poll is a lazy generator
        alerts = list(alerts)
        assert [r.url.path for r in api.requests] == [
            f"/api/v1/calibration-profiles/{PROFILE}/monitors",
            f"/api/v1/calibration-profiles/{PROFILE}/monitors/mon_1/alerts",
            f"/api/v1/calibration-profiles/{PROFILE}/monitors/mon_2/alerts",
        ]
        assert alerts == [
            Alert(
                type="coverage",
                severity="critical",
                profile=PROFILE,
                n=812,
                e_value=21.4,
                target=0.9,
                false_alarm_rate=0.05,
                detected_at="2026-09-20T10:00:00Z",
                details={"window": "7d"},
            )
        ]

    def test_poll_one_monitor_skips_the_listing(self, api: MockAPI, client):
        api.reply(
            200,
            {"alerts": [{"type": "fingerprint", "severity": "warning", "message": "model version changed"}]},
        )
        alerts = list(client.calibration_profiles.monitors.poll(PROFILE, monitor_id="mon_9"))
        assert len(api.requests) == 1
        assert api.last.url.path == f"/api/v1/calibration-profiles/{PROFILE}/monitors/mon_9/alerts"
        assert (
            alerts[0].type == "fingerprint"
            and alerts[0].severity == "warning"
            and alerts[0].profile == PROFILE
        )


# -- hosted resources (async) -----------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncResources:
    async def test_profile_lifecycle(self, api: MockAPI, make_async_client):
        client = make_async_client()
        api.reply(201, profile_payload(n=0))
        api.reply(200, profile_payload())
        api.reply(200, {"profiles": [profile_payload()]})
        api.reply(200, profile_payload(n=1205))
        api.reply(202, {"job_id": "job_3"})
        api.reply(200, AUDIT)

        created = await client.calibration_profiles.create(PROFILE, backend=OPENAI, alpha=0.05)
        fetched = await client.calibration_profiles.get(PROFILE)
        listed = await client.calibration_profiles.list()
        added = await client.calibration_profiles.add_examples(
            PROFILE, [{"context": "c", "label": "billing"}]
        )
        job = await client.calibration_profiles.label_with_judge(
            PROFILE, judge=OPENAI, unlabelled_examples=["x", "y"]
        )
        audit = await client.calibration_profiles.audit(PROFILE, [{"context": "c", "label": "billing"}])

        assert [(r.method, r.url.path) for r in api.requests] == [
            ("POST", "/api/v1/calibration-profiles"),
            ("GET", f"/api/v1/calibration-profiles/{PROFILE}"),
            ("GET", "/api/v1/calibration-profiles"),
            ("POST", f"/api/v1/calibration-profiles/{PROFILE}/examples"),
            ("POST", f"/api/v1/calibration-profiles/{PROFILE}/label-with-judge"),
            ("POST", f"/api/v1/calibration-profiles/{PROFILE}/audit"),
        ]
        assert api.body(0)["alpha"] == 0.05
        assert created.n == 0 and fetched.n == 1204 and listed[0].name == PROFILE and added.n == 1205
        assert job == {"job_id": "job_3"}
        assert audit.passed

    async def test_create_validation(self, api: MockAPI, make_async_client):
        with pytest.raises(ConfigurationError):
            await make_async_client().calibration_profiles.create("p", backend=OPENAI, method="nope")
        assert api.requests == []

    async def test_monitors(self, api: MockAPI, make_async_client):
        client = make_async_client()
        monitors = client.calibration_profiles.monitors
        api.route("POST", f"/api/v1/calibration-profiles/{PROFILE}/monitors", monitor_payload("mon_1"))
        api.route(
            "GET", f"/api/v1/calibration-profiles/{PROFILE}/monitors", {"monitors": [monitor_payload("mon_1")]}
        )
        api.route(
            "GET",
            f"/api/v1/calibration-profiles/{PROFILE}/monitors/mon_1/alerts",
            {"alerts": [{"type": "coverage", "e_value": 25.0}]},
        )

        created = await monitors.create(PROFILE, target=0.9)
        listed = await monitors.list(PROFILE)
        alerts = await monitors.poll(PROFILE)
        single = await monitors.poll(PROFILE, monitor_id="mon_1")

        assert created.id == "mon_1" and [m.id for m in listed] == ["mon_1"]
        assert isinstance(alerts, list) and alerts == single
        assert alerts[0].profile == PROFILE and alerts[0].e_value == 25.0
        assert api.body(0) == {"type": "coverage", "false_alarm_rate": 0.05, "target": 0.9}


# -- LocalMonitor --------------------------------------------------------------------


class TestLocalMonitor:
    def test_coverage_monitor_fires_once_on_a_clearly_bad_stream(self):
        monitor = LocalMonitor(type="coverage", target=0.90, false_alarm_rate=0.05, profile=PROFILE)
        alerts = [monitor.update(i % 2 == 0) for i in range(400)]  # 50% realized coverage
        fired = [a for a in alerts if a is not None]
        assert len(fired) == 1
        alert = fired[0]
        assert isinstance(alert, Alert)
        assert alert.type == "coverage" and alert.profile == PROFILE
        assert alert.target == 0.90 and alert.false_alarm_rate == 0.05
        assert alert.e_value >= 1 / 0.05
        assert alert.n == alerts.index(alert) + 1
        assert alert.n < 100  # a drop this large is detected quickly

    def test_coverage_monitor_stays_quiet_on_a_healthy_stream(self):
        monitor = LocalMonitor(type="coverage", target=0.90)
        # 95% coverage, deterministic: comfortably above target, never an alarm
        assert all(monitor.update(i % 20 != 0) is None for i in range(5000))

    def test_coverage_monitor_stays_quiet_exactly_at_target(self):
        monitor = LocalMonitor(type="coverage", target=0.90)
        assert all(monitor.update(i % 10 != 0) is None for i in range(5000))

    def test_risk_monitor_fires_once(self):
        monitor = LocalMonitor(
            type="risk", target=0.05, false_alarm_rate=0.01, profile="high-stakes-transfers"
        )
        rng = np.random.default_rng(7)
        fired = [a for a in (monitor.update(bool(loss)) for loss in rng.random(2000) < 0.30) if a is not None]
        assert len(fired) == 1
        assert fired[0].type == "risk" and fired[0].target == 0.05 and fired[0].e_value >= 100

    def test_risk_monitor_quiet_when_losses_are_rare(self):
        monitor = LocalMonitor(type="risk", target=0.05)
        rng = np.random.default_rng(11)
        assert all(monitor.update(bool(loss)) is None for loss in rng.random(3000) < 0.01)

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="coverage"):
            LocalMonitor(type="fingerprint", target=0.9)

    def test_false_alarm_rate_validated(self):
        with pytest.raises(ValueError):
            LocalMonitor(type="coverage", target=0.9, false_alarm_rate=0.0)


class TestAlert:
    def test_from_payload_keeps_unknown_fields(self):
        a = Alert.from_payload({"type": "risk", "profile": "p", "segment": "enterprise"})
        assert a.type == "risk" and a.severity == "critical" and a.details == {"segment": "enterprise"}


# -- local audits ---------------------------------------------------------------------


class TestClopperPearson:
    @pytest.mark.parametrize(
        "successes, trials, expected",
        [
            (5, 10, (0.187086, 0.812914)),
            (45, 50, (0.781865, 0.966725)),
            (90, 100, (0.823777, 0.950995)),
            (950, 1000, (0.934610, 0.962665)),
        ],
    )
    def test_close_to_reference_values(self, successes, trials, expected):
        lower, upper = clopper_pearson(successes, trials)
        assert lower == pytest.approx(expected[0], abs=1e-3)
        assert upper == pytest.approx(expected[1], abs=1e-3)

    @pytest.mark.parametrize(
        "successes, trials, expected",
        [(45, 50, (0.7818646, 0.9667249)), (90, 100, (0.8237774, 0.9509953))],
    )
    def test_matches_reference_values_exactly(self, successes, trials, expected):
        lower, upper = clopper_pearson(successes, trials)
        assert lower == pytest.approx(expected[0], abs=1e-6)
        assert upper == pytest.approx(expected[1], abs=1e-6)

    def test_edges(self):
        assert clopper_pearson(0, 20)[0] == 0.0
        assert clopper_pearson(20, 20)[1] == 1.0
        assert clopper_pearson(0, 10)[1] == pytest.approx(0.308497, abs=1e-3)

    def test_interval_contains_the_point_estimate_and_narrows_with_n(self):
        widths = []
        for trials in (20, 200, 2000):
            lower, upper = clopper_pearson(int(0.9 * trials), trials)
            assert 0.0 <= lower < 0.9 < upper <= 1.0
            widths.append(upper - lower)
        assert widths == sorted(widths, reverse=True)

    def test_higher_confidence_is_wider(self):
        lo95, hi95 = clopper_pearson(90, 100, confidence=0.95)
        lo99, hi99 = clopper_pearson(90, 100, confidence=0.99)
        assert lo99 < lo95 and hi99 > hi95

    def test_trials_must_be_positive(self):
        with pytest.raises(ValueError):
            clopper_pearson(0, 0)


class TestLocalAudits:
    def test_coverage_pass(self):
        result = audit_coverage([True] * 950 + [False] * 50, target=0.90)
        assert result.passed and result.result == "pass"
        assert result.realized_coverage == 0.95 and result.sample == 1000 and result.target == 0.90
        assert result.ci_lower < 0.95 < result.ci_upper

    def test_coverage_slightly_under_target_by_chance_still_passes(self):
        result = audit_coverage([True] * 88 + [False] * 12, target=0.90)
        assert result.realized_coverage == 0.88
        assert result.ci_upper > 0.90 and result.passed

    def test_coverage_fail(self):
        result = audit_coverage([True] * 800 + [False] * 200, target=0.90)
        assert not result.passed and result.ci_upper < 0.90

    def test_coverage_accepts_truthy_values(self):
        assert audit_coverage([1, 0, 1, 1], target=0.5).realized_coverage == 0.75

    def test_coverage_empty(self):
        with pytest.raises(ValueError):
            audit_coverage([], target=0.9)

    def test_risk_pass(self):
        result = audit_risk([1] * 10 + [0] * 990, target=0.05)
        assert result.passed
        assert result.realized_coverage == pytest.approx(0.99)
        assert result.target == pytest.approx(0.95)
        assert result.ci_lower < 0.99 < result.ci_upper

    def test_risk_fail(self):
        result = audit_risk([1.0] * 200 + [0.0] * 800, target=0.05)
        assert not result.passed and result.ci_upper < 0.95


# -- label-efficient calibration ---------------------------------------------------------


class TestLabelEfficient:
    def test_perfect_judge(self):
        human = [1, 0, 1, 1, 0, 1, 0, 0]
        q = judge_quality(human, human)
        assert q.agreement_with_humans == 1.0
        assert q.correlation == pytest.approx(1.0)
        assert q.effective_label_multiplier > 1e6

    def test_uninformative_judge(self):
        rng = np.random.default_rng(3)
        human = rng.integers(0, 2, 5000)
        judge = rng.integers(0, 2, 5000)
        q = judge_quality(human, judge)
        assert q.agreement_with_humans == pytest.approx(0.5, abs=0.03)
        assert abs(q.correlation) < 0.05
        assert q.effective_label_multiplier == pytest.approx(1.0, abs=0.01)

    def test_constant_judge_has_no_correlation(self):
        q = judge_quality([1, 0, 1, 0], [1, 1, 1, 1])
        assert q.correlation == 0.0 and q.effective_label_multiplier == 1.0 and q.agreement_with_humans == 0.5

    def test_empty(self):
        q = judge_quality([], [])
        assert q.agreement_with_humans == 0.0 and q.correlation == 0.0

    def test_estimate_rate_with_a_good_judge(self):
        # A property over many draws, not one lucky (or unlucky) seed: the
        # 95% interval covers the true rate about 95% of the time, and a
        # judge that agrees with humans ~95% of the time narrows it.
        truth, pool_size, n_human, trials = 0.8, 5000, 200, 300
        covered, width_ratios = [], []
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            labels = (rng.random(pool_size + n_human) < truth).astype(float)
            judge = np.where(rng.random(labels.size) < 0.05, 1 - labels, labels)
            human, judge_on_human = labels[:n_human], judge[:n_human]
            result = estimate_rate_with_judge(human, judge_on_human, judge[n_human:], alpha=0.05)
            assert isinstance(result, PPIResult) and 0.0 <= result.lam <= 1.0
            covered.append(result.ci_lower < truth < result.ci_upper)
            human_only_half_width = 1.96 * np.std(human, ddof=1) / np.sqrt(n_human)
            width_ratios.append((result.ci_upper - result.ci_lower) / 2 / human_only_half_width)
        assert np.mean(covered) >= 0.90
        assert np.median(width_ratios) < 0.8

    def test_misaligned_labels(self):
        with pytest.raises(ValueError, match="aligned"):
            estimate_rate_with_judge([1, 0, 1], [1, 0], [1, 1, 0, 0])

    def test_needs_two_human_labels(self):
        with pytest.raises(ValueError):
            estimate_rate_with_judge([1], [1], [1, 0, 1])
