"""An offline stand-in for the hosted CLI API, used by every hosted example.

``MockCLIServer`` answers the same endpoints as ``https://cci.gitdate.ink/api/v1``
through ``httpx.MockTransport``, so an example runs end to end with no
network access and no API key:

    POST /evaluate                                   typed answers + guarantee cards
    GET  /backends                                   configured backends
    GET  /calibration-profiles                       list profiles
    POST /calibration-profiles                       create a profile
    GET  /calibration-profiles/{name}                read a profile
    POST /calibration-profiles/{name}/examples       add labelled examples
    POST /calibration-profiles/{name}/label-with-judge
    POST /calibration-profiles/{name}/audit          audit on fresh labelled examples
    POST /calibration-profiles/{name}/monitors       create a drift monitor
    GET  /calibration-profiles/{name}/monitors       list monitors
    GET  /calibration-profiles/{name}/monitors/{id}/alerts

The answers are canned but not random: each one is computed from the
request (the query ids and types, the options, the context text, the
evidence a ``CustomBackend`` sent) with small deterministic scoring rules,
and every answer carries a guarantee card built from the named profile's
state (method, alpha/target, ``calibration_n``, the Beta coverage interval
for that n, the last audit date). A profile below its minimum size gets a
``heuristic`` card and a 424 status, exactly as the hosted API fails closed.

Nothing here is statistics you should rely on; it exists so the examples
show real request and response shapes offline. The offline example
(``offline_calibration_no_network.py``) runs the real engine instead.

Usage from an example::

    from _mock import open_client, parse_args

    args = parse_args(__doc__)
    with open_client(args.server, backend=...) as client:
        ...
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import itertools
import json
import math
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

import httpx

from cli_sdk import CLIClient
from cli_sdk.calibration.audit import audit_coverage
from cli_sdk.calibration.profile import recommended_size
from cli_sdk.constants import DEFAULT_BASE_URL, ENV_API_KEY
from cli_sdk.stats.conformal import coverage_confidence_interval, minimum_calibration_size

MOCK_API_KEY = "sk_test_mock_offline"
STATS_VERSION = "cli-stats 0.1.0"

# Word-prefix keyword weights for the three support teams used throughout the
# examples. A token matches a keyword when it starts with it ("refunded"
# matches "refund").
_TEAM_KEYWORDS: dict[str, dict[str, float]] = {
    "billing": {
        "charge": 1.0, "refund": 1.0, "invoice": 1.0, "payment": 0.8, "payout": 1.0,
        "billing": 1.0, "billed": 1.0, "vat": 0.8, "receipt": 0.8, "card": 0.6,
        "renewal": 0.8, "paid": 0.6, "stripe": 0.5, "subscription": 0.5, "money": 0.6,
    },
    "technical": {
        "error": 1.0, "bug": 1.0, "broken": 1.0, "crash": 1.0, "api": 1.0,
        "webhook": 1.0, "integrat": 1.0, "connect": 0.9, "login": 0.8, "log in": 0.8,
        "500": 1.0, "timeout": 1.0, "time out": 1.0, "outage": 1.0, "sync": 0.8,
        "stripe": 0.6, "down": 0.6, "failing": 0.6, "export": 0.6, "sso": 0.9,
    },
    "sales": {
        "pricing": 1.0, "price": 0.9, "cost": 0.8, "seats": 1.0, "upgrade": 0.9,
        "plan": 0.5, "enterprise": 0.7, "discount": 1.0, "quote": 1.0, "demo": 1.0,
        "trial": 0.8, "contract": 0.8, "nonprofit": 0.9, "annual": 0.4,
    },
}

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me",
    "my", "not", "of", "on", "or", "our", "so", "that", "the", "their", "them", "then",
    "there", "these", "this", "to", "up", "was", "we", "what", "when", "which", "who",
    "will", "with", "you", "your",
})

# Detected access level per provider, as the hosted adapters report it.
_ACCESS_LEVELS = {
    "openai": "L1", "azure-openai": "L1", "anthropic": "L0", "gemini": "L0",
    "bedrock": "L0", "openrouter": "L0", "vllm": "L3", "sglang": "L3",
}


# -- small deterministic helpers -------------------------------------------


def _stable_unit(*parts: Any) -> float:
    """A deterministic pseudo-random number in [0, 1) derived from ``parts``."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(v) for v in value)
    return "" if value is None else str(value)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _content_words(text: str) -> set[str]:
    return {t for t in _tokens(text) if len(t) > 2 and t not in _STOPWORDS}


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    top = max(logits.values())
    exp = {k: math.exp(v - top) for k, v in logits.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _venn_abers_interval(p: float, n: int) -> list[float]:
    """A plausible [p0, p1]: wider for mid-range p and for small calibration sets."""
    width = 0.03 + (1.6 / math.sqrt(max(n, 1))) * 4 * p * (1 - p)
    return [round(max(0.0, p - 0.45 * width), 3), round(min(1.0, p + 0.55 * width), 3)]


def _today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).date().isoformat()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# -- the mock server ---------------------------------------------------------


class MockCLIServer:
    """In-memory, deterministic imitation of the hosted CLI API.

    ``requests`` records every request the transport received, in order, as
    ``{"method", "path", "body"}`` so an example can show exactly what the
    SDK put on the wire. ``simulate_drift=True`` makes the monitor alert
    endpoints return a coverage and a fingerprint alert.
    """

    def __init__(self, simulate_drift: bool = False, seed_profiles: bool = True) -> None:
        self.simulate_drift = simulate_drift
        self.profiles: dict[str, dict[str, Any]] = {}
        self.monitors: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self._request_ids = itertools.count(1)
        self._monitor_ids = itertools.count(1)
        if seed_profiles:
            self._seed_profiles()

    # -- wiring --------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1")
        body = json.loads(request.content) if request.content else None
        self.requests.append({"method": request.method, "path": path, "body": body})
        headers = {"x-request-id": f"req_mock_{next(self._request_ids):06d}"}

        if not request.headers.get("authorization", "").startswith("Bearer "):
            return httpx.Response(401, json={"message": "missing or invalid API key"}, headers=headers)

        status, payload = self._route(request.method, path, body or {})
        return httpx.Response(status, json=payload, headers=headers)

    def _route(self, method: str, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        parts = [p for p in path.split("/") if p]
        if method == "POST" and parts == ["evaluate"]:
            return self._evaluate(body)
        if method == "GET" and parts == ["backends"]:
            return 200, {"backends": [
                {"provider": provider, "connection": "default", "access_level": level}
                for provider, level in _ACCESS_LEVELS.items()
            ]}
        if parts[:1] != ["calibration-profiles"]:
            return 404, {"message": f"no route for {method} {path}"}

        if len(parts) == 1:
            if method == "GET":
                return 200, {"profiles": [self._profile_payload(p) for p in self.profiles.values()]}
            if method == "POST":
                return self._create_profile(body)
        name = parts[1]
        profile = self.profiles.get(name)
        if profile is None:
            return 404, {"message": f"calibration profile {name!r} not found"}
        rest = parts[2:]
        if method == "GET" and not rest:
            return 200, self._profile_payload(profile)
        if method == "POST" and rest == ["examples"]:
            return self._add_examples(profile, body)
        if method == "POST" and rest == ["label-with-judge"]:
            return self._label_with_judge(profile, body)
        if method == "POST" and rest == ["audit"]:
            return self._audit(profile, body)
        if rest[:1] == ["monitors"]:
            return self._monitors(profile, method, rest[1:], body)
        return 404, {"message": f"no route for {method} {path}"}

    # -- profiles ------------------------------------------------------

    def _new_profile(self, name: str, backend: dict[str, Any], method: str, alpha: float,
                     group_by: Optional[str] = None, prompt_template_hash: Optional[str] = None,
                     n: int = 0, last_audit: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        fingerprint = {k: backend[k] for k in ("provider", "model", "deployment", "base_url",
                                                "engine_version", "quantization", "name") if backend.get(k)}
        if prompt_template_hash:
            fingerprint["prompt_template_hash"] = prompt_template_hash
        return {
            "name": name, "method": method, "alpha": alpha, "n": n, "version": 1,
            "backend_fingerprint": fingerprint, "group_by": group_by,
            "group_counts": {}, "labels": [], "last_audit": last_audit,
        }

    def _seed_profiles(self) -> None:
        """Profiles the non-onboarding examples expect to exist already."""
        seeds = [
            ("rag-factuality-v1", {"provider": "anthropic", "model": "claude-sonnet-5"},
             "conformal-factuality", 0.05, 140, "2026-09-12", 60),
            ("pairwise-judge-v2", {"provider": "openai", "model": "gpt-4.1-mini-2025-04-14"},
             "trust-or-escalate", 0.10, 1800, "2026-09-15", 300),
            ("cost-routing-v1", {"provider": "vllm", "model": "llama-3.3-70b"},
             "calibrated-cascade", 0.10, 900, "2026-09-10", 200),
            ("toy-intent-v1", {"provider": "client_evidence", "name": "toy-intent-bow"},
             "APS", 0.10, 640, "2026-09-20", 200),
        ]
        for name, backend, method, alpha, n, audit_date, audit_n in seeds:
            self.profiles[name] = self._new_profile(
                name, backend, method, alpha, n=n,
                last_audit={"date": audit_date, "sample": audit_n, "result": "pass"},
            )

    def _profile_payload(self, profile: dict[str, Any]) -> dict[str, Any]:
        n, alpha = profile["n"], profile["alpha"]
        minimum = minimum_calibration_size(alpha)
        payload = {
            "name": profile["name"], "method": profile["method"], "alpha": alpha,
            "n": n, "version": profile["version"],
            "recommended_n": recommended_size(alpha), "minimum_n": minimum,
            "backend_fingerprint": profile["backend_fingerprint"],
            "status": "serving" if n >= minimum else "collecting",
            "last_audit": profile["last_audit"],
        }
        if n > 0:
            payload["realized_coverage_ci"] = [round(v, 3) for v in coverage_confidence_interval(n, alpha)]
        if profile["group_by"]:
            payload["group_by"] = profile["group_by"]
            payload["groups"] = {
                g: {"n": c, "status": "calibrated" if c >= minimum else "underpowered"}
                for g, c in sorted(profile["group_counts"].items())
            }
        return payload

    def _create_profile(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        name = body.get("name")
        if not name:
            return 422, {"message": "name is required", "field": "name"}
        if name in self.profiles:
            return 409, {"message": f"calibration profile {name!r} already exists"}
        profile = self._new_profile(name, body.get("backend") or {}, body.get("method", "APS"),
                                    float(body.get("alpha", 0.1)), body.get("group_by"),
                                    body.get("prompt_template_hash"))
        self.profiles[name] = profile
        return 200, self._profile_payload(profile)

    def _add_examples(self, profile: dict[str, Any], body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        examples = body.get("examples") or []
        for example in examples:
            label = example.get("label")
            if isinstance(label, str) and label not in profile["labels"]:
                profile["labels"].append(label)
            group_by = profile["group_by"]
            if group_by:
                group = example.get("group") or (example.get("context") or {}).get(group_by, "unknown")
                profile["group_counts"][group] = profile["group_counts"].get(group, 0) + 1
        profile["n"] += len(examples)
        return 200, self._profile_payload(profile)

    def _label_with_judge(self, profile: dict[str, Any], body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        pool = body.get("unlabelled_examples") or []
        k = min(int(body.get("human_labelled_sample_size", 300)), len(pool))
        step = max(1, len(pool) // max(k, 1))
        return 200, {
            "job_id": f"job_mock_{profile['name']}",
            "status": "awaiting_human_labels",
            "judge_labelled": len(pool),
            "human_labelling_tasks": [pool[i]["context"] for i in range(0, len(pool), step)][:k],
            "judge_quality": {"agreement_with_humans": None, "note": "computed once human labels arrive"},
        }

    def _audit(self, profile: dict[str, Any], body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        examples = body.get("examples") or []
        if not examples:
            return 422, {"message": "audit requires at least one example", "field": "examples"}
        target = 1 - profile["alpha"]
        options = profile["labels"] or list(_TEAM_KEYWORDS)
        covered = []
        for example in examples:
            probs = self._option_probabilities(example.get("context"), options, None)
            prediction_set = self._aps_set(probs, profile["alpha"])
            covered.append(example.get("label") in prediction_set)
        result = audit_coverage(covered, target=target)
        profile["last_audit"] = {"date": _today(), "sample": result.sample, "result": result.result}
        return 200, {
            "result": result.result,
            "realized_coverage": round(result.realized_coverage, 4),
            "ci": [round(result.ci_lower, 4), round(result.ci_upper, 4)],
            "sample": result.sample,
            "target": target,
            "date": _today(),
        }

    def _monitors(self, profile: dict[str, Any], method: str, rest: list[str],
                  body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        name = profile["name"]
        if method == "POST" and not rest:
            monitor = {
                "id": f"mon_{next(self._monitor_ids):04d}", "profile": name,
                "type": body.get("type", "coverage"), "target": body.get("target"),
                "false_alarm_rate": body.get("false_alarm_rate"),
                "labelled_sample_rate": body.get("labelled_sample_rate"),
                "status": "active", "created_at": _now(),
            }
            self.monitors[monitor["id"]] = monitor
            return 200, monitor
        if method == "GET" and not rest:
            return 200, {"monitors": [m for m in self.monitors.values() if m["profile"] == name]}
        if method == "GET" and len(rest) == 2 and rest[1] == "alerts":
            monitor = self.monitors.get(rest[0])
            if monitor is None or monitor["profile"] != name:
                return 404, {"message": f"monitor {rest[0]!r} not found on {name!r}"}
            return 200, {"alerts": self._alerts_for(monitor)}
        return 404, {"message": "no such monitor route"}

    def _alerts_for(self, monitor: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.simulate_drift:
            return []
        far = monitor.get("false_alarm_rate") or 0.05
        if monitor["type"] == "coverage":
            return [{
                "type": "coverage", "severity": "critical", "n": 412, "e_value": 23.8,
                "target": monitor.get("target"), "false_alarm_rate": far, "detected_at": _now(),
                "message": (f"running coverage 0.83 on 412 labelled production tickets; "
                            f"e-value 23.8 crossed 1/{far:g} = {1 / far:g}"),
                "realized_coverage": 0.83,
            }]
        if monitor["type"] == "fingerprint":
            fingerprint = self.profiles[monitor["profile"]]["backend_fingerprint"]
            return [{
                "type": "fingerprint", "severity": "warning", "detected_at": _now(),
                "message": "backend reported a model version that differs from the calibration fingerprint",
                "expected": fingerprint.get("model"), "observed": f"{fingerprint.get('model')}-2026-09-02",
            }]
        return []

    # -- evaluate ------------------------------------------------------

    def _evaluate(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        context = body.get("context")
        backend = body.get("backend") or {}
        queries = body.get("queries") or {}
        if not queries:
            return 422, {"message": "queries must not be empty", "field": "queries"}

        answers: dict[str, Any] = {}
        warnings: list[str] = []
        calls = 0
        insufficient = False
        for qid, query in queries.items():
            name = query.get("calibration_profile")
            profile = self.profiles.get(name)
            if profile is None:
                return 422, {"message": f"query {qid!r} names unknown calibration_profile {name!r}",
                             "field": f"queries.{qid}.calibration_profile"}
            builder = getattr(self, f"_answer_{query.get('type')}", None)
            if builder is None:
                return 422, {"message": f"unknown query type {query.get('type')!r}", "field": f"queries.{qid}.type"}
            answer, backend_calls = builder(context, backend, query, profile)
            minimum = minimum_calibration_size(profile["alpha"])
            if profile["n"] < minimum:
                insufficient = True
                answer["guarantee"] = {
                    "type": "heuristic",
                    "statement": (f"profile '{name}' has n={profile['n']}, below the minimum "
                                  f"{minimum} for alpha={profile['alpha']:g}; no formal guarantee"),
                    "calibration_profile": name, "calibration_n": profile["n"],
                }
                warnings.append(f"{qid}: calibration profile '{name}' is below its minimum size")
            answers[qid] = answer
            calls += backend_calls

        provider = backend.get("provider")
        backend_echo = {k: backend[k] for k in ("provider", "model", "name") if backend.get(k)}
        if provider:
            backend_echo["access_level"] = backend.get("access_level") or _ACCESS_LEVELS.get(provider, "L0")
        payload = {
            "backend": backend_echo,
            "answers": answers,
            "usage": {"backend_calls": calls, "backend_tokens": calls * 287},
            "warnings": warnings,
        }
        return (424 if insufficient else 200), payload

    def _card(self, profile: dict[str, Any], guarantee_type: str, **fields: Any) -> dict[str, Any]:
        card = {"type": guarantee_type, **{k: v for k, v in fields.items() if v is not None}}
        card["calibration_profile"] = profile["name"]
        card["calibration_n"] = profile["n"]
        if guarantee_type == "coverage" and profile["n"] > 0 and "alpha" in card:
            card["coverage_ci"] = [round(v, 3) for v in coverage_confidence_interval(profile["n"], card["alpha"])]
        if profile["last_audit"]:
            card["last_audited"] = f"{profile['last_audit']['date']}T00:00:00Z"
        card["stats_version"] = STATS_VERSION
        return card

    @staticmethod
    def _calls_for(backend: dict[str, Any], samples_needed: bool = True) -> int:
        """Backend calls one query costs at the backend's access level."""
        if backend.get("provider") == "client_evidence":
            return 0  # the SDK computed the evidence locally
        level = _ACCESS_LEVELS.get(backend.get("provider", ""), "L0")
        if level == "L0" and samples_needed:
            return int(backend.get("sample_count") or 20)
        return 1

    # Option probabilities: from client evidence when present, else from the
    # keyword model above (support teams) or a stable hash (anything else).
    def _option_probabilities(self, context: Any, options: list[str],
                              evidence: Optional[dict[str, Any]]) -> dict[str, float]:
        if evidence and "option_probabilities" in evidence:
            raw = {o: max(0.0, float(evidence["option_probabilities"].get(o, 0.0))) for o in options}
            total = sum(raw.values()) or 1.0
            return {o: v / total for o, v in raw.items()}
        if evidence and "samples" in evidence:
            samples = [str(s).strip().lower() for s in evidence["samples"]]
            counts = {o: 0.5 + sum(s == o.lower() for s in samples) for o in options}
            total = sum(counts.values())
            return {o: c / total for o, c in counts.items()}

        if isinstance(context, dict):
            # Score free-text fields only; categorical fields such as
            # account_tier="enterprise" are metadata, not ticket wording.
            text = " ".join(v for v in context.values() if isinstance(v, str) and " " in v.strip())
        else:
            text = _flatten_text(context)
        text = text.lower()
        tokens = _tokens(text)
        logits = {}
        for option in options:
            weights = _TEAM_KEYWORDS.get(option, {option.lower(): 1.0})
            score = 0.0
            for keyword, weight in weights.items():
                if " " in keyword:
                    score += weight * text.count(keyword)
                else:
                    score += weight * sum(t.startswith(keyword) for t in tokens)
            jitter = 0.2 * (_stable_unit(text, option) - 0.5)
            logits[option] = 0.3 + 2.0 * score + jitter
        return _softmax(logits)

    @staticmethod
    def _aps_set(probs: dict[str, float], alpha: float) -> list[str]:
        """Include options in descending probability until the mass reaches q_hat."""
        q_hat = 1 - alpha
        chosen, mass = [], 0.0
        for option, p in sorted(probs.items(), key=lambda kv: -kv[1]):
            chosen.append(option)
            mass += p
            if mass >= q_hat:
                break
        return chosen

    def _answer_set(self, context, backend, query, profile):
        options = list((query.get("options") or {}).keys())
        alpha = float(query.get("alpha") or profile["alpha"])
        method = query.get("method") or "APS"
        probs = self._option_probabilities(context, options, query.get("evidence"))
        if method == "LAC":
            prediction_set = [o for o, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p >= alpha + 0.02]
        else:
            prediction_set = self._aps_set(probs, alpha)
        answer = {
            "type": "set",
            "set": prediction_set,
            "probabilities": {o: round(p, 3) for o, p in probs.items()},
            "venn_abers": {o: _venn_abers_interval(p, profile["n"]) for o, p in probs.items()},
            "guarantee": self._card(profile, "coverage", alpha=alpha, method=method),
        }
        return answer, self._calls_for(backend)

    @staticmethod
    def _gate_threshold(guarantee: str, target: float) -> float:
        slack = {"fdr": 1.3, "risk": 1.1, "risk_high_probability": 1.6}.get(guarantee, 1.2)
        return round(min(0.995, max(0.5, 1 - slack * target)), 3)

    def _answer_gate(self, context, backend, query, profile):
        evidence = query.get("evidence")
        if evidence and ("option_probabilities" in evidence or "samples" in evidence):
            confidence = self._option_probabilities(context, ["true", "false"], evidence)["true"]
        else:
            options = profile["labels"] or list(_TEAM_KEYWORDS)
            confidence = max(self._option_probabilities(context, options, None).values())
        guarantee = query.get("guarantee", "risk")
        target = float(query.get("target"))
        threshold = self._gate_threshold(guarantee, target)
        if confidence >= threshold:
            decision = "auto_approve"
        elif confidence < 0.45:
            decision = "abstain"
        else:
            decision = "escalate"
        # One decision per request: fdr uses the selective Learn-then-Test bound,
        # which is a high-probability statement and always carries delta.
        method = {"risk": "CRC", "risk_high_probability": "RCPS", "fdr": "LTT"}[guarantee]
        delta = query.get("delta") if guarantee != "fdr" else (query.get("delta") or 0.10)
        card = self._card(profile, guarantee, target=target, delta=delta, method=method,
                          realized_upper_bound=round(target * 0.94, 3) if guarantee == "fdr" else None)
        answer = {"type": "gate", "decision": decision, "guarantee": card,
                  "explanation": {"score": round(confidence, 3), "threshold": threshold}}
        return answer, self._calls_for(backend)

    def _answer_belief(self, context, backend, query, profile):
        evidence = query.get("evidence")
        if evidence:
            p = self._option_probabilities(context, ["true", "false"], evidence)["true"]
        else:
            p = 0.15 + 0.8 * _stable_unit(_flatten_text(context), _flatten_text(query.get("instructions")))
        p0, p1 = _venn_abers_interval(p, profile["n"])
        answer = {"type": "belief", "probability": round((p0 + p1) / 2, 3), "venn_abers": [p0, p1],
                  "guarantee": self._card(profile, "calibration", method="IVAP")}
        return answer, self._calls_for(backend)

    def _answer_interval(self, context, backend, query, profile):
        levels = query.get("levels") or []
        evidence = query.get("evidence") or {}
        if "level_probabilities" in evidence:
            probs = {int(k): float(v) for k, v in evidence["level_probabilities"].items()}
            total = sum(probs.values()) or 1.0
            point = sum(k * v for k, v in probs.items()) / total
        else:
            point = (len(levels) - 1) * _stable_unit(_flatten_text(context), "interval")
        lo, hi = max(0.0, math.floor(point - 0.4)), min(len(levels) - 1.0, math.ceil(point + 0.4))
        alpha = float(query.get("alpha") or profile["alpha"])
        answer = {"type": "interval", "point_estimate": round(point, 2), "interval": [lo, hi],
                  "legend": {str(i): name for i, name in enumerate(levels)},
                  "guarantee": self._card(profile, "coverage", alpha=alpha, method=query.get("method") or "CQR")}
        return answer, self._calls_for(backend)

    def _answer_claim(self, context, backend, query, profile):
        context = context if isinstance(context, dict) else {"draft_answer": _flatten_text(context)}
        draft = _flatten_text(context.get("draft_answer", ""))
        source_field = query.get("support_source") or next(
            (k for k in ("retrieved_docs", "documents", "sources") if k in context), None)
        source_words = _content_words(_flatten_text(context.get(source_field))) if source_field else set()
        source_numbers = set(re.findall(r"\d+", _flatten_text(context.get(source_field))))
        alpha = float(query.get("alpha") or profile["alpha"])
        threshold = round(0.55 + 2.0 * alpha, 3)  # calibrated support threshold for this alpha

        retained, dropped = [], []
        for claim in (c.strip() for c in re.split(r"(?<=[.!?])\s+", draft)):
            if not claim:
                continue
            words = _content_words(claim)
            support = len(words & source_words) / max(len(words), 1)
            if any(num not in source_numbers for num in re.findall(r"\d+", claim)):
                support *= 0.4  # a number the sources never state is treated as unsupported
            if support >= threshold:
                retained.append(claim)
            else:
                dropped.append({"text": claim, "reason": "unsupported", "score": round(support, 3)})

        card = self._card(profile, "risk", target=alpha, method="conformal-factuality",
                          statement=(f"P(every retained claim is supported) >= {1 - alpha:.2f}, "
                                     "over answers exchangeable with this profile"))
        answer = {"type": "claim", "retained_claims": retained, "dropped_claims": dropped,
                  "guarantee": card, "support_threshold": threshold}
        # L0 backends score support by resampling: one decomposition call plus the samples.
        calls = 1 + (self._calls_for(backend) if _ACCESS_LEVELS.get(backend.get("provider", "")) == "L0" else 1)
        return answer, calls

    @staticmethod
    def _stage_name(stage: dict[str, Any]) -> str:
        backend = stage.get("backend")
        if isinstance(backend, dict):
            return str(backend.get("model") or backend.get("provider"))
        return str(backend)

    def _answer_judge(self, context, backend, query, profile):
        context = context if isinstance(context, dict) else {}
        prompt_words = _content_words(_flatten_text(context.get("prompt")))

        def quality(response: Any) -> float:
            words = _content_words(_flatten_text(response))
            return len(words & prompt_words) / max(len(prompt_words), 1)

        margin = quality(context.get("response_a")) - quality(context.get("response_b"))
        cascade = query.get("cascade") or [{"backend": backend}]
        alpha = float(query.get("alpha") or profile["alpha"])
        stages, winner, escalated_to, calls = [], None, None, 0
        for i, stage in enumerate(cascade):
            name = self._stage_name(stage)
            if name == "human_queue":
                escalated_to, winner = "human_queue", None
                stages.append({"stage": name, "decided": False})
                break
            strength = 1.0 if (calls == 0 or "mini" in name) else 1.8
            p_a = _sigmoid(6.0 * strength * margin)
            half_width = 0.15 / strength
            interval = [round(max(0.0, p_a - half_width), 3), round(min(1.0, p_a + half_width), 3)]
            decided = interval[0] >= 0.62 or interval[1] <= 0.38
            calls += 1
            stages.append({"stage": name, "p_response_a": round(p_a, 3), "venn_abers": interval, "decided": decided})
            if decided:
                winner = "response_a" if interval[0] >= 0.62 else "response_b"
                escalated_to = None if i == 0 else name
                break
        card = self._card(profile, "risk_high_probability", target=alpha, delta=0.05, method="trust-or-escalate",
                          statement=f"human agreement >= {1 - alpha:.2f} on verdicts the cascade returns")
        answer = {"type": "judge", "winner": winner, "escalated_to": escalated_to,
                  "guarantee": card, "stages": stages}
        return answer, calls

    def _answer_route(self, context, backend, query, profile):
        text = _flatten_text(context)
        words = len(_tokens(text))
        hard = sum(text.lower().count(m) for m in ("prove", "derive", "compare", "contract", "legal",
                                                   "tax", "step by step", "why does", "trade-off"))
        confidence = max(0.05, min(0.99, 0.97 - 0.012 * max(0, words - 10) - 0.22 * hard))
        cascade = query.get("cascade") or []
        escalate = confidence < 0.72 and len(cascade) > 1
        cheap_cost = round(0.02 + 0.0008 * words, 3)
        cost = cheap_cost + (round(0.18 + 0.006 * words, 3) if escalate else 0.0)
        served = (cascade[1] if escalate else cascade[0]).get("backend") or {}
        served_by = {k: served[k] for k in ("provider", "model") if isinstance(served, dict) and served.get(k)}
        alpha = float(query.get("alpha") or profile["alpha"])
        if query.get("guarantee", "cost_budget") == "cost_budget":
            card = self._card(profile, "cost_budget", target_cents=query.get("target_cents"), alpha=alpha,
                              method="calibrated-cascade")
        else:
            card = self._card(profile, "risk", target=alpha, method="calibrated-cascade",
                              statement=f"answer is correct with probability >= {1 - alpha:.2f}")
        answer = {
            "type": "route", "served_by": served_by, "escalated": escalate,
            "cost_cents": round(cost, 3),
            "output": f"[{served_by.get('model', 'model')}] answer to: {text[:48]}",
            "guarantee": card,
            "tier_confidence": _venn_abers_interval(confidence, profile["n"]),
            "stages_run": [self._stage_name(s) for s in cascade[: 2 if escalate else 1]],
        }
        return answer, 2 if escalate else 1


# -- client construction ---------------------------------------------------


def parse_args(description: Optional[str] = None, *, drift_flag: bool = False) -> argparse.Namespace:
    """Parse ``--mock`` (and optionally ``--simulate-drift``).

    The mock is used when ``--mock`` is passed or when ``CLI_API_KEY`` is
    unset, so every example runs offline by default. ``args.server`` is the
    ``MockCLIServer`` to use, or ``None`` for the live API.
    """
    parser = argparse.ArgumentParser(description=(description or "").split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true",
                        help=f"run against the offline mock API (the default when {ENV_API_KEY} is unset)")
    if drift_flag:
        parser.add_argument("--simulate-drift", action="store_true",
                            help="make the mock's drift monitors report alerts")
    args = parser.parse_args()
    args.mock = args.mock or not os.environ.get(ENV_API_KEY)
    if args.mock:
        args.server = MockCLIServer(simulate_drift=getattr(args, "simulate_drift", False))
        print(f"[mock] offline mock of {DEFAULT_BASE_URL} (no network; drop --mock and set {ENV_API_KEY} for the live API)\n")
    else:
        args.server = None
    return args


@contextmanager
def open_client(server: Optional[MockCLIServer], *, event_hooks: Optional[dict] = None,
                **client_kwargs: Any) -> Iterator[CLIClient]:
    """A ``CLIClient`` on the mock transport when ``server`` is given, else the live API.

    ``event_hooks`` are passed to the underlying ``httpx.Client`` in both
    modes, so an example can inspect the exact request bodies it sends.
    """
    timeout = client_kwargs.get("timeout", 60.0)
    if server is None:
        http = httpx.Client(timeout=timeout, event_hooks=event_hooks or {})
        with http, CLIClient(http_client=http, **client_kwargs) as client:
            yield client
        return
    http = httpx.Client(transport=server.transport(), event_hooks=event_hooks or {})
    with http, CLIClient(api_key=MOCK_API_KEY, base_url=DEFAULT_BASE_URL, http_client=http,
                         **client_kwargs) as client:
        yield client
