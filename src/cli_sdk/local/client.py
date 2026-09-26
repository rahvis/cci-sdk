"""``LocalCLIClient``: the CLI guarantee engine, in-process, on your own model.

Same queries, same typed answers, same guarantee cards as ``CLIClient`` —
but calibration and scoring run inside your process against the model you
bring (OpenAI, Azure OpenAI, Anthropic, Gemini, or an open-weight model on
vLLM / SGLang). Nothing is sent anywhere except to your model provider.

::

    from cli_sdk.local import LocalCLIClient
    from cli_sdk.evidence import OpenAIEvidenceBackend
    from cli_sdk import Gate

    client = LocalCLIClient(OpenAIEvidenceBackend("gpt-4.1-mini-2025-04-14", api_key="YOUR_OPENAI_API_KEY"))
    refund = Gate(instructions="Is this refund within policy?", calibration_profile="refunds-v1",
                  guarantee="risk", target=0.05)
    client.calibrate(refund, labelled_examples)          # scores each example once, stores evidence
    result = client.evaluate(context=ticket, queries={"refund": refund})
    result.answers["refund"].approved

Fail-closed rules: when a profile is missing, too small for the requested
alpha/target, or was calibrated with a different backend or a different
prompt, the answer is labelled ``heuristic`` and takes its safe form (Set:
every option; Gate: escalate; Belief: [0, 1]; Interval: every level; Claim:
nothing retained; Judge / Route: human queue). With
``strict_guarantees=True`` the client raises ``InsufficientCalibrationError``
instead.
"""

from __future__ import annotations

import math
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Mapping, Optional, Sequence

from cli_sdk.answers import EvaluateResponse, GateAnswer, parse_answer
from cli_sdk.answers.response import Usage
from cli_sdk.backends.base import CustomBackend, access_rank
from cli_sdk.calibration.audit import AuditResult, audit_coverage, audit_risk
from cli_sdk.calibration.examples import normalize_examples
from cli_sdk.calibration.profile import CalibrationProfile, GroupStatus, recommended_size
from cli_sdk.constants import ACCESS_LEVEL_L1
from cli_sdk.exceptions import ConfigurationError, InsufficientCalibrationError
from cli_sdk.local import engine, scoring
from cli_sdk.local.scoring import UsageCounter
from cli_sdk.local.store import LocalProfileStore, ProfileRecord, query_fingerprint
from cli_sdk.monitoring.monitor import LocalMonitor
from cli_sdk.queries import Gate, Query
from cli_sdk.stats.conformal._quantile import coverage_confidence_interval, minimum_calibration_size

ProgressFn = Callable[[int, int], None]


class LocalCLIClient:
    """Calibrate and evaluate CLI queries fully in-process.

    Parameters
    ----------
    backend:
        The default evidence backend: any ``CustomBackend``, typically one
        from ``cli_sdk.evidence``.
    backends:
        Named backends for ``Judge`` and ``Route`` cascades, e.g.
        ``{"small": vllm_backend("google/gemma-3-4b-it"), "large": OpenAIEvidenceBackend(...)}``.
    store:
        Directory (or ``LocalProfileStore``) holding calibration profiles.
    sample_count:
        Samples per score for access-level L0 backends.
    strict_guarantees:
        Raise instead of returning ``heuristic`` answers.
    max_workers:
        Concurrent model calls while calibrating.
    """

    def __init__(
        self,
        backend: Optional[CustomBackend] = None,
        *,
        backends: Optional[Mapping[str, CustomBackend]] = None,
        store: str | LocalProfileStore = ".cli_profiles",
        sample_count: int = 20,
        strict_guarantees: bool = False,
        max_workers: int = 4,
    ) -> None:
        self.backends = dict(backends or {})
        if backend is None and len(self.backends) == 1:
            backend = next(iter(self.backends.values()))
        self.backend = backend
        self.store = store if isinstance(store, LocalProfileStore) else LocalProfileStore(store)
        if sample_count < 1:
            raise ValueError("sample_count must be >= 1")
        self.sample_count = sample_count
        self.strict_guarantees = strict_guarantees
        self.max_workers = max(1, max_workers)

    # ------------------------------------------------------------------
    # backends
    # ------------------------------------------------------------------

    def _default(self, backend: Optional[CustomBackend]) -> CustomBackend:
        chosen = backend or self.backend
        if chosen is None:
            raise ConfigurationError("no backend: pass LocalCLIClient(backend=...) or evaluate(..., backend=...)")
        return chosen

    def _resolve(self, name: str) -> CustomBackend:
        if name not in self.backends:
            raise ConfigurationError(f"cascade names backend {name!r}; register it in LocalCLIClient(backends={{...}})")
        return self.backends[name]

    def _backend_fingerprint(self, backend: CustomBackend) -> dict[str, Any]:
        fp = scoring.backend_fingerprint(backend)
        if access_rank(backend.access_level) < access_rank(ACCESS_LEVEL_L1):
            # At L0 every score is a smoothed frequency over sample_count draws, so
            # the sample count is part of the scoring function: a profile scored
            # with 20 samples does not describe scores computed from 8.
            fp["sample_count"] = self.sample_count
        return fp

    def _fingerprint(self, query: Query, backend: CustomBackend) -> dict[str, Any]:
        if query.type == "judge":
            return {"stages": [self._backend_fingerprint(backend if n is None else self._resolve(n))
                               for n in engine.judge_stages(query)]}
        if query.type == "route":
            return {"tiers": [self._backend_fingerprint(self._resolve(t["backend"]))
                              for t in engine.route_tiers(query)]}
        return self._backend_fingerprint(backend)

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        query: Query,
        examples: Sequence[Any],
        *,
        backend: Optional[CustomBackend] = None,
        replace: bool = True,
        progress: Optional[ProgressFn] = None,
    ) -> CalibrationProfile:
        """Score labelled examples with the backend and store them as the query's profile.

        Each example is ``{"context": ..., "label": ...}`` (or a
        ``CalibrationExample``). Label by query type:

        ===========  ==========================================================
        Set          the correct option key
        Interval     the correct level
        Belief       ``True`` if the statement is true
        Gate         ``True`` if approving would be correct (loss = approving a wrong one)
        Claim        list of ``{"text": str, "supported": bool}`` for the answer's claims
        Judge        ``"response_a"`` or ``"response_b"``: the human preference
        Route        the correct option key of ``Route.task``
        ===========  ==========================================================

        ``replace=False`` appends to an existing profile instead.
        """
        backend = self._default(backend)
        payload = query.to_payload()
        fingerprint = self._fingerprint(query, backend)
        qfp = query_fingerprint(payload)
        existing = self.store.load(query.calibration_profile)
        if existing is not None and not replace:
            if existing.query_fingerprint != qfp or existing.backend_fingerprint != fingerprint:
                raise ConfigurationError(
                    f"profile {query.calibration_profile!r} was calibrated with a different query or backend; "
                    "calibrate(..., replace=True) to start a new version"
                )
            record = existing
        else:
            record = ProfileRecord(
                name=query.calibration_profile,
                query_type=query.type,
                query=payload,
                query_fingerprint=qfp,
                backend_fingerprint=fingerprint,
                method=engine.profile_method(query.type, payload, getattr(query, "method", None)),
                group_by=getattr(query, "group_by", None),
                version=(existing.version + 1) if existing is not None else 1,
            )
        normalized = normalize_examples(list(examples))
        record.records.extend(self._score_examples(query, backend, normalized, progress))
        self.store.save(record)
        return self._profile(record, query)

    def add_examples(self, query: Query, examples: Sequence[Any], *, backend: Optional[CustomBackend] = None,
                     progress: Optional[ProgressFn] = None) -> CalibrationProfile:
        """Append labelled examples to the query's existing profile."""
        return self.calibrate(query, examples, backend=backend, replace=False, progress=progress)

    def _score_examples(self, query: Query, backend: CustomBackend, examples: list[dict[str, Any]],
                        progress: Optional[ProgressFn]) -> list[dict[str, Any]]:
        evidence_fn = engine.EVIDENCE[query.type]
        group_by = getattr(query, "group_by", None)

        def one(example: dict[str, Any]) -> dict[str, Any]:
            context, label = example["context"], example["label"]
            if query.type == "claim":
                claims = _claim_labels(label)
                ev = evidence_fn(backend, query, context, self.sample_count, None, self._resolve,
                                 claims=[c["text"] for c in claims])
                for scored, labelled in zip(ev["claims"], claims):
                    scored["supported"] = bool(labelled["supported"])
                label = None
            else:
                ev = evidence_fn(backend, query, context, self.sample_count, None, self._resolve)
            rec: dict[str, Any] = {"evidence": ev, "label": label, "source": example.get("source", "human")}
            group = example.get("group")
            if group is None and group_by and isinstance(context, Mapping):
                group = context.get(group_by)
            if group is not None:
                rec["group"] = scoring.canonical(group)
            return rec

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for i, rec in enumerate(pool.map(one, examples), start=1):
                results.append(rec)
                if progress:
                    progress(i, len(examples))
        return results

    # ------------------------------------------------------------------
    # profiles
    # ------------------------------------------------------------------

    def _profile(self, record: ProfileRecord, query: Optional[Query] = None) -> CalibrationProfile:
        q = query.to_payload() if query is not None else record.query
        level = q.get("alpha") or q.get("target") or engine.DEFAULT_ALPHA
        groups: dict[str, GroupStatus] = {}
        # The size evaluation actually requires for this query (RCPS, LTT and
        # Venn-Abers need more than the split-conformal floor).
        minimum = engine.minimum_examples({"type": record.query_type, **q})
        if record.group_by:
            group_minimum = minimum_calibration_size(level)
            counts: dict[str, int] = {}
            for r in record.records:
                counts[r.get("group", "None")] = counts.get(r.get("group", "None"), 0) + 1
            groups = {g: GroupStatus(n=c, status="calibrated" if c >= group_minimum else "underpowered")
                      for g, c in sorted(counts.items())}
        return CalibrationProfile(
            name=record.name,
            method=engine.profile_method(record.query_type, q, record.method),
            alpha=float(level),
            n=record.n,
            version=record.version,
            recommended_n=recommended_size(level),
            minimum_n=minimum,
            realized_coverage_ci=coverage_confidence_interval(record.n, level) if record.n else None,
            backend_fingerprint=record.backend_fingerprint,
            group_by=record.group_by,
            groups=groups,
            status="serving" if record.n >= minimum else "collecting",
        )

    def get_profile(self, name: str) -> CalibrationProfile:
        record = self.store.load(name)
        if record is None:
            raise ConfigurationError(f"no local calibration profile named {name!r} in {self.store.root}")
        return self._profile(record)

    def list_profiles(self) -> list[CalibrationProfile]:
        return [self.get_profile(name) for name in self.store.names()]

    def calibration_status(self, query: Query, *, backend: Optional[CustomBackend] = None) -> tuple[bool, str]:
        """``(True, "")`` when the query's profile exists and matches this query and backend.

        Otherwise ``(False, reason)``: the profile is missing, or was built with
        a different prompt, options, backend or settings, and would produce
        heuristic answers until recalibrated. Sample-size requirements are
        checked per answer, because they depend on ``alpha`` / ``target``.
        """
        record, reason = self._usable_record(query, self._default(backend))
        return record is not None, reason

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        context: Any,
        queries: Mapping[str, Query],
        *,
        backend: Optional[CustomBackend] = None,
        response_model: Optional[type[EvaluateResponse]] = None,
    ) -> EvaluateResponse:
        """Answer every query about ``context`` with its calibrated guarantee."""
        if not queries:
            raise ValueError("evaluate() needs at least one query")
        backend = self._default(backend)
        usage = UsageCounter()
        warnings: list[str] = []
        answers = {}
        for qid, query in queries.items():
            payload = self._answer(query, context, backend, usage, warnings)
            answers[qid] = payload
        response_cls = response_model or EvaluateResponse
        response = response_cls.from_payload(
            {"answers": answers, "backend": {"provider": "local", **_short_fp(backend)},
             "usage": {"backend_calls": usage.backend_calls}, "warnings": warnings},
            request_id=f"local-{uuid.uuid4().hex[:12]}",
        )
        return response

    def gate_batch(self, contexts: Sequence[Any], query: Gate, *, backend: Optional[CustomBackend] = None) -> list[GateAnswer]:
        """Decide a whole batch with finite-sample FDR control (conformal selection + BH).

        Use this when decisions arrive in batches (a queue of refunds, a list of
        candidate matches): at most ``query.target`` of the approved items are
        wrong in expectation, across the batch.
        """
        if not isinstance(query, Gate):
            raise ConfigurationError("gate_batch takes a Gate query")
        backend = self._default(backend)
        record, reason = self._usable_record(query, backend)
        scores = [engine.binary_evidence(backend, query, c, self.sample_count, None)["score"] for c in contexts]
        warnings: list[str] = []
        if record is None:
            self._maybe_raise(reason)
            payloads = [{"type": "gate", "decision": "escalate", "confidence": s,
                         "guarantee": engine.heuristic_card(query.calibration_profile, reason)} for s in scores]
        else:
            payloads = engine.gate_batch_answers(query, record, scores, warnings)
        return [parse_answer(p) for p in payloads]  # type: ignore[misc]

    def _maybe_raise(self, reason: str) -> None:
        if self.strict_guarantees:
            raise InsufficientCalibrationError(reason)

    def _usable_record(self, query: Query, backend: CustomBackend) -> tuple[Optional[ProfileRecord], str]:
        record = self.store.load(query.calibration_profile)
        if record is None:
            return None, f"calibration profile {query.calibration_profile!r} not found in {self.store.root}"
        if record.query_type != query.type:
            return None, f"profile {record.name!r} calibrates a {record.query_type} query, not {query.type}"
        if record.query_fingerprint != query_fingerprint(query.to_payload()):
            return None, (f"profile {record.name!r} was calibrated with different instructions or options; "
                          "recalibrate after changing the prompt")
        if record.backend_fingerprint != self._fingerprint(query, backend):
            return None, (f"profile {record.name!r} was calibrated with a different backend or settings "
                          f"({_describe_fp(record.backend_fingerprint)}); recalibrate for this backend")
        return record, ""

    def _answer(self, query: Query, context: Any, backend: CustomBackend, usage: UsageCounter,
                warnings: list[str]) -> dict[str, Any]:
        record, reason = self._usable_record(query, backend)
        if record is not None:
            if query.type == "judge":
                return self._judge(query, record, context, backend, usage, warnings)
            if query.type == "route":
                return self._route(query, record, context, usage, warnings)
            evidence = engine.EVIDENCE[query.type](backend, query, context, self.sample_count, usage, self._resolve)
            group = None
            if getattr(query, "group_by", None) and isinstance(context, Mapping):
                group = context.get(query.group_by)
            payload = engine.ANSWER[query.type](query, record, evidence, group, warnings)
        else:
            warnings.append(reason)
            payload = self._fail_closed(query, context, backend, usage, reason)
        if payload["guarantee"]["type"] == "heuristic":
            self._maybe_raise(payload["guarantee"]["statement"])
        return payload

    def _fail_closed(self, query: Query, context: Any, backend: CustomBackend, usage: UsageCounter,
                     reason: str) -> dict[str, Any]:
        card = engine.heuristic_card(query.calibration_profile, reason)
        if self.strict_guarantees:
            raise InsufficientCalibrationError(reason)
        if query.type == "judge":
            return {"type": "judge", "winner": None, "escalated_to": "human_queue", "guarantee": card}
        if query.type == "route":
            return {"type": "route", "served_by": {"backend": "human_queue"}, "escalated": True,
                    "output": None, "guarantee": card}
        empty = ProfileRecord(name=query.calibration_profile, query_type=query.type, query={},
                              query_fingerprint="", backend_fingerprint={})
        evidence = engine.EVIDENCE[query.type](backend, query, context, self.sample_count, usage, self._resolve)
        payload = engine.ANSWER[query.type](query, empty, evidence, None, [])
        payload["guarantee"] = card
        return payload

    # -- cascades ---------------------------------------------------------

    def _judge(self, query, record: ProfileRecord, context, backend, usage, warnings) -> dict[str, Any]:
        try:
            thresholds, alpha, delta = engine.judge_thresholds(query, record)
        except engine.Heuristic as reason:
            return {"type": "judge", "winner": None, "escalated_to": "human_queue",
                    "guarantee": engine.heuristic_card(record.name, str(reason), record.n)}
        stages = engine.judge_stages(query)
        card = engine._card(record, "risk_high_probability", "trust-or-escalate", alpha=alpha, delta=delta,
                            target=alpha, statement=(
                                f"With {1 - delta:.0%} confidence, verdicts that are not escalated agree with "
                                f"human labels at least {1 - alpha:.0%} of the time (n={record.n})."))
        trail = []
        for i, name in enumerate(stages):
            stage_backend = backend if name is None else self._resolve(name)
            probs = scoring.option_probabilities(stage_backend, context, query.instructions,
                                                 scoring.JUDGE_OPTIONS, self.sample_count, usage)
            confidence = max(probs.values())
            winner = max(probs, key=probs.get)
            label = name or "default"
            trail.append({"stage": label, "winner": winner, "confidence": confidence,
                          "threshold": None if math.isinf(thresholds[i]) else thresholds[i]})
            if confidence >= thresholds[i]:
                return {"type": "judge", "winner": winner, "escalated_to": None if i == 0 else label,
                        "decided_by": label, "stages": trail, "guarantee": card}
        return {"type": "judge", "winner": None, "escalated_to": "human_queue", "stages": trail, "guarantee": card}

    def _route(self, query, record: ProfileRecord, context, usage, warnings) -> dict[str, Any]:
        try:
            q_hats, alpha, method = engine.route_thresholds(query, record)
        except engine.Heuristic as reason:
            return {"type": "route", "served_by": {"backend": "human_queue"}, "escalated": True, "output": None,
                    "guarantee": engine.heuristic_card(record.name, str(reason), record.n)}
        task = engine.route_task(query)
        keys = list(task.options)
        tiers = engine.route_tiers(query)
        card = engine._card(record, "risk", "conformal-cascade", alpha=alpha, target=alpha, statement=(
            f"The served answer is wrong at most {alpha:.0%} of the time: each of the {len(tiers)} tiers "
            f"answers only when its {1 - alpha / len(tiers):.1%} conformal set is a single option (n={record.n})."))
        cost = 0.0
        trail = []
        import numpy as np

        for t, tier in enumerate(tiers):
            probs = scoring.option_probabilities(self._resolve(tier["backend"]), context, task.instructions,
                                                 task.options, self.sample_count, usage)
            cost += float(tier.get("cost_cents") or 0.0)
            members = engine._set_predict(method, np.array([probs[k] for k in keys]), q_hats[t])
            trail.append({"backend": tier["backend"], "set": [keys[j] for j in members]})
            if len(members) == 1:
                return {"type": "route", "served_by": {"backend": tier["backend"], "tier": t},
                        "escalated": t > 0, "cost_cents": cost, "output": keys[members[0]],
                        "tiers": trail, "guarantee": card}
        return {"type": "route", "served_by": {"backend": "human_queue"}, "escalated": True,
                "cost_cents": cost, "output": None, "tiers": trail, "guarantee": card}

    # ------------------------------------------------------------------
    # audit and monitoring
    # ------------------------------------------------------------------

    def audit(self, query: Query, examples: Sequence[Any], *, backend: Optional[CustomBackend] = None,
              confidence: float = 0.95) -> AuditResult:
        """Check the guarantee on fresh labelled examples that were not used to calibrate.

        Passes unless the Clopper-Pearson interval shows real evidence of a
        violation. Supported for Set, Interval, Gate, Claim, Judge and Route.
        """
        backend = self._default(backend)
        normalized = normalize_examples(list(examples))
        if not normalized:
            raise ValueError("audit needs at least one labelled example")
        qtype = query.type
        if qtype == "belief":
            raise ConfigurationError("Belief is a calibrated probability, not a coverage or risk guarantee; "
                                     "audit a Gate or Set built on the same question instead")
        outcomes: list[bool] = []
        gate_outcomes: list[tuple[bool, bool]] = []  # (approved, wrong)
        for example in normalized:
            context, label = example["context"], example["label"]
            if qtype == "claim":
                claims = _claim_labels(label)
                record, reason = self._usable_record(query, backend)
                if record is None:
                    raise InsufficientCalibrationError(reason)
                ev = engine.claim_evidence(backend, query, context, self.sample_count, None,
                                           claims=[c["text"] for c in claims])
                answer = engine.claim_answer(query, record, ev, None, [])
                truth = {c["text"]: bool(c["supported"]) for c in claims}
                outcomes.append(all(truth.get(text, False) for text in answer["retained_claims"]))
                continue
            answer = self.evaluate(context, {"q": query}, backend=backend).answers["q"]
            if answer.is_heuristic:
                raise InsufficientCalibrationError(f"cannot audit: {answer.guarantee.statement}")
            if qtype == "set":
                outcomes.append(scoring.canonical(label) in answer.set)
            elif qtype == "interval":
                index = list(query.levels).index(scoring.canonical(label))
                outcomes.append(answer.contains(index))
            elif qtype == "gate":
                wrong = not bool(engine._binary_labels([{"label": label}])[0])
                gate_outcomes.append((answer.approved, wrong))
            elif qtype == "judge":
                outcomes.append(answer.winner is None or answer.winner == label)
            elif qtype == "route":
                outcomes.append(answer.output is None or answer.output == scoring.canonical(label))
        if qtype == "gate":
            if query.guarantee == "fdr":
                # FDR is the error rate among approved items.
                among_approved = [wrong for approved, wrong in gate_outcomes if approved]
                if not among_approved:
                    raise ValueError("no audit example was approved, so the approved-error rate is undefined")
                return audit_risk(among_approved, query.target, confidence)
            return audit_risk([approved and wrong for approved, wrong in gate_outcomes], query.target, confidence)
        level = getattr(query, "alpha", None) or engine.DEFAULT_ALPHA
        return audit_coverage(outcomes, 1 - level, confidence)

    def monitor(self, query: Query, false_alarm_rate: float = 0.05) -> LocalMonitor:
        """An anytime-valid drift monitor matching the query's guarantee.

        Feed it production outcomes as labels arrive: ``covered`` (bool) for
        Set / Interval / Claim, ``is_loss`` (approved-and-wrong) for Gate
        ``risk`` and ``risk_high_probability``. For Gate ``fdr``, whose bound is
        on the error rate *among approved* decisions, call ``update`` only for
        auto-approved decisions, with ``True`` when the approval was wrong.
        """
        if query.type == "gate":
            return LocalMonitor("risk", target=query.target, false_alarm_rate=false_alarm_rate,
                                profile=query.calibration_profile)
        level = getattr(query, "alpha", None) or engine.DEFAULT_ALPHA
        return LocalMonitor("coverage", target=1 - level, false_alarm_rate=false_alarm_rate,
                            profile=query.calibration_profile)


def _claim_labels(label: Any) -> list[dict[str, Any]]:
    if not isinstance(label, list) or not all(isinstance(c, dict) and "text" in c and "supported" in c for c in label):
        raise ConfigurationError('Claim calibration labels are lists of {"text": str, "supported": bool}')
    return label


def _short_fp(backend: CustomBackend) -> dict[str, Any]:
    fp = scoring.backend_fingerprint(backend)
    return {k: fp[k] for k in ("model", "access_level") if k in fp}


def _describe_fp(fp: dict[str, Any]) -> str:
    if "model" in fp:
        return f"{fp.get('provider')}:{fp.get('model')} at {fp.get('access_level')}"
    return "a different backend configuration"
