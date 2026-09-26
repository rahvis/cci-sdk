"""Transport-independent logic shared by the sync and async clients."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

from cli_sdk.answers.response import EvaluateResponse
from cli_sdk.backends.base import BackendLike, CustomBackend, backend_payload
from cli_sdk.client.retries import RetryConfig
from cli_sdk.constants import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    ENV_API_KEY,
    ENV_BASE_URL,
    USER_AGENT,
)
from cli_sdk.exceptions import (
    AuthenticationError,
    BackendError,
    CLIError,
    ConfigurationError,
    InsufficientCalibrationError,
    RateLimitError,
    ValidationError,
)
from cli_sdk.queries.base import Query

# Query types whose backends always come from their own `cascade`: they
# need no request-level `backend`, and a CustomBackend collects no evidence
# for them.
_CASCADE_TYPES = {"route"}

# Hosts treated as a local self-hosted deployment, where no API key is needed.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _declares_cascade(query_payload: Mapping[str, Any]) -> bool:
    """True when a query names its own backends (api-reference.mdx: the
    request-level ``backend`` is "required unless every query specifies
    ``cascade``")."""
    return query_payload["type"] in _CASCADE_TYPES or bool(query_payload.get("cascade"))


class BaseClient:
    """Configuration and request/response handling common to both clients."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        backend: Optional[BackendLike] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retry: Optional[RetryConfig] = None,
        strict_guarantees: bool = False,
        sample_count: int = 20,
        default_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get(ENV_API_KEY)
        self.base_url = (base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
        self.default_backend = backend
        self.timeout = timeout
        self.retry = retry or RetryConfig()
        self.strict_guarantees = strict_guarantees
        self.sample_count = sample_count
        self._extra_headers = dict(default_headers or {})
        if not self.api_key and not self._is_local_base_url():
            raise ConfigurationError(
                f"no API key: pass api_key=... or set {ENV_API_KEY} "
                "(not required when base_url points at a local self-hosted deployment)"
            )

    # -- configuration -------------------------------------------------

    def _is_local_base_url(self) -> bool:
        # Compare the parsed host exactly: a substring test would also accept
        # hosts such as "localhost.example.com" or a query string that merely
        # mentions localhost.
        try:
            host = urlsplit(self.base_url).hostname
        except ValueError:
            return False
        return host in _LOCAL_HOSTS

    def _headers(self, idempotency_key: Optional[str] = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self._extra_headers)
        return headers

    # -- request building ----------------------------------------------

    def _build_evaluate_payload(
        self,
        context: Any,
        queries: Mapping[str, Query],
        backend: Optional[BackendLike],
    ) -> dict[str, Any]:
        if not queries:
            raise ConfigurationError("evaluate requires at least one query")

        chosen_backend = backend if backend is not None else self.default_backend
        query_payloads = {}
        for qid, query in queries.items():
            if not isinstance(query, Query):
                raise ConfigurationError(
                    f"query {qid!r} must be a Belief/Set/Interval/Gate/Claim/Judge/Route, "
                    f"got {type(query).__name__}"
                )
            query_payloads[qid] = query.to_payload()

        needs_backend = any(not _declares_cascade(p) for p in query_payloads.values())
        if needs_backend and chosen_backend is None:
            raise ConfigurationError(
                "no backend: pass backend=... to evaluate() or to the client constructor "
                "(only queries that declare their own `cascade`, such as Route, can omit it)"
            )

        payload: dict[str, Any] = {"context": context, "queries": query_payloads}
        if chosen_backend is not None:
            payload["backend"] = backend_payload(chosen_backend)

        # A CustomBackend runs your model locally: compute evidence here and
        # send only the numbers/samples, never the model or its weights.
        if isinstance(chosen_backend, CustomBackend):
            for qid, qpayload in query_payloads.items():
                if qpayload["type"] in _CASCADE_TYPES:
                    continue
                qpayload["evidence"] = chosen_backend.collect_evidence(
                    qpayload, context, sample_count=self.sample_count
                )
        return payload

    # -- response handling ---------------------------------------------

    def _check_evaluate_status(self, status_code: int, body: Any, headers: Mapping[str, str]) -> None:
        """Raise for an evaluate response, except a tolerated 424.

        424: a named profile lacks the examples its guarantee needs. The body
        still carries heuristic-labelled answers; they are returned unless
        ``strict_guarantees`` is on, in which case a 424 always raises, even
        when the body carries no answers to inspect.
        """
        if status_code == 424 and not self.strict_guarantees:
            if isinstance(body, dict) and body.get("answers"):
                return
            # No heuristic answers to return: raise rather than hand back an
            # empty response the caller could mistake for "nothing heuristic".
        self._raise_for_status(status_code, body, headers)

    def _parse_evaluate(
        self,
        data: dict[str, Any],
        response_model: Optional[type[EvaluateResponse]],
        request_id: Optional[str],
    ) -> EvaluateResponse:
        model = response_model or EvaluateResponse
        if not (isinstance(model, type) and issubclass(model, EvaluateResponse)):
            raise ConfigurationError("response_model must be a subclass of EvaluateResponse")
        response = model.from_payload(data, request_id=request_id)
        if self.strict_guarantees and response.heuristic_answers:
            raise InsufficientCalibrationError(
                "strict_guarantees is on and these answers carry no formal guarantee: "
                + ", ".join(response.heuristic_answers)
            )
        return response

    @staticmethod
    def _raise_for_status(status_code: int, body: Any, headers: Mapping[str, str]) -> None:
        if status_code < 400:
            return
        message = _error_message(body, status_code)
        if status_code == 401:
            raise AuthenticationError(message)
        if status_code == 422:
            field = body.get("field") if isinstance(body, dict) else None
            raise ValidationError(message, field=field)
        if status_code == 424:
            raise InsufficientCalibrationError(message)
        if status_code == 429:
            retry_after = headers.get("retry-after")
            raise RateLimitError(message, retry_after=float(retry_after) if _is_number(retry_after) else None)
        if status_code in (502, 529):
            raise BackendError(message)
        raise CLIError(f"HTTP {status_code}: {message}")


def _error_message(body: Any, status_code: int) -> str:
    if isinstance(body, dict):
        for key in ("message", "detail", "error"):
            if key in body:
                return str(body[key])
    return f"request failed with status {status_code}"


def _is_number(value: Optional[str]) -> bool:
    if value is None:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True
