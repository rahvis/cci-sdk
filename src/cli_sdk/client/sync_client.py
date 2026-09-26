"""Synchronous client."""

from __future__ import annotations

import uuid

import time
from typing import Any, Mapping, Optional

import httpx

from cli_sdk.exceptions import APIConnectionError

from cli_sdk.answers.response import EvaluateResponse
from cli_sdk.backends.base import BackendLike
from cli_sdk.calibration.profile import CalibrationProfiles
from cli_sdk.client.base import BaseClient
from cli_sdk.client.retries import RetryConfig
from cli_sdk.queries.base import Query


class CLIClient(BaseClient):
    """Synchronous client for Conformal Logit Inference.

    ::

        from cli_sdk import CLIClient, Set

        with CLIClient() as client:
            result = client.evaluate(
                context={"ticket": "I was charged twice."},
                backend={"provider": "openai", "model": "gpt-4.1-2025-04-14"},
                queries={"department": Set(instructions="Which team?",
                                            options={"billing": None, "technical": None},
                                            calibration_profile="support-routing-v3")},
            )

    ``api_key`` defaults to ``CLI_API_KEY``; ``base_url`` to ``CLI_BASE_URL``
    or the hosted API. Pass ``http_client`` to supply your own
    ``httpx.Client`` (for proxies, custom TLS, or testing with
    ``httpx.MockTransport``).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        backend: Optional[BackendLike] = None,
        timeout: float = 60.0,
        retry: Optional[RetryConfig] = None,
        strict_guarantees: bool = False,
        sample_count: int = 20,
        default_headers: Optional[Mapping[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            backend=backend,
            timeout=timeout,
            retry=retry,
            strict_guarantees=strict_guarantees,
            sample_count=sample_count,
            default_headers=default_headers,
        )
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout)
        self.calibration_profiles = CalibrationProfiles(self)

    # -- lifecycle -------------------------------------------------------

    def __enter__(self) -> "CLIClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    # -- transport -------------------------------------------------------

    def _send(self, method: str, path: str, json: Any = None) -> tuple[int, Any, Mapping[str, str]]:
        """Send with retries; returns (status, parsed body, headers) without raising."""
        url = f"{self.base_url}{path}"
        # One key per logical call, reused across its retries: the API applies a key once.
        headers = self._headers(idempotency_key=str(uuid.uuid4()) if method.upper() == "POST" else None)
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._http.request(method, url, json=json, headers=headers)
            except httpx.TransportError as exc:
                if self.retry.should_retry_connection_error(attempt):
                    time.sleep(self.retry.delay(attempt))
                    continue
                raise APIConnectionError(f"{method} {path}: could not reach the API after {attempt} "
                                         f"attempt(s): {type(exc).__name__}: {exc}") from exc
            body = _json_or_none(response)
            retry_after = response.headers.get("retry-after")
            if self.retry.should_retry(response.status_code, attempt, retry_after):
                time.sleep(self.retry.delay(attempt, retry_after))
                continue
            return response.status_code, body, response.headers

    def _request(self, method: str, path: str, json: Any = None) -> Any:
        status, body, headers = self._send(method, path, json=json)
        self._raise_for_status(status, body, headers)
        return body

    # -- API -------------------------------------------------------------

    def evaluate(
        self,
        context: Any,
        queries: Mapping[str, Query],
        backend: Optional[BackendLike] = None,
        response_model: Optional[type[EvaluateResponse]] = None,
    ) -> EvaluateResponse:
        """Evaluate one context against one or more guarantee-bearing queries."""
        payload = self._build_evaluate_payload(context, queries, backend)
        status, body, headers = self._send("POST", "/evaluate", json=payload)
        # A 424 still carries heuristic-labelled answers: they are returned
        # unless strict_guarantees is on, in which case this raises.
        self._check_evaluate_status(status, body, headers)
        return self._parse_evaluate(body or {}, response_model, headers.get("x-request-id"))

    def list_backends(self) -> list[dict[str, Any]]:
        """Configured backend connections and their detected access levels."""
        return self._request("GET", "/backends").get("backends", [])


def _json_or_none(response: httpx.Response) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        return {"message": response.text}
