"""Asynchronous client."""

from __future__ import annotations

import uuid

import asyncio
from typing import Any, Mapping, Optional

import httpx

from cli_sdk.exceptions import APIConnectionError

from cli_sdk.answers.response import EvaluateResponse
from cli_sdk.backends.base import BackendLike
from cli_sdk.calibration.profile import AsyncCalibrationProfiles
from cli_sdk.client.base import BaseClient
from cli_sdk.client.retries import RetryConfig
from cli_sdk.client.sync_client import _json_or_none
from cli_sdk.queries.base import Query


class AsyncCLIClient(BaseClient):
    """Asynchronous client for Conformal Logit Inference.

    Same interface as ``CLIClient``, with ``await`` on every call::

        async with AsyncCLIClient() as client:
            result = await client.evaluate(context=..., backend=..., queries=...)

    Note: a ``CustomBackend`` computes evidence by calling your model's
    methods synchronously before the request is sent. Run slow custom
    backends in a thread (``asyncio.to_thread``) if they would block the
    event loop.
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
        http_client: Optional[httpx.AsyncClient] = None,
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
        self._http = http_client or httpx.AsyncClient(timeout=timeout)
        self.calibration_profiles = AsyncCalibrationProfiles(self)

    async def __aenter__(self) -> "AsyncCLIClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def _send(self, method: str, path: str, json: Any = None) -> tuple[int, Any, Mapping[str, str]]:
        url = f"{self.base_url}{path}"
        # One key per logical call, reused across its retries: the API applies a key once.
        headers = self._headers(idempotency_key=str(uuid.uuid4()) if method.upper() == "POST" else None)
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._http.request(method, url, json=json, headers=headers)
            except httpx.TransportError as exc:
                if self.retry.should_retry_connection_error(attempt):
                    await asyncio.sleep(self.retry.delay(attempt))
                    continue
                raise APIConnectionError(f"{method} {path}: could not reach the API after {attempt} "
                                         f"attempt(s): {type(exc).__name__}: {exc}") from exc
            body = _json_or_none(response)
            retry_after = response.headers.get("retry-after")
            if self.retry.should_retry(response.status_code, attempt, retry_after):
                await asyncio.sleep(self.retry.delay(attempt, retry_after))
                continue
            return response.status_code, body, response.headers

    async def _request(self, method: str, path: str, json: Any = None) -> Any:
        status, body, headers = await self._send(method, path, json=json)
        self._raise_for_status(status, body, headers)
        return body

    async def evaluate(
        self,
        context: Any,
        queries: Mapping[str, Query],
        backend: Optional[BackendLike] = None,
        response_model: Optional[type[EvaluateResponse]] = None,
    ) -> EvaluateResponse:
        payload = self._build_evaluate_payload(context, queries, backend)
        status, body, headers = await self._send("POST", "/evaluate", json=payload)
        self._check_evaluate_status(status, body, headers)
        return self._parse_evaluate(body or {}, response_model, headers.get("x-request-id"))

    async def list_backends(self) -> list[dict[str, Any]]:
        return (await self._request("GET", "/backends")).get("backends", [])
