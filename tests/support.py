"""Test support: the scripted ``MockAPI`` transport and canned API payloads.

Imported by the test modules directly (``from support import ...``); the
pytest fixtures that wire it into clients live in ``conftest.py``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from cli_sdk.constants import DEFAULT_BASE_URL

API_KEY = "sk-test-123"
BASE_URL = DEFAULT_BASE_URL
PROFILE = "support-routing-v3"


class MockAPI:
    """A scripted fake of the CLI HTTP API.

    ``reply(...)`` queues responses consumed in order; ``route(...)`` pins a
    response to a (method, path) pair regardless of order. When neither
    matches, the transport answers 500 so a missing script fails loudly.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._queue: list[tuple[int, Any, dict[str, str], Optional[bytes]]] = []
        self._routes: dict[tuple[str, str], tuple[int, Any, dict[str, str], Optional[bytes]]] = {}

    # -- scripting -------------------------------------------------------

    def reply(
        self,
        status: int = 200,
        json_body: Any = None,
        headers: Optional[dict[str, str]] = None,
        content: Optional[bytes] = None,
    ) -> "MockAPI":
        self._queue.append((status, json_body, dict(headers or {}), content))
        return self

    def route(self, method: str, path: str, json_body: Any = None, status: int = 200) -> "MockAPI":
        self._routes[(method, path)] = (status, json_body, {}, None)
        return self

    # -- transport -------------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        if key in self._routes:
            spec = self._routes[key]
        elif self._queue:
            spec = self._queue.pop(0)
        else:
            return httpx.Response(500, json={"message": f"unscripted request {request.method} {request.url}"})
        status, body, headers, content = spec
        if content is not None:
            return httpx.Response(status, content=content, headers=headers)
        if body is None:
            return httpx.Response(status, headers=headers)
        return httpx.Response(status, json=body, headers=headers)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    # -- inspection ------------------------------------------------------

    @property
    def last(self) -> httpx.Request:
        assert self.requests, "no request was sent"
        return self.requests[-1]

    def body(self, index: int = -1) -> Any:
        return json.loads(self.requests[index].content)


# -- canned answer payloads (shapes from apps/docs/pages/primitives/*.mdx) --


def coverage_guarantee(**overrides: Any) -> dict[str, Any]:
    card = {
        "type": "coverage",
        "alpha": 0.10,
        "method": "APS",
        "calibration_profile": PROFILE,
        "calibration_n": 1204,
        "coverage_ci": [0.884, 0.915],
        "last_audited": "2026-09-18T00:00:00Z",
    }
    card.update(overrides)
    return card


def set_answer(**overrides: Any) -> dict[str, Any]:
    answer = {
        "type": "set",
        "set": ["billing", "technical"],
        "probabilities": {"billing": 0.61, "technical": 0.33, "sales": 0.06},
        "guarantee": coverage_guarantee(),
    }
    answer.update(overrides)
    return answer


def heuristic_set_answer() -> dict[str, Any]:
    return set_answer(guarantee={"type": "heuristic", "calibration_profile": PROFILE, "calibration_n": 4})


def evaluate_body(answers: dict[str, Any], **extra: Any) -> dict[str, Any]:
    body = {
        "backend": {"provider": "openai", "model": "gpt-4.1-2025-04-14"},
        "answers": answers,
        "usage": {"backend_calls": 1, "backend_tokens": 296},
    }
    body.update(extra)
    return body
