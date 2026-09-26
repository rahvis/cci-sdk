"""Shared fixtures for the client-side tests.

Every HTTP test runs against ``MockAPI``, an ``httpx.MockTransport`` that
records each request and replies from a scripted queue (or a per-route
table), injected through ``CLIClient(http_client=httpx.Client(transport=...))``
and ``AsyncCLIClient(http_client=httpx.AsyncClient(transport=...))``. No test
touches the network, and no test ever really sleeps: the ``sleeps`` fixture
is autouse and replaces the retry loops' sleep functions with recorders.
"""

from __future__ import annotations

import types
from typing import Any

import httpx
import pytest

from cli_sdk.client import AsyncCLIClient, CLIClient
from support import API_KEY, MockAPI


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the developer's real CLI_API_KEY / CLI_BASE_URL leak into a test."""
    monkeypatch.delenv("CLI_API_KEY", raising=False)
    monkeypatch.delenv("CLI_BASE_URL", raising=False)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry back-off delays instead of sleeping (sync and async)."""
    import cli_sdk.client.async_client as async_module
    import cli_sdk.client.sync_client as sync_module

    calls: list[float] = []

    async def fake_async_sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr(sync_module, "time", types.SimpleNamespace(sleep=calls.append))
    monkeypatch.setattr(async_module, "asyncio", types.SimpleNamespace(sleep=fake_async_sleep))
    return calls


@pytest.fixture
def api() -> MockAPI:
    return MockAPI()


@pytest.fixture
def make_client(api: MockAPI):
    """Factory for a sync client wired to ``api``; extra kwargs go to CLIClient."""
    created: list[CLIClient] = []

    def factory(**kwargs: Any) -> CLIClient:
        kwargs.setdefault("api_key", API_KEY)
        client = CLIClient(http_client=httpx.Client(transport=api.transport()), **kwargs)
        created.append(client)
        return client

    yield factory
    for client in created:
        client._http.close()


@pytest.fixture
def client(make_client) -> CLIClient:
    return make_client()


@pytest.fixture
def make_async_client(api: MockAPI):
    """Factory for an async client wired to ``api``; extra kwargs go to AsyncCLIClient."""

    def factory(**kwargs: Any) -> AsyncCLIClient:
        kwargs.setdefault("api_key", API_KEY)
        return AsyncCLIClient(http_client=httpx.AsyncClient(transport=api.transport()), **kwargs)

    return factory
