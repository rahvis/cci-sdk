"""The sync and async clients against a mocked ``POST /v1/evaluate``.

Covers the exact request (URL, method, headers, flat JSON body), client
configuration from arguments and environment variables, backend
serialization (remote adapters and client-side ``CustomBackend`` evidence),
the retry policy, HTTP error mapping, and ``424`` heuristic fallbacks.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import cli_sdk
from cli_sdk import (
    AnthropicBackend,
    AsyncCLIClient,
    AuthenticationError,
    AzureOpenAIBackend,
    BackendError,
    BedrockBackend,
    Belief,
    CLIClient,
    CLIError,
    Claim,
    ConfigurationError,
    CustomBackend,
    EvaluateResponse,
    Gate,
    GeminiBackend,
    InsufficientCalibrationError,
    Interval,
    Judge,
    OpenAIBackend,
    OpenRouterBackend,
    RateLimitError,
    RetryConfig,
    Route,
    Set,
    SetAnswer,
    SGLangBackend,
    ValidationError,
    VLLMBackend,
)
from cli_sdk.backends import access_rank, backend_payload
from cli_sdk.client import NO_RETRY
from cli_sdk.constants import USER_AGENT
from support import (
    API_KEY,
    BASE_URL,
    PROFILE,
    MockAPI,
    evaluate_body,
    heuristic_set_answer,
    set_answer,
)

CONTEXT = {"ticket": "My payouts have been failing for 3 days."}
OPENAI = {"provider": "openai", "model": "gpt-4.1-2025-04-14"}
ROUTE_CASCADE = [
    {"backend": {"provider": "vllm", "model": "llama-3.3-70b", "base_url": "http://internal-vllm:8000"}},
    {"backend": {"provider": "openai", "model": "gpt-4.1"}},
]


def department_query(**overrides: Any) -> Set:
    kwargs = dict(
        instructions="Which team should handle this ticket?",
        options={"billing": None, "technical": None, "sales": None},
        calibration_profile=PROFILE,
        alpha=0.1,
        method="APS",
    )
    kwargs.update(overrides)
    return Set(**kwargs)


def route_query() -> Route:
    return Route(cascade=ROUTE_CASCADE, calibration_profile="cost-routing-v1", target_cents=0.4, alpha=0.1)


def route_answer() -> dict:
    return {
        "type": "route",
        "served_by": {"provider": "vllm", "model": "llama-3.3-70b"},
        "escalated": False,
        "cost_cents": 0.03,
        "guarantee": {
            "type": "cost_budget",
            "target_cents": 0.4,
            "alpha": 0.1,
            "calibration_profile": "cost-routing-v1",
        },
    }


# -- the request -------------------------------------------------------------


class TestEvaluateRequest:
    def test_exact_url_method_headers_and_body(self, api: MockAPI, client: CLIClient):
        api.reply(200, evaluate_body({"department": set_answer()}), headers={"x-request-id": "req_123"})
        result = client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})

        request = api.last
        assert request.method == "POST"
        assert str(request.url) == "https://cci.gitdate.ink/api/v1/evaluate"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        assert request.headers["user-agent"] == USER_AGENT == f"cci-sdk-python/{cli_sdk.__version__}"
        assert request.headers["content-type"] == "application/json"
        assert api.body() == {
            "context": CONTEXT,
            "backend": {"provider": "openai", "model": "gpt-4.1-2025-04-14"},
            "queries": {
                "department": {
                    "type": "set",
                    "instructions": "Which team should handle this ticket?",
                    "options": {"billing": None, "technical": None, "sales": None},
                    "calibration_profile": PROFILE,
                    "alpha": 0.1,
                    "method": "APS",
                }
            },
        }
        assert len(api.requests) == 1
        assert result.request_id == "req_123"
        assert isinstance(result.answers["department"], SetAnswer)

    def test_no_nested_calibration_object_anywhere(self, api: MockAPI, client: CLIClient):
        api.reply(200, evaluate_body({}))
        client.evaluate(
            context=CONTEXT,
            backend=OPENAI,
            queries={
                "department": department_query(),
                "urgent": Belief(instructions="Urgent?", calibration_profile="urgency-v1"),
                "route": Gate(
                    instructions="Auto-route?", calibration_profile=PROFILE, guarantee="fdr", target=0.05
                ),
            },
        )
        raw = api.last.content.decode()
        assert '"calibration"' not in raw
        for query in api.body()["queries"].values():
            assert "calibration" not in query
            assert query["calibration_profile"]

    def test_body_is_exactly_query_to_payload(self, api: MockAPI, client: CLIClient):
        queries = {
            "department": department_query(),
            "frustration": Interval(
                instructions="How frustrated?", levels=["calm", "angry"], calibration_profile="f-v1"
            ),
            "facts": Claim(
                instructions="Answer from the passages.", calibration_profile="rag-v1", alpha=0.05
            ),
        }
        api.reply(200, evaluate_body({}))
        client.evaluate(context="plain string context", backend=OPENAI, queries=queries)
        body = api.body()
        assert body["context"] == "plain string context"
        assert body["queries"] == {qid: q.to_payload() for qid, q in queries.items()}

    def test_remote_backend_objects_serialize(self, api: MockAPI, client: CLIClient):
        api.reply(200, evaluate_body({}))
        client.evaluate(
            context=CONTEXT,
            backend=OpenAIBackend(model="gpt-4.1-2025-04-14", reasoning_effort="none"),
            queries={"department": department_query()},
        )
        assert api.body()["backend"] == {
            "provider": "openai",
            "model": "gpt-4.1-2025-04-14",
            "access_hint": "auto",
            "reasoning_effort": "none",
        }

    def test_default_backend_from_constructor(self, api: MockAPI, make_client):
        client = make_client(backend=OPENAI)
        api.reply(200, evaluate_body({}))
        client.evaluate(context=CONTEXT, queries={"department": department_query()})
        assert api.body()["backend"] == OPENAI

    def test_per_call_backend_overrides_the_default(self, api: MockAPI, make_client):
        client = make_client(backend=OPENAI)
        api.reply(200, evaluate_body({}))
        client.evaluate(
            context=CONTEXT,
            backend={"provider": "anthropic", "model": "claude-sonnet-4-5"},
            queries={"department": department_query()},
        )
        assert api.body()["backend"] == {"provider": "anthropic", "model": "claude-sonnet-4-5"}

    def test_missing_backend_is_a_configuration_error_before_any_request(
        self, api: MockAPI, client: CLIClient
    ):
        with pytest.raises(ConfigurationError, match="no backend"):
            client.evaluate(context=CONTEXT, queries={"department": department_query()})
        assert api.requests == []

    def test_route_needs_no_backend(self, api: MockAPI, client: CLIClient):
        api.reply(200, evaluate_body({"answer": route_answer()}))
        result = client.evaluate(context={"query": "Where is my order?"}, queries={"answer": route_query()})
        body = api.body()
        assert "backend" not in body
        assert body["queries"]["answer"]["cascade"] == ROUTE_CASCADE
        assert result.answers["answer"].served_by["provider"] == "vllm"

    def test_judge_with_a_cascade_needs_no_backend(self, api: MockAPI, client: CLIClient):
        # apps/docs/pages/api-reference.mdx: `backend` is "required unless
        # every query specifies `cascade`".
        judge = Judge(
            instructions="Which response is better?",
            calibration_profile="pairwise-judge-v2",
            cascade=[
                {"backend": {"provider": "openai", "model": "gpt-4.1-mini"}},
                {"backend": "human_queue"},
            ],
        )
        api.reply(200, evaluate_body({}))
        client.evaluate(
            context={"prompt": "p", "response_a": "a", "response_b": "b"}, queries={"verdict": judge}
        )
        assert "backend" not in api.body()

    def test_judge_without_a_cascade_still_needs_a_backend(self, client: CLIClient):
        judge = Judge(instructions="Which response is better?", calibration_profile="pairwise-judge-v2")
        with pytest.raises(ConfigurationError, match="no backend"):
            client.evaluate(context={}, queries={"verdict": judge})

    def test_route_mixed_with_other_queries_still_needs_a_backend(self, api: MockAPI, client: CLIClient):
        with pytest.raises(ConfigurationError, match="no backend"):
            client.evaluate(
                context=CONTEXT, queries={"answer": route_query(), "department": department_query()}
            )
        assert api.requests == []

    def test_empty_queries_rejected(self, client: CLIClient):
        with pytest.raises(ConfigurationError, match="at least one query"):
            client.evaluate(context=CONTEXT, backend=OPENAI, queries={})

    def test_non_query_values_rejected(self, api: MockAPI, client: CLIClient):
        with pytest.raises(ConfigurationError, match="'department'"):
            client.evaluate(
                context=CONTEXT, backend=OPENAI, queries={"department": department_query().to_payload()}
            )
        assert api.requests == []

    def test_backend_dict_requires_provider(self, client: CLIClient):
        with pytest.raises(ConfigurationError, match="provider"):
            client.evaluate(
                context=CONTEXT, backend={"model": "gpt-4.1"}, queries={"department": department_query()}
            )

    def test_unsupported_backend_type(self, client: CLIClient):
        with pytest.raises(ConfigurationError, match="unsupported backend"):
            client.evaluate(context=CONTEXT, backend="openai", queries={"department": department_query()})

    def test_default_headers_are_merged(self, api: MockAPI, make_client):
        client = make_client(default_headers={"X-Team": "support"})
        api.reply(200, evaluate_body({}))
        client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert api.last.headers["x-team"] == "support"
        assert api.last.headers["authorization"] == f"Bearer {API_KEY}"

    def test_invalid_response_model(self, api: MockAPI, client: CLIClient):
        api.reply(200, evaluate_body({"department": set_answer()}))
        with pytest.raises(ConfigurationError, match="response_model"):
            client.evaluate(
                context=CONTEXT,
                backend=OPENAI,
                queries={"department": department_query()},
                response_model=dict,
            )

    def test_typed_response_model(self, api: MockAPI, client: CLIClient):
        class RoutingResponse(EvaluateResponse):
            department: SetAnswer

        api.reply(200, evaluate_body({"department": set_answer()}))
        result = client.evaluate(
            context=CONTEXT,
            backend=OPENAI,
            queries={"department": department_query()},
            response_model=RoutingResponse,
        )
        assert isinstance(result, RoutingResponse)
        assert result.department.set == ["billing", "technical"]

    def test_list_backends(self, api: MockAPI, client: CLIClient):
        api.reply(200, {"backends": [{"provider": "openai", "model": "gpt-4.1", "access_level": "L1"}]})
        assert client.list_backends() == [{"provider": "openai", "model": "gpt-4.1", "access_level": "L1"}]
        assert api.last.method == "GET" and str(api.last.url) == f"{BASE_URL}/backends"


# -- configuration -------------------------------------------------------------


class TestConfiguration:
    def _mock(self) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"backends": []}))
        )

    def test_api_key_from_environment(self, monkeypatch: pytest.MonkeyPatch, api: MockAPI):
        monkeypatch.setenv("CLI_API_KEY", "sk-from-env")
        client = CLIClient(http_client=httpx.Client(transport=api.transport()))
        api.reply(200, {"backends": []})
        client.list_backends()
        assert api.last.headers["authorization"] == "Bearer sk-from-env"

    def test_explicit_api_key_beats_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLI_API_KEY", "sk-from-env")
        assert CLIClient(api_key="sk-explicit", http_client=self._mock()).api_key == "sk-explicit"

    def test_base_url_from_environment(self, monkeypatch: pytest.MonkeyPatch, api: MockAPI):
        monkeypatch.setenv("CLI_BASE_URL", "https://cli.internal.example.com/v1/")
        client = CLIClient(api_key=API_KEY, http_client=httpx.Client(transport=api.transport()))
        assert client.base_url == "https://cli.internal.example.com/v1"
        api.reply(200, evaluate_body({}))
        client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert str(api.last.url) == "https://cli.internal.example.com/v1/evaluate"

    def test_explicit_base_url_beats_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLI_BASE_URL", "https://env.example.com/v1")
        client = CLIClient(api_key=API_KEY, base_url="https://arg.example.com/v1", http_client=self._mock())
        assert client.base_url == "https://arg.example.com/v1"

    def test_default_base_url(self):
        assert CLIClient(api_key=API_KEY, http_client=self._mock()).base_url == BASE_URL

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://localhost:8080/v1",
            "http://127.0.0.1:8080/v1",
            "http://0.0.0.0:9000",
            "http://[::1]:8080/v1",
            "http://LOCALHOST/v1",
        ],
    )
    def test_no_key_needed_for_a_local_deployment(self, api: MockAPI, base_url: str):
        client = CLIClient(base_url=base_url, http_client=httpx.Client(transport=api.transport()))
        assert client.api_key is None
        api.reply(200, {"backends": []})
        client.list_backends()
        assert "authorization" not in api.last.headers

    def test_no_key_needed_when_env_base_url_is_local(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CLI_BASE_URL", "http://localhost:8080/v1")
        assert CLIClient(http_client=self._mock()).api_key is None

    @pytest.mark.parametrize(
        "base_url",
        [
            None,
            "https://cci.gitdate.ink/api/v1",
            "https://localhost.attacker.example/v1",
            "https://cli.example.com/v1?next=http://localhost",
            "https://127.0.0.1.nip.io/v1",
        ],
    )
    def test_key_required_for_any_non_local_host(self, base_url):
        with pytest.raises(ConfigurationError, match="CLI_API_KEY"):
            CLIClient(base_url=base_url, http_client=self._mock())

    def test_unparsable_base_url_is_not_treated_as_local(self):
        with pytest.raises(ConfigurationError, match="CLI_API_KEY"):
            CLIClient(base_url="http://[::1", http_client=self._mock())

    def test_async_client_has_the_same_key_rule(self):
        with pytest.raises(ConfigurationError, match="CLI_API_KEY"):
            AsyncCLIClient()

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_an_owned_http_client(self):
        async with AsyncCLIClient(api_key=API_KEY) as client:
            inner = client._http
        assert inner.is_closed

    def test_context_manager_closes_only_an_owned_http_client(self):
        injected = self._mock()
        with CLIClient(api_key=API_KEY, http_client=injected):
            pass
        assert not injected.is_closed
        with CLIClient(api_key=API_KEY) as owned:
            inner = owned._http
        assert inner.is_closed
        injected.close()

    def test_lazy_exports(self):
        assert cli_sdk.CLIClient is CLIClient
        assert cli_sdk.RetryConfig is RetryConfig
        with pytest.raises(AttributeError):
            cli_sdk.NotAThing  # noqa: B018


# -- remote backend adapters ------------------------------------------------------


class TestRemoteBackends:
    @pytest.mark.parametrize(
        "backend, expected",
        [
            (
                OpenAIBackend(model="gpt-4.1-2025-04-14"),
                {"provider": "openai", "model": "gpt-4.1-2025-04-14", "access_hint": "auto"},
            ),
            (
                AzureOpenAIBackend(model="gpt-4.1", deployment="prod-gpt41", api_version="2025-04-01"),
                {
                    "provider": "azure-openai",
                    "model": "gpt-4.1",
                    "access_hint": "auto",
                    "deployment": "prod-gpt41",
                    "api_version": "2025-04-01",
                },
            ),
            (
                AnthropicBackend(model="claude-sonnet-4-5", sample_count=10),
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "access_hint": "auto",
                    "sample_count": 10,
                },
            ),
            (
                GeminiBackend(model="gemini-2.5-pro", thinking_level="low"),
                {
                    "provider": "gemini",
                    "model": "gemini-2.5-pro",
                    "access_hint": "auto",
                    "thinking_level": "low",
                },
            ),
            (
                BedrockBackend(model="arn:model", region="us-east-1", custom_model_import=True),
                {
                    "provider": "bedrock",
                    "model": "arn:model",
                    "access_hint": "auto",
                    "region": "us-east-1",
                    "custom_model_import": True,
                },
            ),
            (
                OpenRouterBackend(model="meta-llama/llama-3.3-70b", upstream="together"),
                {
                    "provider": "openrouter",
                    "model": "meta-llama/llama-3.3-70b",
                    "access_hint": "auto",
                    "pin_upstream": True,
                    "upstream": "together",
                },
            ),
            (
                VLLMBackend(model="llama-3.3-70b", base_url="http://vllm:8000", engine_version="0.11.0"),
                {
                    "provider": "vllm",
                    "model": "llama-3.3-70b",
                    "access_hint": "auto",
                    "base_url": "http://vllm:8000",
                    "engine_version": "0.11.0",
                },
            ),
            (
                SGLangBackend(model="qwen3-32b", base_url="http://sglang:30000", access_hint="exact"),
                {
                    "provider": "sglang",
                    "model": "qwen3-32b",
                    "access_hint": "exact",
                    "base_url": "http://sglang:30000",
                },
            ),
        ],
        ids=lambda v: getattr(v, "provider", None) or "expected",
    )
    def test_payloads_omit_unset_fields(self, backend, expected):
        assert backend_payload(backend) == expected

    def test_connection_name_is_sent_when_set(self):
        assert (
            OpenAIBackend(model="gpt-4.1", connection="prod-openai").to_payload()["connection"]
            == "prod-openai"
        )

    @pytest.mark.parametrize("cls", [OpenAIBackend, AnthropicBackend, GeminiBackend, OpenRouterBackend])
    def test_model_required(self, cls):
        with pytest.raises(ConfigurationError, match="model"):
            cls()

    @pytest.mark.parametrize("cls", [VLLMBackend, SGLangBackend])
    def test_self_hosted_engines_require_base_url(self, cls):
        with pytest.raises(ConfigurationError, match="base_url"):
            cls(model="llama-3.3-70b")

    def test_access_hint_validated(self):
        with pytest.raises(ConfigurationError, match="access_hint"):
            OpenAIBackend(model="gpt-4.1", access_hint="telepathy")

    def test_access_rank(self):
        assert [access_rank(level) for level in ("L0", "L1", "L2", "L3", "L4")] == [0, 1, 2, 3, 4]
        with pytest.raises(ConfigurationError, match="L7"):
            access_rank("L7")


# -- CustomBackend: evidence computed client-side -------------------------------------


class RecordingBackend(CustomBackend):
    """A fake local model that records every evidence call it receives."""

    name = "recording-engine"

    def __init__(self, level: str = "L0", **kwargs: Any) -> None:
        self.access_level = level
        super().__init__(**kwargs)
        self.calls: list[tuple[str, Any, Any, Any]] = []

    def sample(self, context, instructions, n):
        self.calls.append(("sample", context, instructions, n))
        return [f"sample-{i}" for i in range(n)]

    def score_options(self, context, instructions, options):
        self.calls.append(("score_options", context, instructions, dict(options)))
        return {key: round(1.0 / len(options), 4) for key in options}

    def score_text(self, context, instructions, candidate):
        self.calls.append(("score_text", context, instructions, candidate))
        return 0.5


class HiddenStateBackend(RecordingBackend):
    def hidden_states(self, context, instructions):
        self.calls.append(("hidden_states", context, instructions, None))
        return [0.1, -0.2, 0.3]


class TestCustomBackend:
    def _evaluate(self, api: MockAPI, client: CLIClient, backend: CustomBackend, queries: dict) -> dict:
        api.reply(200, evaluate_body({}))
        client.evaluate(context=CONTEXT, backend=backend, queries=queries)
        return api.body()

    def test_backend_serializes_as_client_evidence(self, api: MockAPI, make_client):
        client = make_client()
        backend = RecordingBackend("L0", base_url="http://localhost:8000", api_token="secret-local-token")
        body = self._evaluate(api, client, backend, {"department": department_query()})
        assert body["backend"] == {
            "provider": "client_evidence",
            "name": "recording-engine",
            "access_level": "L0",
        }
        raw = api.last.content.decode()
        assert "secret-local-token" not in raw and "localhost:8000" not in raw

    def test_l0_sends_samples(self, api: MockAPI, make_client):
        client = make_client(sample_count=5)
        backend = RecordingBackend("L0")
        body = self._evaluate(api, client, backend, {"department": department_query()})
        query = body["queries"]["department"]
        assert query["evidence"] == {"samples": [f"sample-{i}" for i in range(5)]}
        assert backend.calls == [("sample", CONTEXT, "Which team should handle this ticket?", 5)]
        # everything else in the query is still the flat payload
        assert {k: v for k, v in query.items() if k != "evidence"} == department_query().to_payload()

    def test_default_sample_count_is_twenty(self, api: MockAPI, client: CLIClient):
        body = self._evaluate(api, client, RecordingBackend("L0"), {"department": department_query()})
        assert len(body["queries"]["department"]["evidence"]["samples"]) == 20

    def test_l1_set_scores_every_option(self, api: MockAPI, client: CLIClient):
        backend = RecordingBackend("L1")
        body = self._evaluate(api, client, backend, {"department": department_query()})
        assert backend.calls == [
            (
                "score_options",
                CONTEXT,
                "Which team should handle this ticket?",
                {"billing": None, "technical": None, "sales": None},
            ),
        ]
        assert body["queries"]["department"]["evidence"] == {
            "option_probabilities": {"billing": 0.3333, "technical": 0.3333, "sales": 0.3333}
        }

    @pytest.mark.parametrize(
        "query, options",
        [
            (Belief(instructions="Urgent?", calibration_profile="urgency-v1"), {"true": None, "false": None}),
            (
                Gate(instructions="Approve?", calibration_profile="p", target=0.05),
                {"true": None, "false": None},
            ),
            (
                Judge(instructions="Which is better?", calibration_profile="j"),
                {"response_a": None, "response_b": None},
            ),
        ],
        ids=["belief", "gate", "judge"],
    )
    def test_l1_binary_and_pairwise_options(self, api: MockAPI, client: CLIClient, query, options):
        backend = RecordingBackend("L2")
        body = self._evaluate(api, client, backend, {"q": query})
        assert backend.calls[0][0] == "score_options" and backend.calls[0][3] == options
        assert set(body["queries"]["q"]["evidence"]["option_probabilities"]) == set(options)

    def test_l4_adds_hidden_states(self, api: MockAPI, client: CLIClient):
        backend = HiddenStateBackend("L4")
        body = self._evaluate(api, client, backend, {"department": department_query()})
        evidence = body["queries"]["department"]["evidence"]
        assert evidence["hidden_states"] == [0.1, -0.2, 0.3]
        assert set(evidence["option_probabilities"]) == {"billing", "technical", "sales"}

    def test_l4_without_hidden_states_still_sends_probabilities(self, api: MockAPI, client: CLIClient):
        body = self._evaluate(api, client, RecordingBackend("L4"), {"department": department_query()})
        evidence = body["queries"]["department"]["evidence"]
        assert "hidden_states" not in evidence
        assert "option_probabilities" in evidence

    def test_below_l4_never_asks_for_hidden_states(self, api: MockAPI, client: CLIClient):
        backend = HiddenStateBackend("L3")
        body = self._evaluate(api, client, backend, {"department": department_query()})
        assert "hidden_states" not in body["queries"]["department"]["evidence"]
        assert [c[0] for c in backend.calls] == ["score_options"]

    def test_interval_level_probabilities(self, api: MockAPI, client: CLIClient):
        backend = RecordingBackend("L1")
        interval = Interval(
            instructions="How frustrated?", levels=["calm", "annoyed", "furious"], calibration_profile="f"
        )
        body = self._evaluate(api, client, backend, {"frustration": interval})
        assert backend.calls[0][3] == {"0": "calm", "1": "annoyed", "2": "furious"}
        assert set(body["queries"]["frustration"]["evidence"]["level_probabilities"]) == {"0", "1", "2"}

    def test_interval_at_l0_samples(self, api: MockAPI, make_client):
        client = make_client(sample_count=3)
        interval = Interval(
            instructions="How frustrated?", levels=["calm", "furious"], calibration_profile="f"
        )
        body = self._evaluate(api, client, RecordingBackend("L0"), {"frustration": interval})
        assert body["queries"]["frustration"]["evidence"] == {"samples": ["sample-0", "sample-1", "sample-2"]}

    def test_claim_evidence(self, api: MockAPI, make_client):
        client = make_client(sample_count=2)
        claim = Claim(instructions="Answer from the passages.", calibration_profile="rag-v1")
        body_l0 = self._evaluate(api, client, RecordingBackend("L0"), {"facts": claim})
        assert body_l0["queries"]["facts"]["evidence"] == {"samples": ["sample-0", "sample-1"]}
        body_l2 = self._evaluate(api, client, RecordingBackend("L2"), {"facts": claim})
        assert body_l2["queries"]["facts"]["evidence"] == {
            "samples": ["sample-0", "sample-1"],
            "claim_scoring": "client",
        }

    def test_route_gets_no_evidence(self, api: MockAPI, client: CLIClient):
        backend = RecordingBackend("L1")
        body = self._evaluate(
            api, client, backend, {"answer": route_query(), "department": department_query()}
        )
        assert "evidence" not in body["queries"]["answer"]
        assert "evidence" in body["queries"]["department"]
        assert len(backend.calls) == 1

    def test_missing_method_surfaces_a_clear_not_implemented_error(self, api: MockAPI, client: CLIClient):
        class SamplingOnly(CustomBackend):
            access_level = "L1"  # claims logprobs, but never implemented score_options

            def sample(self, context, instructions, n):
                return ["billing"] * n

        with pytest.raises(NotImplementedError) as excinfo:
            client.evaluate(
                context=CONTEXT, backend=SamplingOnly(), queries={"department": department_query()}
            )
        message = str(excinfo.value)
        assert "score_options" in message
        assert "L1" in message
        assert "set" in message
        assert api.requests == []  # the guarantee is never silently downgraded to a sampling request

    def test_l0_backend_without_sample(self, api: MockAPI, client: CLIClient):
        with pytest.raises(NotImplementedError, match="sample"):
            client.evaluate(
                context=CONTEXT, backend=CustomBackend(), queries={"department": department_query()}
            )
        assert api.requests == []

    def test_invalid_access_level_rejected_at_construction(self):
        class Broken(CustomBackend):
            access_level = "L9"

        with pytest.raises(ConfigurationError, match="L9"):
            Broken()

    def test_reexported_from_backends_custom(self):
        from cli_sdk.backends.custom import CustomBackend as Reexported

        assert Reexported is CustomBackend

    def test_unknown_query_type_is_a_configuration_error(self, api: MockAPI, client: CLIClient):
        from dataclasses import dataclass, field

        from cli_sdk import Query

        @dataclass
        class Forecast(Query):
            type: str = field(default="forecast", init=False)
            instructions: str = "Next week's volume?"

        with pytest.raises(ConfigurationError, match="'forecast'"):
            client.evaluate(context=CONTEXT, backend=RecordingBackend("L1"), queries={"f": Forecast()})
        assert api.requests == []

    def test_default_backend_can_be_custom(self, api: MockAPI, make_client):
        backend = RecordingBackend("L1")
        client = make_client(backend=backend)
        api.reply(200, evaluate_body({}))
        client.evaluate(context=CONTEXT, queries={"department": department_query()})
        assert api.body()["backend"]["provider"] == "client_evidence"
        assert "option_probabilities" in api.body()["queries"]["department"]["evidence"]

    def test_evidence_does_not_leak_between_calls(self, api: MockAPI, client: CLIClient):
        query = department_query()
        self._evaluate(api, client, RecordingBackend("L0"), {"department": query})
        assert "evidence" not in query.to_payload()
        api.reply(200, evaluate_body({}))
        client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": query})
        assert "evidence" not in api.body()["queries"]["department"]


# -- retries -----------------------------------------------------------------------


class TestRetries:
    def test_429_then_success_honors_retry_after(self, api: MockAPI, client: CLIClient, sleeps: list):
        api.reply(429, {"message": "slow down"}, headers={"retry-after": "2"})
        api.reply(200, evaluate_body({"department": set_answer()}))
        result = client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(api.requests) == 2
        assert sleeps == [2.0]
        assert api.body(0) == api.body(1)  # the retried request is identical
        assert result.answers["department"].set == ["billing", "technical"]

    @pytest.mark.parametrize("status", [429, 502, 529])
    def test_retryable_statuses(self, api: MockAPI, client: CLIClient, sleeps: list, status: int):
        api.reply(status, {"message": "transient"}, headers={"retry-after": "0.25"})
        api.reply(200, evaluate_body({}))
        client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(api.requests) == 2
        assert sleeps == [0.25]

    def test_mixed_transient_failures(self, api: MockAPI, client: CLIClient, sleeps: list):
        api.reply(502, {"message": "backend"}).reply(529, {"message": "overloaded"}).reply(
            200, evaluate_body({})
        )
        client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(api.requests) == 3
        assert len(sleeps) == 2

    def test_retry_after_is_honoured_in_full(self, api: MockAPI, make_client, sleeps: list):
        client = make_client(retry=RetryConfig(max_attempts=2, max_delay=3.0))
        api.reply(429, {}, headers={"retry-after": "20"}).reply(200, evaluate_body({}))
        client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert sleeps == [20.0]  # never retry before the time the server asked for

    def test_retry_after_beyond_the_limit_raises_instead_of_waiting(self, api: MockAPI, make_client, sleeps: list):
        client = make_client(retry=RetryConfig(max_attempts=3, max_retry_after=60.0))
        api.reply(429, {"message": "slow down"}, headers={"retry-after": "120"})
        with pytest.raises(RateLimitError) as info:
            client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert sleeps == [] and info.value.retry_after == 120.0

    def test_posts_carry_one_idempotency_key_across_retries(self, api: MockAPI, make_client, sleeps: list):
        client = make_client(retry=RetryConfig(max_attempts=2))
        api.reply(502, {"message": "backend"}).reply(200, evaluate_body({}))
        client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        keys = [r.headers.get("idempotency-key") for r in api.requests]
        assert len(keys) == 2 and keys[0] and keys[0] == keys[1]

    def test_connection_errors_are_retried_then_wrapped(self, make_client, sleeps: list):
        import httpx

        from cli_sdk import APIConnectionError, CLIClient

        calls = []

        def refuse(request):
            calls.append(request)
            raise httpx.ConnectError("connection refused", request=request)

        client = CLIClient(api_key="sk-test", http_client=httpx.Client(transport=httpx.MockTransport(refuse)),
                           retry=RetryConfig(max_attempts=3))
        with pytest.raises(APIConnectionError):
            client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(calls) == 3 and len(sleeps) == 2

    def test_backoff_without_retry_after_is_bounded(self, api: MockAPI, make_client, sleeps: list):
        client = make_client(retry=RetryConfig(max_attempts=4, base_delay=0.5, max_delay=8.0))
        for _ in range(3):
            api.reply(502, {"message": "backend"})
        api.reply(200, evaluate_body({}))
        client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(sleeps) == 3
        for attempt, delay in enumerate(sleeps, start=1):
            assert 0.0 <= delay <= 0.5 * 2 ** (attempt - 1)

    def test_http_date_retry_after_falls_back_to_backoff(self, api: MockAPI, make_client, sleeps: list):
        client = make_client(retry=RetryConfig(max_attempts=2, base_delay=0.5))
        api.reply(429, {}, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}).reply(
            200, evaluate_body({})
        )
        client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(sleeps) == 1 and 0.0 <= sleeps[0] <= 0.5

    def test_max_attempts_respected_then_rate_limit_error(self, api: MockAPI, make_client, sleeps: list):
        client = make_client(retry=RetryConfig(max_attempts=3))
        for _ in range(5):
            api.reply(429, {"message": "rate limited"}, headers={"retry-after": "1.5"})
        with pytest.raises(RateLimitError) as excinfo:
            client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(api.requests) == 3
        assert sleeps == [1.5, 1.5]
        assert excinfo.value.retry_after == 1.5
        assert "rate limited" in str(excinfo.value)

    def test_default_policy_makes_four_attempts(self, api: MockAPI, client: CLIClient, sleeps: list):
        for _ in range(6):
            api.reply(529, {"message": "overloaded"})
        with pytest.raises(BackendError):
            client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(api.requests) == 4 and len(sleeps) == 3

    def test_no_retry_policy(self, api: MockAPI, make_client, sleeps: list):
        client = make_client(retry=NO_RETRY)
        api.reply(502, {"message": "backend down"})
        with pytest.raises(BackendError, match="backend down"):
            client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(api.requests) == 1 and sleeps == []

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 424, 500])
    def test_non_retryable_statuses_are_sent_once(self, api: MockAPI, make_client, sleeps: list, status: int):
        client = make_client(strict_guarantees=True)
        api.reply(status, {"message": "nope"})
        with pytest.raises(CLIError):
            client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(api.requests) == 1 and sleeps == []

    def test_custom_retry_statuses(self, api: MockAPI, make_client, sleeps: list):
        client = make_client(retry=RetryConfig(max_attempts=2, retry_statuses=frozenset({503})))
        api.reply(503, {"message": "unavailable"}).reply(200, evaluate_body({}))
        client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(api.requests) == 2

    @pytest.mark.parametrize("kwargs", [{"max_attempts": 0}, {"base_delay": -1.0}, {"max_delay": -0.5}])
    def test_retry_config_validation(self, kwargs):
        with pytest.raises(ValueError):
            RetryConfig(**kwargs)

    def test_retry_config_delay_contract(self):
        config = RetryConfig(base_delay=1.0, max_delay=4.0)
        assert config.delay(1, "3") == 3.0
        assert config.delay(1, "-5") == 0.0
        assert config.delay(1, "99") == 60.0  # honoured in full up to max_retry_after
        assert not config.should_retry(429, 1, "120")  # beyond max_retry_after: raise, do not wait
        assert all(0.0 <= config.delay(10) <= 4.0 for _ in range(50))
        assert (
            config.should_retry(429, 1)
            and not config.should_retry(429, 4)
            and not config.should_retry(401, 1)
        )


# -- error mapping -----------------------------------------------------------------


class TestErrors:
    def _call(self, client: CLIClient):
        return client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})

    def test_401(self, api: MockAPI, client: CLIClient):
        api.reply(401, {"message": "invalid API key"})
        with pytest.raises(AuthenticationError, match="invalid API key"):
            self._call(client)

    def test_422_carries_the_field(self, api: MockAPI, client: CLIClient):
        api.reply(422, {"message": "alpha must be in (0, 1)", "field": "queries.department.alpha"})
        with pytest.raises(ValidationError) as excinfo:
            self._call(client)
        assert excinfo.value.field == "queries.department.alpha"
        assert str(excinfo.value) == "alpha must be in (0, 1)"

    def test_422_without_field(self, api: MockAPI, client: CLIClient):
        api.reply(422, {"detail": "bad request body"})
        with pytest.raises(ValidationError, match="bad request body") as excinfo:
            self._call(client)
        assert excinfo.value.field is None

    def test_429_without_retry_after(self, api: MockAPI, make_client):
        client = make_client(retry=NO_RETRY)
        api.reply(429, {"error": "too many requests"})
        with pytest.raises(RateLimitError, match="too many requests") as excinfo:
            self._call(client)
        assert excinfo.value.retry_after is None

    def test_429_with_an_http_date_retry_after(self, api: MockAPI, make_client):
        client = make_client(retry=NO_RETRY)
        api.reply(429, {"message": "slow down"}, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
        with pytest.raises(RateLimitError) as excinfo:
            self._call(client)
        assert excinfo.value.retry_after is None

    @pytest.mark.parametrize("status", [502, 529])
    def test_backend_errors(self, api: MockAPI, make_client, status):
        client = make_client(retry=NO_RETRY)
        api.reply(status, {"message": "backend returned L0, query requires L1"})
        with pytest.raises(BackendError, match="requires L1"):
            self._call(client)

    def test_other_statuses_are_plain_cli_errors(self, api: MockAPI, client: CLIClient):
        api.reply(500, {"message": "internal"})
        with pytest.raises(CLIError, match="HTTP 500: internal") as excinfo:
            self._call(client)
        assert type(excinfo.value) is CLIError

    def test_non_json_error_body(self, api: MockAPI, client: CLIClient):
        api.reply(503, content=b"<html>Service Unavailable</html>")
        with pytest.raises(CLIError, match="Service Unavailable"):
            self._call(client)

    def test_empty_error_body(self, api: MockAPI, client: CLIClient):
        api.reply(404)
        with pytest.raises(CLIError, match="status 404"):
            self._call(client)

    def test_every_sdk_error_is_a_cli_error(self):
        for exc in (
            AuthenticationError,
            ValidationError,
            InsufficientCalibrationError,
            RateLimitError,
            BackendError,
            ConfigurationError,
        ):
            assert issubclass(exc, CLIError)


# -- 424: heuristic fallback ---------------------------------------------------------


class TestInsufficientCalibration:
    def _call(self, client):
        return client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})

    def test_424_returns_labelled_heuristic_answers_by_default(self, api: MockAPI, client: CLIClient):
        api.reply(
            424, evaluate_body({"department": heuristic_set_answer()}), headers={"x-request-id": "req_424"}
        )
        result = self._call(client)
        answer = result.answers["department"]
        assert answer.is_heuristic
        assert answer.guarantee.describe() == "Heuristic value: no formal statistical guarantee."
        assert result.heuristic_answers == ["department"]
        assert result.request_id == "req_424"
        assert len(api.requests) == 1

    def test_424_raises_in_strict_mode(self, api: MockAPI, make_client):
        client = make_client(strict_guarantees=True)
        api.reply(
            424,
            {**evaluate_body({"department": heuristic_set_answer()}), "message": "profile has 4 examples"},
        )
        with pytest.raises(InsufficientCalibrationError, match="4 examples"):
            self._call(client)

    def test_424_raises_in_strict_mode_even_without_answers(self, api: MockAPI, make_client):
        client = make_client(strict_guarantees=True)
        api.reply(424, {"message": "profile 'support-routing-v3' has 4 of 9 required examples"})
        with pytest.raises(InsufficientCalibrationError, match="4 of 9"):
            self._call(client)

    def test_strict_mode_rejects_heuristic_answers_on_200(self, api: MockAPI, make_client):
        client = make_client(strict_guarantees=True)
        api.reply(200, evaluate_body({"department": heuristic_set_answer(), "other": set_answer()}))
        with pytest.raises(InsufficientCalibrationError, match="department"):
            self._call(client)

    def test_strict_mode_rejects_answers_without_a_card(self, api: MockAPI, make_client):
        client = make_client(strict_guarantees=True)
        bare = {k: v for k, v in set_answer().items() if k != "guarantee"}
        api.reply(200, evaluate_body({"department": bare}))
        with pytest.raises(InsufficientCalibrationError):
            self._call(client)

    def test_strict_mode_passes_guaranteed_answers(self, api: MockAPI, make_client):
        client = make_client(strict_guarantees=True)
        api.reply(200, evaluate_body({"department": set_answer()}))
        assert not self._call(client).answers["department"].is_heuristic

    def test_424_from_a_resource_endpoint_raises(self, api: MockAPI, client: CLIClient):
        api.reply(424, {"message": "not enough examples"})
        with pytest.raises(InsufficientCalibrationError):
            client.list_backends()


# -- async client --------------------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncClient:
    async def test_exact_request(self, api: MockAPI, make_async_client):
        api.reply(200, evaluate_body({"department": set_answer()}), headers={"x-request-id": "req_async"})
        async with make_async_client() as client:
            result = await client.evaluate(
                context=CONTEXT, backend=OPENAI, queries={"department": department_query()}
            )
        request = api.last
        assert request.method == "POST" and str(request.url) == f"{BASE_URL}/evaluate"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        assert request.headers["user-agent"] == USER_AGENT
        assert api.body()["queries"]["department"] == department_query().to_payload()
        assert result.request_id == "req_async"
        assert result.answers["department"].top == "billing"

    async def test_async_client_does_not_close_an_injected_http_client(self, api: MockAPI):
        http = httpx.AsyncClient(transport=api.transport())
        async with AsyncCLIClient(api_key=API_KEY, http_client=http):
            pass
        assert not http.is_closed
        await http.aclose()

    async def test_missing_backend(self, api: MockAPI, make_async_client):
        client = make_async_client()
        with pytest.raises(ConfigurationError, match="no backend"):
            await client.evaluate(context=CONTEXT, queries={"department": department_query()})
        assert api.requests == []

    async def test_route_without_backend_and_default_backend(self, api: MockAPI, make_async_client):
        client = make_async_client(backend=OPENAI)
        api.reply(200, evaluate_body({"answer": route_answer()})).reply(200, evaluate_body({}))
        await client.evaluate(context={}, queries={"answer": route_query()})
        # a default backend is still attached when one is configured
        assert api.body(0)["backend"] == OPENAI
        await client.evaluate(context={}, queries={"department": department_query()})
        assert api.body(1)["backend"] == OPENAI

    async def test_retries_use_async_sleep(self, api: MockAPI, make_async_client, sleeps: list):
        client = make_async_client(retry=RetryConfig(max_attempts=3))
        api.reply(429, {}, headers={"retry-after": "1"}).reply(529, {}, headers={"retry-after": "2"})
        api.reply(200, evaluate_body({}))
        await client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(api.requests) == 3 and sleeps == [1.0, 2.0]

    async def test_rate_limit_after_max_attempts(self, api: MockAPI, make_async_client, sleeps: list):
        client = make_async_client(retry=RetryConfig(max_attempts=2))
        api.reply(429, {}, headers={"retry-after": "4"}).reply(429, {}, headers={"retry-after": "4"})
        with pytest.raises(RateLimitError) as excinfo:
            await client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert excinfo.value.retry_after == 4.0 and len(api.requests) == 2 and sleeps == [4.0]

    @pytest.mark.parametrize(
        "status, body, exc",
        [
            (401, {"message": "bad key"}, AuthenticationError),
            (422, {"message": "bad", "field": "queries.department.options"}, ValidationError),
        ],
    )
    async def test_errors_are_not_retried(self, api: MockAPI, make_async_client, sleeps, status, body, exc):
        client = make_async_client()
        api.reply(status, body)
        with pytest.raises(exc):
            await client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
        assert len(api.requests) == 1 and sleeps == []

    async def test_424_heuristic_then_strict(self, api: MockAPI, make_async_client):
        api.reply(424, evaluate_body({"department": heuristic_set_answer()}))
        result = await make_async_client().evaluate(
            context=CONTEXT, backend=OPENAI, queries={"department": department_query()}
        )
        assert result.heuristic_answers == ["department"]

        api.reply(424, {"message": "profile too small"})
        with pytest.raises(InsufficientCalibrationError, match="too small"):
            await make_async_client(strict_guarantees=True).evaluate(
                context=CONTEXT, backend=OPENAI, queries={"department": department_query()}
            )

    async def test_custom_backend_evidence(self, api: MockAPI, make_async_client):
        backend = HiddenStateBackend("L4")
        api.reply(200, evaluate_body({}))
        await make_async_client().evaluate(
            context=CONTEXT, backend=backend, queries={"department": department_query()}
        )
        body = api.body()
        assert body["backend"]["provider"] == "client_evidence"
        assert set(body["queries"]["department"]["evidence"]) == {"option_probabilities", "hidden_states"}

    async def test_list_backends(self, api: MockAPI, make_async_client):
        api.reply(200, {"backends": [{"provider": "vllm", "access_level": "L4"}]})
        assert await make_async_client().list_backends() == [{"provider": "vllm", "access_level": "L4"}]


def test_request_bodies_are_valid_json_with_no_nan(api: MockAPI, client: CLIClient):
    api.reply(200, evaluate_body({}))
    client.evaluate(context=CONTEXT, backend=OPENAI, queries={"department": department_query()})
    json.loads(api.last.content, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
