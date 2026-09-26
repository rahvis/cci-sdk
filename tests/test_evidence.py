"""Evidence backends: prompt rendering, parsing, and the requests sent to each provider.

Provider SDK clients are replaced by recording fakes, so these tests need
neither keys nor the provider packages.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cli_sdk.evidence import (
    AnthropicEvidenceBackend,
    AzureOpenAIEvidenceBackend,
    MockEvidenceBackend,
    OpenAICompatibleEvidenceBackend,
    OpenAIEvidenceBackend,
    sglang_backend,
    vllm_backend,
)
from cli_sdk.evidence import _prompts
from cli_sdk.evidence.base import EvidenceError

BINARY = {"true": None, "false": None}


class FakeCompletions:
    """Records chat.completions.create kwargs and replies like the OpenAI SDK."""

    def __init__(self, top=None, texts=None):
        self.calls = []
        self.top = top or [("A", -0.1), ("B", -2.5), (" A", -4.0)]
        self.texts = texts or ["A"]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("logprobs"):
            top = [SimpleNamespace(token=t, logprob=lp) for t, lp in self.top]
            content = [SimpleNamespace(token=self.top[0][0], logprob=self.top[0][1], top_logprobs=top)]
            choice = SimpleNamespace(message=SimpleNamespace(content=self.top[0][0]),
                                     logprobs=SimpleNamespace(content=content))
            return SimpleNamespace(choices=[choice])
        n = kwargs.get("n", 1)
        choices = [SimpleNamespace(message=SimpleNamespace(content=self.texts[i % len(self.texts)]), logprobs=None)
                   for i in range(n)]
        return SimpleNamespace(choices=choices)


def fake_openai(**kwargs):
    completions = FakeCompletions(**kwargs)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


# ---------------------------------------------------------------------------
# prompts and parsing
# ---------------------------------------------------------------------------


def test_letter_probabilities_merge_variants_and_share_residual():
    letters = {"A": "yes", "B": "no", "C": "unsure"}
    probs = _prompts.letter_probabilities([("A", -0.2), (" A", -3.0), ("B", -2.0)], letters)
    assert set(probs) == {"yes", "no", "unsure"}
    assert probs["yes"] > probs["no"] > probs["unsure"] > 0
    assert sum(probs.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("reply,expected", [
    ("A", "approve"), ("a.", "approve"), ("(B)", "deny"), ("B. deny", "deny"),
    ("deny", "deny"), ("Approve it", "approve"), ("The answer is A", _prompts.INVALID), ("", _prompts.INVALID),
])
def test_parse_letter(reply, expected):
    letters = {"A": "approve", "B": "deny"}
    assert _prompts.parse_letter(reply, letters, {"approve": None, "deny": None}) == expected


def test_parse_option_refuses_ambiguous_replies():
    options = {"billing": None, "technical": None}
    assert _prompts.parse_option("billing and technical", options) == _prompts.INVALID
    assert _prompts.parse_option("Billing.", options) == "billing"


def test_context_rendering_is_order_independent():
    assert _prompts.render_context({"b": 1, "a": 2}) == _prompts.render_context({"a": 2, "b": 1})


# ---------------------------------------------------------------------------
# OpenAI-compatible backends
# ---------------------------------------------------------------------------


def test_openai_scoring_request_shape():
    client, completions = fake_openai()
    backend = OpenAIEvidenceBackend("gpt-4.1-mini-2025-04-14", client=client)
    probs = backend.score_options({"ticket": "refund"}, "Approve?", BINARY)
    assert probs["true"] > 0.9
    call = completions.calls[-1]
    assert call["logprobs"] is True and call["top_logprobs"] == 20
    assert call["temperature"] == 0.0 and call["max_completion_tokens"] == 1
    assert "max_tokens" not in call
    assert call["messages"][0]["role"] == "system"


def test_openai_reasoning_effort_drops_temperature_when_sampling():
    client, completions = fake_openai(texts=["A", "B"])
    backend = OpenAIEvidenceBackend("gpt-6-astra", client=client, use_logprobs=False, reasoning_effort="low")
    assert backend.access_level == "L0"
    keys = backend.sample_options({}, "Approve?", BINARY, 4)
    assert keys == ["true", "false", "true", "false"]
    call = completions.calls[-1]
    assert "temperature" not in call and call["reasoning_effort"] == "low" and call["n"] == 4


def test_self_hosted_factories():
    client, completions = fake_openai()
    v = vllm_backend("google/gemma-4-12B-it", client=client)
    assert v.provider == "vllm" and v.name == "vllm:google/gemma-4-12B-it"
    v.score_options({}, "q", BINARY)
    call = completions.calls[-1]
    assert call["max_tokens"] == 1
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    s = sglang_backend("google/gemma-4-12B-it", client=client, disable_thinking=False)
    assert s.provider == "sglang" and "extra_body" not in (s._common_kwargs())
    assert v.fingerprint() != s.fingerprint()


def test_too_many_options_for_logprobs():
    client, _ = fake_openai()
    backend = OpenAICompatibleEvidenceBackend("m", client=client)
    with pytest.raises(EvidenceError):
        backend.score_options({}, "q", {f"o{i}": None for i in range(21)})


def test_missing_logprobs_raise_evidence_error():
    class NoLogprobs(FakeCompletions):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A"), logprobs=None)])

    completions = NoLogprobs()
    backend = OpenAICompatibleEvidenceBackend("m", client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    with pytest.raises(EvidenceError, match="use_logprobs=False"):
        backend.score_options({}, "q", BINARY)


def test_azure_uses_deployment_and_completion_tokens():
    client, completions = fake_openai()
    backend = AzureOpenAIEvidenceBackend("my-deployment", azure_endpoint="https://example.openai.azure.com",
                                         client=client)
    backend.score_options({}, "q", BINARY)
    assert completions.calls[-1]["model"] == "my-deployment"
    assert "max_completion_tokens" in completions.calls[-1]
    assert backend.fingerprint()["azure_endpoint"] == "https://example.openai.azure.com"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class FakeMessages:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers[(len(self.calls) - 1) % len(self.answers)]
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=answer)])


def test_anthropic_samples_with_structured_enum_and_no_temperature():
    messages = FakeMessages([json.dumps({"answer": "true"}), json.dumps({"answer": "false"})])
    backend = AnthropicEvidenceBackend("claude-sonnet-5", client=SimpleNamespace(messages=messages),
                                       max_concurrency=1)
    assert backend.access_level == "L0"
    keys = backend.sample_options({"case": 1}, "Approve?", BINARY, 4)
    assert sorted(keys) == ["false", "false", "true", "true"]
    call = messages.calls[-1]
    assert "temperature" not in call and "top_p" not in call
    schema = call["output_config"]["format"]["schema"]
    assert schema["properties"]["answer"]["enum"] == ["true", "false"]


def test_anthropic_effort_is_part_of_output_config_and_fingerprint():
    messages = FakeMessages([json.dumps({"answer": "true"})])
    backend = AnthropicEvidenceBackend("claude-sonnet-5", effort="low", client=SimpleNamespace(messages=messages))
    backend.sample_options({}, "q", BINARY, 1)
    assert messages.calls[-1]["output_config"]["effort"] == "low"
    assert backend.fingerprint()["effort"] == "low"
    with pytest.raises(NotImplementedError):
        backend.score_options({}, "q", BINARY)


def test_anthropic_invalid_structured_reply_is_invalid():
    messages = FakeMessages([json.dumps({"answer": "maybe"})])
    backend = AnthropicEvidenceBackend(client=SimpleNamespace(messages=messages))
    assert backend.sample_options({}, "q", BINARY, 1) == [_prompts.INVALID]


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------


def test_mock_is_deterministic_and_keyless():
    a = MockEvidenceBackend(keywords={"billing": ["invoice"]}, seed=3)
    b = MockEvidenceBackend(keywords={"billing": ["invoice"]}, seed=3)
    options = {"billing": None, "technical": None}
    ctx = {"text": "my invoice is wrong"}
    assert a.score_options(ctx, "route", options) == b.score_options(ctx, "route", options)
    assert a.score_options(ctx, "route", options)["billing"] > 0.5
    l0 = MockEvidenceBackend(access_level="L0")
    with pytest.raises(NotImplementedError):
        l0.score_options(ctx, "route", options)
    assert len(l0.sample_options(ctx, "route", options, 7)) == 7


def test_anthropic_thinking_is_disabled_where_allowed_and_budgeted_where_not():
    messages = FakeMessages([json.dumps({"answer": "true"})])
    sonnet = AnthropicEvidenceBackend("claude-sonnet-5", client=SimpleNamespace(messages=messages))
    sonnet.sample_options({}, "q", BINARY, 1)
    assert messages.calls[-1]["thinking"] == {"type": "disabled"}
    assert messages.calls[-1]["max_tokens"] == 256
    always = AnthropicEvidenceBackend("claude-opus-5-5", client=SimpleNamespace(messages=messages))
    always.sample_options({}, "q", BINARY, 1)
    assert "thinking" not in messages.calls[-1] and messages.calls[-1]["max_tokens"] >= 2048
    assert sonnet.fingerprint()["thinking"] == "disabled" and always.fingerprint()["thinking"] == "model-default"


def test_anthropic_truncated_or_refused_replies_are_invalid():
    class Truncated(FakeMessages):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(stop_reason="max_tokens",
                                   content=[SimpleNamespace(type="thinking", thinking="...", signature="x")])

    backend = AnthropicEvidenceBackend("claude-opus-5-5", client=SimpleNamespace(messages=Truncated([])))
    assert backend.sample_options({}, "q", BINARY, 2) == [_prompts.INVALID, _prompts.INVALID]


def test_gemini_backend_is_sampling_only_with_room_to_think():
    from cli_sdk.evidence import gemini_backend

    client, completions = fake_openai(texts=["A"])
    backend = gemini_backend("gemini-3.8-flash", client=client)
    assert backend.access_level == "L0" and backend.provider == "gemini"
    backend.sample_options({}, "q", BINARY, 3)
    assert len(completions.calls) == 3 and all("n" not in c for c in completions.calls)
    assert completions.calls[-1]["max_tokens"] == 1024
