"""Evidence from OpenAI, Azure OpenAI, and any OpenAI-compatible server.

One implementation covers:

- OpenAI (``OpenAIEvidenceBackend``)
- Azure OpenAI (``AzureOpenAIEvidenceBackend``)
- open-weight models on vLLM or SGLang, e.g. Gemma (``vllm_backend`` /
  ``sglang_backend``, or ``OpenAICompatibleEvidenceBackend`` directly)
- Google Gemini through its OpenAI-compatible endpoint (``gemini_backend``;
  sampling only)

Access level: L1 when ``use_logprobs=True`` (first-token ``top_logprobs``,
at most 20 per position), otherwise L0 (sampling).

OpenAI returns log-probabilities only from non-reasoning configurations:
the gpt-4.1 family, or models that accept ``reasoning_effort="none"``
(for example gpt-6-sol, gpt-6-luna, gpt-5.1 and later). For any other
reasoning model, construct the backend with ``use_logprobs=False``.

Requires ``pip install "cli-sdk[openai]"``.
"""

from __future__ import annotations

from typing import Any, Optional

from cli_sdk.evidence.base import EvidenceBackend, EvidenceError



def _require_openai():
    try:
        import openai  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError('this backend needs the OpenAI SDK: pip install "cli-sdk[openai]"') from exc
    return openai


class OpenAICompatibleEvidenceBackend(EvidenceBackend):
    """Evidence from any server that speaks the OpenAI Chat Completions API."""

    provider = "openai-compatible"

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_logprobs: bool = True,
        top_logprobs: int = 20,
        supports_n: bool = True,
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,
        extra_body: Optional[dict[str, Any]] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
        timeout: float = 60.0,
        token_limit_param: str = "max_tokens",
        client: Any = None,
        name: Optional[str] = None,
    ) -> None:
        if token_limit_param not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("token_limit_param must be 'max_tokens' or 'max_completion_tokens'")
        self.token_limit_param = token_limit_param
        self.supports_logprobs = use_logprobs
        self.access_level = "L1" if use_logprobs else "L0"
        self.top_logprobs = max(1, min(20, top_logprobs))
        self.supports_n = supports_n
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.extra_body = extra_body or {}
        self._base_url = base_url
        if client is None:
            openai = _require_openai()
            client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self._client = client
        super().__init__(model, temperature=temperature, max_tokens=max_tokens, name=name, base_url=base_url)

    # -- helpers -----------------------------------------------------------

    def _token_limit_kwargs(self, max_tokens: int) -> dict[str, Any]:
        # OpenAI deprecated max_tokens for max_completion_tokens; self-hosted
        # OpenAI-compatible servers universally accept max_tokens.
        return {self.token_limit_param: max_tokens}

    def _common_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return kwargs

    @staticmethod
    def _messages(system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # -- provider primitives -------------------------------------------------

    def _complete(self, system: str, user: str, *, n: int, temperature: float, max_tokens: int) -> list[str]:
        messages = self._messages(system, user)
        kwargs = {**self._common_kwargs(), **self._token_limit_kwargs(max_tokens)}
        sampling = {} if self.reasoning_effort not in (None, "none") else {"temperature": temperature}
        try:
            if self.supports_n and n > 1:
                replies: list[str] = []
                remaining = n
                while remaining > 0:
                    batch = min(remaining, 128)
                    response = self._client.chat.completions.create(
                        model=self.model, messages=messages, n=batch, **sampling, **kwargs
                    )
                    replies.extend((choice.message.content or "") for choice in response.choices)
                    remaining -= batch
                return replies
            replies = []
            for _ in range(n):
                response = self._client.chat.completions.create(
                    model=self.model, messages=messages, **sampling, **kwargs
                )
                replies.append(response.choices[0].message.content or "")
            return replies
        except Exception as exc:  # surface provider errors with context
            raise EvidenceError(f"{self.name}: completion failed: {exc}") from exc

    def _first_token_logprobs(self, system: str, user: str) -> list[tuple[str, float]]:
        kwargs = {**self._common_kwargs(), **self._token_limit_kwargs(1)}
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, user),
                temperature=0.0,
                logprobs=True,
                top_logprobs=self.top_logprobs,
                **kwargs,
            )
        except Exception as exc:
            raise EvidenceError(
                f"{self.name}: log-probability request failed ({exc}). If this model does not "
                "return logprobs (e.g. a reasoning configuration), construct the backend with "
                "use_logprobs=False to fall back to sampling."
            ) from exc
        choice = response.choices[0]
        content = getattr(getattr(choice, "logprobs", None), "content", None)
        if not content:
            raise EvidenceError(
                f"{self.name}: the response carried no log-probabilities, so this model or server "
                "does not return them; construct the backend with use_logprobs=False to sample instead"
            )
        first = content[0]
        pairs = [(entry.token, entry.logprob) for entry in (first.top_logprobs or [])]
        if not pairs:
            pairs = [(first.token, first.logprob)]
        return pairs

    def fingerprint(self) -> dict[str, Any]:
        fp = super().fingerprint()
        fp.update({
            "base_url": self._base_url,
            "top_logprobs": self.top_logprobs if self.supports_logprobs else None,
            "reasoning_effort": self.reasoning_effort,
            "extra_body": self.extra_body or None,
        })
        return fp


class OpenAIEvidenceBackend(OpenAICompatibleEvidenceBackend):
    """OpenAI. Prefer a dated snapshot, e.g. ``gpt-4.1-mini-2025-04-14``, so a model update never
    silently changes the scoring function behind a calibration profile."""

    provider = "openai"

    def __init__(self, model: str, **kwargs: Any) -> None:
        kwargs.setdefault("token_limit_param", "max_completion_tokens")
        super().__init__(model, **kwargs)


class AzureOpenAIEvidenceBackend(OpenAICompatibleEvidenceBackend):
    """Azure OpenAI. ``model`` is your deployment name.

    Set the deployment's version-upgrade policy to NoAutoUpgrade so a
    calibration profile is never silently scored by a different model.
    """

    provider = "azure-openai"

    def __init__(
        self,
        deployment: str,
        *,
        azure_endpoint: str,
        api_key: Optional[str] = None,
        api_version: str = "2024-10-21",
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        if client is None:
            openai = _require_openai()
            client = openai.AzureOpenAI(azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version)
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        kwargs.setdefault("token_limit_param", "max_completion_tokens")
        super().__init__(deployment, client=client, **kwargs)

    def fingerprint(self) -> dict[str, Any]:
        fp = super().fingerprint()
        fp.update({"azure_endpoint": self.azure_endpoint, "api_version": self.api_version})
        return fp


def _self_hosted_extra_body(kwargs: dict[str, Any], disable_thinking: bool) -> None:
    if disable_thinking:
        extra = dict(kwargs.pop("extra_body", None) or {})
        template = dict(extra.get("chat_template_kwargs") or {})
        template.setdefault("enable_thinking", False)
        extra["chat_template_kwargs"] = template
        kwargs["extra_body"] = extra


def vllm_backend(model: str, base_url: str = "http://localhost:8000/v1", *, api_key: str = "EMPTY",
                 disable_thinking: bool = True, **kwargs: Any) -> OpenAICompatibleEvidenceBackend:
    """An open-weight model (e.g. ``google/gemma-4-12B-it``) served by vLLM.

    ``model`` must equal the served model name. Launch the server with the
    model's own sampling defaults switched off, so sampled evidence is not
    truncated by ``top_k`` / ``top_p`` from ``generation_config.json``::

        vllm serve google/gemma-4-12B-it --max-logprobs 20 --generation-config vllm

    ``disable_thinking`` sends ``chat_template_kwargs={"enable_thinking": False}``
    so the first generated token is the answer, not reasoning (Gemma 4 and
    other hybrid-thinking templates honour it; others ignore it).
    """
    _self_hosted_extra_body(kwargs, disable_thinking)
    backend = OpenAICompatibleEvidenceBackend(model, api_key=api_key, base_url=base_url, name=f"vllm:{model}", **kwargs)
    backend.provider = "vllm"
    return backend


def sglang_backend(model: str, base_url: str = "http://localhost:30000/v1", *, api_key: str = "EMPTY",
                   disable_thinking: bool = True, **kwargs: Any) -> OpenAICompatibleEvidenceBackend:
    """An open-weight model served by SGLang's OpenAI-compatible server::

        python -m sglang.launch_server --model-path google/gemma-4-12B-it --port 30000

    Scoring runs at temperature 0, where SGLang returns unscaled
    log-probabilities (at temperature > 0 it returns temperature-scaled
    ones unless the server sets ``SGLANG_RETURN_ORIGINAL_LOGPROB=1``).
    Per-request seeds are honoured only with ``--enable-deterministic-inference``.
    """
    _self_hosted_extra_body(kwargs, disable_thinking)
    backend = OpenAICompatibleEvidenceBackend(model, api_key=api_key, base_url=base_url, name=f"sglang:{model}", **kwargs)
    backend.provider = "sglang"
    return backend


def gemini_backend(model: str, *, api_key: Optional[str] = None, **kwargs: Any) -> OpenAICompatibleEvidenceBackend:
    """Google Gemini through its OpenAI-compatible endpoint (sampling, access level L0).

    Gemini 3.x models return no log-probabilities, so scores come from
    sampling. They also think before answering, so the reply budget is
    generous (``max_tokens=1024``), and each sample is its own request.
    """
    kwargs.setdefault("use_logprobs", False)
    kwargs.setdefault("supports_n", False)
    kwargs.setdefault("max_tokens", 1024)
    backend = OpenAICompatibleEvidenceBackend(
        model,
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        name=f"gemini:{model}",
        **kwargs,
    )
    backend.provider = "gemini"
    return backend
