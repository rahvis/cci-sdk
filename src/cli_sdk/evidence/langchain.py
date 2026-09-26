"""Evidence from any LangChain chat model.

Wrap the same chat model your LangChain or LangGraph agent already uses, so
the model that acts is the model whose decisions are calibrated::

    from langchain_openai import ChatOpenAI
    from cli_sdk.evidence import LangChainEvidenceBackend

    llm = ChatOpenAI(model="gpt-4.1-2025-04-14")
    evidence = LangChainEvidenceBackend(llm)                      # L0: sampling
    evidence = LangChainEvidenceBackend(llm, use_logprobs=True)   # L1 when the model returns logprobs

With ``use_logprobs=True`` the backend binds ``logprobs=True, top_logprobs=20``
and reads ``response_metadata["logprobs"]["content"][0]["top_logprobs"]``.
Only ``ChatOpenAI`` / ``AzureChatOpenAI`` on the Chat Completions API populate
it (construct them with ``use_responses_api=False``; ``ChatOpenAI(base_url=...)``
pointed at vLLM or SGLang works the same way). ``ChatAnthropic`` and
``ChatGoogleGenerativeAI`` return no log-probabilities: use the default
sampling mode with them.

Requires ``langchain-core``.
"""

from __future__ import annotations

from typing import Any, Optional

from cli_sdk.evidence.base import EvidenceBackend, EvidenceError


class LangChainEvidenceBackend(EvidenceBackend):
    provider = "langchain"

    def __init__(
        self,
        chat_model: Any,
        *,
        use_logprobs: bool = False,
        top_logprobs: int = 20,
        name: Optional[str] = None,
        max_tokens: int = 256,
    ) -> None:
        self.chat_model = chat_model
        self.supports_logprobs = use_logprobs
        self.access_level = "L1" if use_logprobs else "L0"
        self.top_logprobs = max(1, min(20, top_logprobs))
        model_name = (
            getattr(chat_model, "model_name", None)
            or getattr(chat_model, "model", None)
            or type(chat_model).__name__
        )
        super().__init__(str(model_name), max_tokens=max_tokens, name=name or f"langchain:{model_name}")

    @staticmethod
    def _messages(system: str, user: str) -> list[Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        return [SystemMessage(content=system), HumanMessage(content=user)]

    @staticmethod
    def _text(message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        # content blocks: keep text parts only
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))

    def _complete(self, system: str, user: str, *, n: int, temperature: float, max_tokens: int) -> list[str]:
        messages = self._messages(system, user)
        try:
            replies = self.chat_model.batch([messages] * n) if n > 1 else [self.chat_model.invoke(messages)]
        except Exception as exc:
            raise EvidenceError(f"{self.name}: invocation failed: {exc}") from exc
        return [self._text(reply) for reply in replies]

    def _first_token_logprobs(self, system: str, user: str) -> list[tuple[str, float]]:
        try:
            bound = self.chat_model.bind(logprobs=True, top_logprobs=self.top_logprobs)
            reply = bound.invoke(self._messages(system, user))
        except Exception as exc:
            raise EvidenceError(
                f"{self.name}: log-probability request failed ({exc}); use use_logprobs=False"
            ) from exc
        logprobs = (getattr(reply, "response_metadata", {}) or {}).get("logprobs") or {}
        content = logprobs.get("content") or []
        if not content:
            raise EvidenceError(f"{self.name}: the model returned no logprobs; use use_logprobs=False")
        first = content[0]
        pairs = [(entry["token"], entry["logprob"]) for entry in first.get("top_logprobs", [])]
        return pairs or [(first["token"], first["logprob"])]

    # The wrapped chat model samples at its own configuration, so these settings
    # (not ``EvidenceBackend.temperature``) define the scoring function.
    _SETTINGS = ("temperature", "top_p", "top_k", "reasoning_effort", "openai_api_base", "azure_endpoint",
                 "deployment_name", "anthropic_api_url", "base_url")

    def fingerprint(self) -> dict[str, Any]:
        """Model name plus the chat model's class and sampling settings.

        A profile calibrated with ``ChatOpenAI(temperature=1.0)`` must not be
        served by the same model at ``temperature=0.7``: the sampled scores
        come from a different distribution.
        """
        fp = super().fingerprint()
        cls = type(self.chat_model)
        settings: dict[str, Any] = {"class": f"{cls.__module__}.{cls.__qualname__}"}
        for attr in self._SETTINGS:
            value = getattr(self.chat_model, attr, None)
            if value is not None:
                settings[attr] = value if isinstance(value, (str, int, float, bool)) else str(value)
        fp["temperature"] = settings.get("temperature")
        fp["chat_model"] = settings
        fp["top_logprobs"] = self.top_logprobs if self.supports_logprobs else None
        return fp
