"""Evidence from Anthropic Claude (Messages API).

Claude returns no token log-probabilities and has no ``n`` parameter, so
this backend is access level L0: every score comes from repeated sampling,
one request per sample. Sampled choices use structured output (a JSON
schema whose ``answer`` is an enum of the option keys), so every sample is a
valid option. The Anthropic SDK 1.x has no ``temperature`` parameter and
current models sample at their default, which is what gives samples their
diversity. Keep ``model`` and ``effort`` fixed between calibration and
serving: both are part of the scoring function.

Thinking: Claude Sonnet 5 and Opus 5 think by default, and Opus 5.5 and
Fable always do; thinking tokens count against ``max_tokens``. With the
default ``thinking="auto"`` the backend disables thinking for evidence
sampling where the model allows it (Sonnet 5; Opus 5 at effort up to
"high") and otherwise leaves room for it. Replies that stop on
``max_tokens`` or ``refusal`` are counted as invalid samples, never guessed.

Calibration at L0 costs ``sample_count`` requests per example. Lower
``LocalCLIClient(sample_count=...)`` to trade score resolution for cost, or
score calibration sets with a logprob-capable backend and keep Claude for
the agent itself.

Requires ``pip install "cli-sdk[anthropic]"``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Optional

from cli_sdk.evidence import _prompts
from cli_sdk.evidence.base import EvidenceBackend, EvidenceError

# Models whose thinking cannot be disabled (a request with thinking disabled is a 400).
_ALWAYS_THINKING = ("claude-opus-5-5", "claude-fable", "claude-mythos")


class AnthropicEvidenceBackend(EvidenceBackend):
    provider = "anthropic"
    access_level = "L0"
    supports_logprobs = False

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        *,
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        effort: Optional[str] = None,
        thinking: str = "auto",
        max_concurrency: int = 8,
        client: Any = None,
        name: Optional[str] = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ImportError('this backend needs the Anthropic SDK: pip install "cli-sdk[anthropic]"') from exc
            client = anthropic.Anthropic(api_key=api_key)
        if thinking not in ("auto", "disabled", "default"):
            raise ValueError("thinking must be 'auto', 'disabled' or 'default'")
        self._client = client
        self.effort = effort
        self.max_concurrency = max(1, max_concurrency)
        self.thinking_disabled = thinking == "disabled" or (
            thinking == "auto"
            and not model.startswith(_ALWAYS_THINKING)
            and effort not in ("xhigh", "max")
        )
        # When the model may think, leave it room: thinking and text share max_tokens.
        self.sample_max_tokens = 256 if self.thinking_disabled else max(2048, max_tokens)
        super().__init__(model, temperature=1.0, max_tokens=max_tokens if self.thinking_disabled
                         else max(2048, max_tokens), name=name)

    def _one(self, system: str, user: str, max_tokens: int, output_format: Optional[dict] = None) -> str:
        kwargs: dict[str, Any] = {}
        config: dict[str, Any] = {}
        if self.effort is not None:
            config["effort"] = self.effort
        if output_format is not None:
            config["format"] = output_format
        if config:
            kwargs["output_config"] = config
        if self.thinking_disabled:
            kwargs["thinking"] = {"type": "disabled"}
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        if getattr(response, "stop_reason", None) in ("refusal", "max_tokens"):
            return ""  # parsed as INVALID: a truncated or refused reply is never guessed
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")

    def _many(self, n: int, fn) -> list[str]:
        try:
            if n == 1:
                return [fn()]
            with ThreadPoolExecutor(max_workers=min(n, self.max_concurrency)) as pool:
                return list(pool.map(lambda _: fn(), range(n)))
        except Exception as exc:
            raise EvidenceError(f"{self.name}: request failed: {exc}") from exc

    def sample_options(self, context: Any, instructions: Any, options: Mapping[str, Optional[str]], n: int) -> list[str]:
        system, user = _prompts.naming_prompt(context, instructions, options)
        schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string", "enum": list(options)}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        }

        def one() -> str:
            text = self._one(system, user, self.sample_max_tokens, output_format=schema)
            try:
                value = json.loads(text).get("answer")
            except (ValueError, AttributeError):
                return _prompts.parse_option(text, options)
            return value if value in options else _prompts.INVALID

        return self._many(n, one)

    def _complete(self, system: str, user: str, *, n: int, temperature: float, max_tokens: int) -> list[str]:
        # temperature is intentionally ignored: Claude samples at its default,
        # and sampling diversity comes from that default.
        return self._many(n, lambda: self._one(system, user, max_tokens))

    def fingerprint(self) -> dict[str, Any]:
        fp = super().fingerprint()
        fp.update({"effort": self.effort, "temperature": "provider-default", "sampling": "structured-enum-v1",
                   "thinking": "disabled" if self.thinking_disabled else "model-default"})
        return fp
