"""Base class for evidence backends that call a real model.

An evidence backend is a ``CustomBackend`` that talks to *your* model —
OpenAI, Azure OpenAI, Anthropic Claude, Gemini, or an open-weight model on
vLLM / SGLang — and turns its replies into the numbers CLI calibrates:

``score_options``  first-token label probabilities (access level L1)
``sample_options`` repeated sampled choices, parsed to option keys (L0)
``generate``       free text (used by ``Claim`` to decompose an answer)

Provider subclasses implement only two primitives, ``_complete`` and, when
the provider returns token log-probabilities, ``_first_token_logprobs``.
Everything else — prompt rendering, parsing, residual-mass handling — is
shared, so two backends are only ever compared on identical scoring rules.

Evidence backends work with both clients:

- ``LocalCLIClient`` calibrates and scores fully in-process (no CLI service).
- ``CLIClient`` computes evidence locally and sends only the evidence.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional

from cli_sdk.backends.base import CustomBackend
from cli_sdk.evidence import _prompts
from cli_sdk.exceptions import BackendError


class EvidenceError(BackendError):
    """The model backend could not produce the requested evidence."""


class EvidenceBackend(CustomBackend):
    """A ``CustomBackend`` backed by a real model API."""

    provider: str = "custom"
    supports_logprobs: bool = False

    def __init__(self, model: str, *, temperature: float = 1.0, max_tokens: int = 256,
                 name: Optional[str] = None, **options: Any) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = name or f"{self.provider}:{model}"
        super().__init__(**options)

    # -- provider primitives (subclasses implement) ----------------------

    def _complete(self, system: str, user: str, *, n: int, temperature: float, max_tokens: int) -> list[str]:
        raise NotImplementedError

    def _first_token_logprobs(self, system: str, user: str) -> list[tuple[str, float]]:
        raise NotImplementedError(f"{type(self).__name__} does not expose token log-probabilities")

    # -- evidence methods --------------------------------------------------

    def score_options(self, context: Any, instructions: Any, options: Mapping[str, Optional[str]]) -> dict[str, float]:
        if not self.supports_logprobs:
            raise NotImplementedError(
                f"{self.name} is configured at access level {self.access_level} and cannot score "
                "options from log-probabilities; the engine will sample instead"
            )
        if len(options) > _prompts.MAX_LETTER_OPTIONS:
            raise EvidenceError(
                f"{len(options)} options exceed the {_prompts.MAX_LETTER_OPTIONS} top-logprob limit; "
                "use a sampling backend (use_logprobs=False) for large option lists"
            )
        system, user, letters = _prompts.letter_prompt(context, instructions, options)
        top = self._first_token_logprobs(system, user)
        if not top:
            raise EvidenceError(f"{self.name} returned no log-probabilities for the answer token")
        return _prompts.letter_probabilities(top, letters)

    def sample_options(self, context: Any, instructions: Any, options: Mapping[str, Optional[str]], n: int) -> list[str]:
        """Sample ``n`` replies to the letter prompt and parse each to an option key (or ``INVALID``).

        The prompt is the one ``score_options`` reads log-probabilities from,
        so sampled frequencies (L0) estimate the same distribution L1 measures.
        """
        system, user, letters = _prompts.letter_prompt(context, instructions, options)
        replies = self._complete(system, user, n=n, temperature=self.temperature, max_tokens=self.max_tokens)
        return [_prompts.parse_letter(reply, letters, options) for reply in replies]

    def sample(self, context: Any, instructions: Any, n: int) -> list[str]:
        system, user = _prompts.generation_prompt(context, instructions)
        return self._complete(system, user, n=n, temperature=self.temperature, max_tokens=self.max_tokens)

    def generate(self, context: Any, instructions: Any, max_tokens: Optional[int] = None) -> str:
        system, user = _prompts.generation_prompt(context, instructions)
        replies = self._complete(system, user, n=1, temperature=0.0, max_tokens=max_tokens or self.max_tokens)
        return replies[0] if replies else ""

    # -- identity -----------------------------------------------------------

    def fingerprint(self) -> dict[str, Any]:
        """Everything that defines this backend's scoring function.

        Stored with every local calibration profile; a mismatch at inference
        time means the profile no longer describes the model that is scoring.
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "access_level": self.access_level,
            "temperature": self.temperature,
            "prompt_version": PROMPT_VERSION,
        }

    def to_payload(self) -> dict[str, Any]:
        payload = super().to_payload()
        payload["fingerprint"] = self.fingerprint()
        return payload


PROMPT_VERSION = hashlib.sha256(
    json.dumps([_prompts.SYSTEM_PROMPT, "letter-v1", "letter-sampling-v1"]).encode()
).hexdigest()[:12]
