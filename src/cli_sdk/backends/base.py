"""Backend adapters: which model produces the raw evidence CLI calibrates.

Two kinds of backend exist:

``RemoteBackend`` subclasses (OpenAI, Azure OpenAI, Anthropic, Gemini,
Bedrock, OpenRouter, vLLM, SGLang)
    Configuration only. They serialize to the ``backend`` object in the
    request, and the CLI service calls the provider on your behalf using a
    stored provider connection in your workspace.

``CustomBackend``
    Bring your own model. You implement the evidence methods your access
    level supports (``score_options``, ``sample``, ``score_text``,
    ``hidden_states``); the SDK calls them locally and sends the resulting
    *evidence* (probabilities, samples, scores, hidden-state vectors at L4)
    to the service, which calibrates it. Your model, its weights and its
    endpoint never leave your infrastructure. The request still carries the
    ``context`` and each query's definition (``instructions``, ``options``,
    ``levels``, ...), which the service needs to calibrate and group answers.

See apps/docs/pages/backends.mdx and apps/docs/pages/concepts/access-ladder.mdx.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional, Union

from cli_sdk.constants import (
    ACCESS_LEVEL_L0,
    ACCESS_LEVEL_L1,
    ACCESS_LEVEL_L2,
    ACCESS_LEVEL_L3,
    ACCESS_LEVEL_L4,
)
from cli_sdk.exceptions import ConfigurationError

ACCESS_LEVELS = (ACCESS_LEVEL_L0, ACCESS_LEVEL_L1, ACCESS_LEVEL_L2, ACCESS_LEVEL_L3, ACCESS_LEVEL_L4)
ACCESS_HINTS = ("auto", "sampling", "logprobs", "prompt-scoring", "exact", "hidden-state")


def access_rank(level: str) -> int:
    """Order access levels so they can be compared (L0 < L1 < ... < L4)."""
    try:
        return ACCESS_LEVELS.index(level)
    except ValueError as exc:
        raise ConfigurationError(f"unknown access level {level!r}; expected one of {ACCESS_LEVELS}") from exc


@dataclass
class RemoteBackend:
    """Configuration for a provider the CLI service calls on your behalf."""

    provider: str = field(default="", init=False)
    model: str = ""
    access_hint: str = "auto"
    connection: Optional[str] = None
    """Name of a stored provider credential in your workspace. Defaults to
    the workspace's default connection for this provider."""

    def __post_init__(self) -> None:
        if not self.model:
            raise ConfigurationError(f"{type(self).__name__} requires `model`")
        if self.access_hint not in ACCESS_HINTS:
            raise ConfigurationError(f"access_hint must be one of {ACCESS_HINTS}")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            payload[f.name] = value
        return payload


class CustomBackend:
    """Bring-your-own-model backend that computes evidence client-side.

    Set ``access_level`` to the highest level your model supports and
    implement the matching methods. Methods you do not implement raise
    ``NotImplementedError``, and a query that needs them fails with a clear
    error instead of silently downgrading its guarantee.

    ============  =========================================================
    Level         Methods used
    ============  =========================================================
    L0            ``sample``
    L1 and above  ``score_options`` (plus ``sample`` for open-ended queries)
    L2 and above  ``score_text`` (reserved: SDK 0.1.0 does not call it yet)
    L4            ``hidden_states`` (for probe-based scores)
    ============  =========================================================
    """

    access_level: str = ACCESS_LEVEL_L0
    name: str = "custom"

    def __init__(self, base_url: Optional[str] = None, **options: Any) -> None:
        self.base_url = base_url
        self.options = options
        access_rank(self.access_level)  # validate at construction time

    # -- evidence methods (implement the ones your access level supports) --

    def score_options(
        self, context: Any, instructions: Any, options: dict[str, Optional[str]]
    ) -> dict[str, float]:
        """Return a probability for each option key (need not sum to 1)."""
        raise NotImplementedError

    def sample(self, context: Any, instructions: Any, n: int) -> list[str]:
        """Return ``n`` independently sampled answers to ``instructions``."""
        raise NotImplementedError

    def score_text(self, context: Any, instructions: Any, candidate: str) -> float:
        """Return a support score (higher is better) for ``candidate``."""
        raise NotImplementedError

    def hidden_states(self, context: Any, instructions: Any) -> list[float]:
        """Return a feature vector for probe-based nonconformity scores."""
        raise NotImplementedError

    # -- serialization ---------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        return {"provider": "client_evidence", "name": self.name, "access_level": self.access_level}

    def _call(self, method: str, qtype: Any, *args: Any) -> Any:
        """Call one evidence method, turning a missing implementation into a clear error."""
        try:
            return getattr(self, method)(*args)
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"{type(self).__name__}.{method}() is not implemented, but access_level "
                f"{self.access_level!r} uses it to collect evidence for {qtype!r} queries. "
                f"Implement {method}() (or declare the access_level your model really supports); "
                "the query was not sent, so its guarantee is never silently downgraded."
            ) from exc

    def collect_evidence(self, query_payload: dict[str, Any], context: Any, sample_count: int = 20) -> dict[str, Any]:
        """Compute the evidence a query needs, at this backend's access level."""
        qtype = query_payload.get("type")
        instructions = query_payload.get("instructions")
        rank = access_rank(self.access_level)

        if qtype in ("set", "belief", "gate", "judge"):
            options = query_payload.get("options")
            if qtype == "belief" or qtype == "gate":
                options = {"true": None, "false": None}
            if qtype == "judge":
                options = {"response_a": None, "response_b": None}
            if rank >= access_rank(ACCESS_LEVEL_L1):
                probs = self._call("score_options", qtype, context, instructions, options or {})
                evidence: dict[str, Any] = {"option_probabilities": probs}
            else:
                evidence = {"samples": self._call("sample", qtype, context, instructions, sample_count)}
            if rank >= access_rank(ACCESS_LEVEL_L4):
                try:
                    evidence["hidden_states"] = self.hidden_states(context, instructions)
                except NotImplementedError:
                    pass
            return evidence

        if qtype == "interval":
            levels = query_payload.get("levels") or []
            level_options = {str(i): name for i, name in enumerate(levels)}
            if rank >= access_rank(ACCESS_LEVEL_L1):
                probs = self._call("score_options", qtype, context, instructions, level_options)
                return {"level_probabilities": probs}
            return {"samples": self._call("sample", qtype, context, instructions, sample_count)}

        if qtype == "claim":
            evidence = {"samples": self._call("sample", qtype, context, instructions, sample_count)}
            if rank >= access_rank(ACCESS_LEVEL_L2):
                evidence["claim_scoring"] = "client"  # server returns claims to score via score_text
            return evidence

        raise ConfigurationError(f"CustomBackend cannot collect evidence for query type {qtype!r}")


BackendLike = Union[RemoteBackend, CustomBackend, dict]


def backend_payload(backend: BackendLike) -> dict[str, Any]:
    """Normalize a backend argument (config object, custom backend, or dict)."""
    if isinstance(backend, (RemoteBackend, CustomBackend)):
        return backend.to_payload()
    if isinstance(backend, dict):
        if "provider" not in backend:
            raise ConfigurationError("backend dict requires a 'provider' key")
        return dict(backend)
    raise ConfigurationError(f"unsupported backend type: {type(backend).__name__}")
