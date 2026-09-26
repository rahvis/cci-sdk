"""Model calls that turn a (context, query) pair into calibratable evidence.

Every function here runs identically at calibration time and at inference
time. That symmetry is the whole point: a conformal guarantee holds for the
*scoring function* the calibration set was scored with, so the same prompt,
the same backend settings, and the same parsing rule must produce both.

Access levels
-------------
L1 and above  one request, first-token log-probabilities over option letters
L0            ``sample_count`` sampled replies, parsed to option keys and
              turned into smoothed frequencies
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from cli_sdk.backends.base import ACCESS_LEVEL_L1, CustomBackend, access_rank
from cli_sdk.evidence import _prompts
from cli_sdk.exceptions import ConfigurationError

BINARY_OPTIONS: dict[str, Optional[str]] = {
    "true": "yes, the statement holds",
    "false": "no, the statement does not hold",
}
JUDGE_OPTIONS: dict[str, Optional[str]] = {
    "response_a": "response A is better",
    "response_b": "response B is better",
}

# Jeffreys-style additive smoothing for sampled frequencies. Any fixed rule is
# valid for conformal calibration as long as it is applied identically at
# calibration and inference; this one keeps every option's score positive.
SMOOTHING = 0.5

CLAIM_SUPPORT_INSTRUCTIONS = (
    "Decide whether the claim below is fully supported by the context. "
    "Answer 'true' only if the context states or directly implies it.\n\nClaim: {claim}"
)
CLAIM_DECOMPOSE_INSTRUCTIONS = (
    "Split the text in the 'answer' field of the context into short, atomic, self-contained "
    "factual claims. Return only a JSON array of strings."
)


@dataclass
class UsageCounter:
    backend_calls: int = 0

    def add(self, n: int = 1) -> None:
        self.backend_calls += n


def backend_fingerprint(backend: CustomBackend) -> dict[str, Any]:
    fingerprint = getattr(backend, "fingerprint", None)
    if callable(fingerprint):
        return dict(fingerprint())
    return {
        "provider": "custom",
        "class": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "name": getattr(backend, "name", "custom"),
        "access_level": backend.access_level,
    }


def _normalize(scores: Mapping[str, float], options: Mapping[str, Any]) -> dict[str, float]:
    clipped = {key: max(float(scores.get(key, 0.0)), 0.0) for key in options}
    total = sum(clipped.values())
    if total <= 0:
        return {key: 1.0 / len(options) for key in options}
    return {key: value / total for key, value in clipped.items()}


def _frequencies(keys: list[str], options: Mapping[str, Any]) -> dict[str, float]:
    counts = {key: 0 for key in options}
    for key in keys:
        if key in counts:
            counts[key] += 1
    valid = sum(counts.values())
    denom = valid + SMOOTHING * len(options)
    return {key: (count + SMOOTHING) / denom for key, count in counts.items()}


def option_probabilities(
    backend: CustomBackend,
    context: Any,
    instructions: Any,
    options: Mapping[str, Optional[str]],
    sample_count: int,
    usage: Optional[UsageCounter] = None,
) -> dict[str, float]:
    """Probability of each option key, at the backend's access level."""
    if access_rank(backend.access_level) >= access_rank(ACCESS_LEVEL_L1):
        try:
            scores = backend.score_options(context, instructions, dict(options))
        except NotImplementedError:
            scores = None
        if scores is not None:
            if usage:
                usage.add(1)
            return _normalize(scores, options)
    sample_options = getattr(backend, "sample_options", None)
    if callable(sample_options):
        keys = sample_options(context, instructions, dict(options), sample_count)
    else:
        # A plain CustomBackend only implements free-text ``sample``: ask it to
        # name an option, then parse replies with the shared, conservative rule.
        listed = "\n".join(f"- {key}" + (f": {desc}" if desc else "") for key, desc in options.items())
        prompt = (
            f"{_prompts.render_instructions(instructions)}\n\nChoose exactly one of these options:\n"
            f"{listed}\n\nReply with the option name exactly as written above and nothing else."
        )
        try:
            replies = backend.sample(context, prompt, sample_count)
        except NotImplementedError as exc:
            raise ConfigurationError(
                f"backend {getattr(backend, 'name', type(backend).__name__)!r} implements neither "
                "score_options (L1) nor sample (L0)"
            ) from exc
        keys = [_prompts.parse_option(reply, options) for reply in replies]
    if usage:
        usage.add(sample_count)
    return _frequencies(list(keys), options)


def probability_true(
    backend: CustomBackend, context: Any, instructions: Any, sample_count: int,
    usage: Optional[UsageCounter] = None,
) -> float:
    return option_probabilities(backend, context, instructions, BINARY_OPTIONS, sample_count, usage)["true"]


def decompose_claims(backend: CustomBackend, context: Any, usage: Optional[UsageCounter] = None) -> list[str]:
    """Split the answer in ``context`` into atomic claims using the backend."""
    generate = getattr(backend, "generate", None)
    text: str
    if callable(generate):
        text = generate(context, CLAIM_DECOMPOSE_INSTRUCTIONS)
    else:
        replies = backend.sample(context, CLAIM_DECOMPOSE_INSTRUCTIONS, 1)
        text = replies[0] if replies else ""
    if usage:
        usage.add(1)
    claims = _prompts.extract_json_list(text)
    if claims is None:
        answer = context.get("answer") if isinstance(context, Mapping) else None
        claims = _prompts.split_sentences(answer if isinstance(answer, str) else text)
    return claims


def claim_support(
    backend: CustomBackend, context: Any, claim: str, sample_count: int,
    usage: Optional[UsageCounter] = None,
) -> float:
    instructions = CLAIM_SUPPORT_INSTRUCTIONS.format(claim=claim)
    support_context = context
    if isinstance(context, Mapping) and "answer" in context:
        # Judge support against the sources, not against the answer being checked.
        support_context = {k: v for k, v in context.items() if k != "answer"}
    return probability_true(backend, support_context, instructions, sample_count, usage)


def canonical(value: Any) -> str:
    """Stable string form used to compare labels and group keys."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)
