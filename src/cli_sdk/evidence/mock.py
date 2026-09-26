"""A deterministic, keyless evidence backend for smoke tests and documentation.

``MockEvidenceBackend`` needs no API key and no network. It scores options by
keyword overlap between the context and each option (plus any keywords you
supply), adds deterministic pseudo-noise so it is *imperfect* — calibration
on it behaves like calibration on a real, fallible model — and samples
reproducibly from those scores. It exists so every example in this
repository runs end to end before you plug in a real model.

Never use it for real decisions: its scores mean nothing outside a demo.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Optional

import numpy as np

from cli_sdk.evidence import _prompts
from cli_sdk.evidence.base import EvidenceBackend

Scorer = Callable[[Any, Any, Mapping[str, Optional[str]]], Mapping[str, float]]

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were with".split()
)


def _words(value: Any) -> set[str]:
    return set(_WORD.findall(_prompts.render_context(value).lower()))


def _stable_uniform(*parts: Any) -> float:
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class MockEvidenceBackend(EvidenceBackend):
    provider = "mock"

    def __init__(
        self,
        scorer: Optional[Scorer] = None,
        *,
        keywords: Optional[Mapping[str, list[str]]] = None,
        access_level: str = "L1",
        sharpness: float = 2.5,
        noise: float = 0.8,
        seed: int = 0,
        name: Optional[str] = None,
    ) -> None:
        self.access_level = access_level
        self.supports_logprobs = access_level != "L0"
        self.scorer = scorer
        self.keywords = {k: [w.lower() for w in v] for k, v in (keywords or {}).items()}
        self.sharpness = sharpness
        self.noise = noise
        self.seed = seed
        self._draws = 0
        super().__init__("mock-model", temperature=1.0, name=name or "mock")

    # -- scoring ---------------------------------------------------------------

    def _probabilities(self, context: Any, instructions: Any, options: Mapping[str, Optional[str]]) -> dict[str, float]:
        if self.scorer is not None:
            raw = dict(self.scorer(context, instructions, options))
            total = sum(max(v, 0.0) for v in raw.values()) or 1.0
            return {k: max(raw.get(k, 0.0), 0.0) / total for k in options}
        context_words = _words(context)
        if set(options) == {"true", "false"}:
            # Yes/no questions: the fraction of the question's specific words
            # (the text after "Claim:" when present) found in the context.
            text = _prompts.render_instructions(instructions)
            focus = text.split("Claim:", 1)[1] if "Claim:" in text else text
            words = set(_WORD.findall(focus.lower())) - _STOPWORDS
            ratio = len(words & context_words) / len(words) if words else 0.5
            jitter = self.noise * (_stable_uniform(self.seed, _prompts.render_context(context), text) - 0.5)
            p_true = 1.0 / (1.0 + math.exp(-(self.sharpness * 2.0 * (ratio - 0.6) + jitter)))
            return {"true": p_true, "false": 1.0 - p_true}
        logits = {}
        for key, description in options.items():
            vocabulary = set(_WORD.findall(f"{key} {description or ''}".lower())) | set(self.keywords.get(key, []))
            overlap = len(vocabulary & context_words)
            jitter = self.noise * (_stable_uniform(self.seed, _prompts.render_context(context), key) - 0.5) * 2
            logits[key] = self.sharpness * overlap + jitter
        peak = max(logits.values())
        weights = {k: math.exp(v - peak) for k, v in logits.items()}
        total = sum(weights.values())
        return {k: w / total for k, w in weights.items()}

    def score_options(self, context: Any, instructions: Any, options: Mapping[str, Optional[str]]) -> dict[str, float]:
        if not self.supports_logprobs:
            raise NotImplementedError("MockEvidenceBackend(access_level='L0') samples instead of scoring")
        return self._probabilities(context, instructions, options)

    def sample_options(self, context: Any, instructions: Any, options: Mapping[str, Optional[str]], n: int) -> list[str]:
        probs = self._probabilities(context, instructions, options)
        seed_material = _stable_uniform(self.seed, _prompts.render_context(context), self._draws)
        self._draws += 1
        rng = np.random.default_rng(int(seed_material * 2**32))
        keys = list(probs)
        return list(rng.choice(keys, size=n, p=[probs[k] for k in keys]))

    # -- text ---------------------------------------------------------------------

    def _complete(self, system: str, user: str, *, n: int, temperature: float, max_tokens: int) -> list[str]:
        return [self._reply(user) for _ in range(n)]

    def generate(self, context: Any, instructions: Any, max_tokens: Optional[int] = None) -> str:
        text = ""
        if isinstance(context, Mapping):
            for key in ("draft_answer", "answer", "summary", "text"):
                if isinstance(context.get(key), str):
                    text = context[key]
                    break
        if not text:
            text = _prompts.render_context(context)
        return json.dumps(_prompts.split_sentences(text))

    @staticmethod
    def _reply(user: str) -> str:
        return "This is a mock reply. Configure a real evidence backend for real answers."

    def fingerprint(self) -> dict[str, Any]:
        fp = super().fingerprint()
        fp.update({"seed": self.seed, "sharpness": self.sharpness, "noise": self.noise,
                   "keywords": self.keywords, "custom_scorer": self.scorer is not None})
        return fp
