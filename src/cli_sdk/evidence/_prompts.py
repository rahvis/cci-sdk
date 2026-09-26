"""Prompt rendering and answer parsing shared by every evidence backend.

Evidence backends ask the user's own model a closed-form question and read
back either (a) the probability of each option's label token (access level
L1, when the provider returns token log-probabilities) or (b) the option it
names when sampled (access level L0). Both paths use the same rendering so a
calibration profile built on one is only ever compared against scores built
the same way — the prompt is part of the scoring function.
"""

from __future__ import annotations

import json
import re
import string
from typing import Any, Mapping, Optional

# Single-character labels are one token in every mainstream tokenizer, which
# is what makes first-token log-probabilities a faithful option score.
OPTION_LETTERS = tuple(string.ascii_uppercase)
MAX_LETTER_OPTIONS = 20  # OpenAI-style APIs return at most 20 top_logprobs

INVALID = "__invalid__"

SYSTEM_PROMPT = (
    "You are a careful decision component inside a larger software system. "
    "Answer only from the provided context. Follow the answer format exactly."
)


def render_context(context: Any) -> str:
    """Render context the same way every time (sorted-key JSON for structures)."""
    if isinstance(context, str):
        return context
    return json.dumps(context, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def render_instructions(instructions: Any) -> str:
    if isinstance(instructions, str):
        return instructions
    return json.dumps(instructions, indent=2, sort_keys=True, ensure_ascii=False, default=str)


def letter_map(options: Mapping[str, Optional[str]]) -> dict[str, str]:
    """Map each option key to a single-letter label, in declaration order."""
    keys = list(options)
    if len(keys) > len(OPTION_LETTERS):
        raise ValueError(f"at most {len(OPTION_LETTERS)} options are supported")
    return {OPTION_LETTERS[i]: key for i, key in enumerate(keys)}


def letter_prompt(context: Any, instructions: Any, options: Mapping[str, Optional[str]]) -> tuple[str, str, dict[str, str]]:
    """(system, user, letter->option) for first-token log-probability scoring."""
    letters = letter_map(options)
    lines = []
    for letter, key in letters.items():
        description = options[key]
        lines.append(f"{letter}. {key}" + (f" — {description}" if description else ""))
    user = (
        f"Context:\n{render_context(context)}\n\n"
        f"Question:\n{render_instructions(instructions)}\n\n"
        "Options:\n" + "\n".join(lines) + "\n\n"
        "Reply with the single letter of the best option and nothing else."
    )
    return SYSTEM_PROMPT, user, letters


def naming_prompt(context: Any, instructions: Any, options: Mapping[str, Optional[str]]) -> tuple[str, str]:
    """(system, user) for sampling: the model names one option key verbatim."""
    lines = [f"- {key}" + (f": {desc}" if desc else "") for key, desc in options.items()]
    user = (
        f"Context:\n{render_context(context)}\n\n"
        f"Question:\n{render_instructions(instructions)}\n\n"
        "Choose exactly one of these options:\n" + "\n".join(lines) + "\n\n"
        "Reply with the option name exactly as written above and nothing else."
    )
    return SYSTEM_PROMPT, user


def generation_prompt(context: Any, instructions: Any) -> tuple[str, str]:
    user = f"Context:\n{render_context(context)}\n\nTask:\n{render_instructions(instructions)}"
    return SYSTEM_PROMPT, user


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_option(text: str, options: Mapping[str, Optional[str]]) -> str:
    """Map a sampled reply to an option key, or INVALID.

    Exact (normalized) match first, then a unique option whose normalized key
    appears as a whole word in the reply. Ambiguous or unmatched replies are
    INVALID rather than guessed, because a guessed label would silently bias
    the frequency score.
    """
    reply = _normalize(text or "")
    normalized = {_normalize(key): key for key in options}
    if reply in normalized:
        return normalized[reply]
    hits = [key for norm, key in normalized.items() if norm and re.search(rf"\b{re.escape(norm)}\b", reply)]
    return hits[0] if len(hits) == 1 else INVALID


def parse_letter(text: str, letters: Mapping[str, str], options: Mapping[str, Optional[str]]) -> str:
    """Map a sampled reply to the letter prompt onto an option key, or INVALID.

    Accepts "A", "A.", "(A)", "a" or "A. refund" (a leading option letter),
    and falls back to ``parse_option`` when the reply names the option instead.
    """
    stripped = (text or "").strip()
    match = re.match(r"^[\(\[]?([A-Za-z])(?:[\)\].:,\s]|$)", stripped)
    if match:
        letter = match.group(1).upper()
        if letter in letters:
            return letters[letter]
    return parse_option(stripped, options)


def letter_probabilities(top_logprobs: list[tuple[str, float]], letters: Mapping[str, str]) -> dict[str, float]:
    """Turn first-token (token, logprob) pairs into per-option probabilities.

    Variants such as " A", "A.", "a" are merged. Options that do not appear in
    the returned top-k share the residual probability mass equally — a fixed
    rule, identical at calibration and inference time. The result is
    renormalized over the options.
    """
    import math

    mass = {key: 0.0 for key in letters.values()}
    seen_total = 0.0
    for token, logprob in top_logprobs:
        if logprob is None or logprob <= -9999:
            continue
        p = math.exp(logprob)
        seen_total += p
        letter = token.strip().rstrip(".):").upper()
        if letter in letters:
            mass[letters[letter]] += p
    missing = [key for key, value in mass.items() if value == 0.0]
    residual = max(0.0, 1.0 - seen_total)
    for key in missing:
        mass[key] = residual / len(missing) if residual > 0 else 1e-9
    total = sum(mass.values())
    return {key: value / total for key, value in mass.items()}


def extract_json_list(text: str) -> Optional[list[str]]:
    """Pull the first JSON array of strings out of a model reply."""
    match = re.search(r"\[.*\]", text or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except ValueError:
        return None
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return [v.strip() for v in value if v.strip()]
    return None


def split_sentences(text: str) -> list[str]:
    """Fallback claim splitter when the model does not return a JSON list."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]
