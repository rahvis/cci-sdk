"""The shared Claude chat model both pipelines use to draft text.

Same model, same key, in both arms — the only variable between the
baseline and CCI pipelines is whether CCI's calibrated layer sits in
front of what gets auto-sent.
"""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic

from common.env import anthropic_model


def chat_model(**overrides):
    return ChatAnthropic(
        model=anthropic_model(),
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=1024,
        **overrides,
    )
