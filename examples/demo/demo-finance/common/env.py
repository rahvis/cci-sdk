"""Load demo/.env and stop with a clear message if a key is still a placeholder."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

DEMO_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    load_dotenv(DEMO_ROOT / ".env")
    # The SDK reads CLI_API_KEY, not CCI_API_KEY; keep just one setting for
    # the person running the demo, per demo/.env.example.
    if os.environ.get("CCI_API_KEY") and not os.environ.get("CLI_API_KEY"):
        os.environ["CLI_API_KEY"] = os.environ["CCI_API_KEY"]

    missing = [name for name in ("ANTHROPIC_API_KEY", "CLI_API_KEY") if not os.environ.get(name)]
    if missing:
        print(
            f"Missing {', '.join(missing)} in demo/.env — copy .env.example to .env and "
            "fill it in from claude_api.md before running this.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def anthropic_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
