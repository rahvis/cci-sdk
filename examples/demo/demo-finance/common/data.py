"""Load the synthetic finance datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"{path} not found — run the matching data/generate_*.py script first.")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_lending_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_jsonl(DATA_DIR / "lending_cases.jsonl")
    return [r for r in rows if r["split"] == "calibration"], [r for r in rows if r["split"] == "test"]


def load_aml_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_jsonl(DATA_DIR / "aml_cases.jsonl")
    return [r for r in rows if r["split"] == "calibration"], [r for r in rows if r["split"] == "test"]
