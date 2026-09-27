"""Run the baseline (no-CCI) lending pipeline over the held-out test cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_lending_cases
from common.env import load_env

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "baseline_lending.jsonl"


def main() -> None:
    load_env()
    from baseline.lending_graph import build_graph

    _, test_cases = load_lending_cases()
    graph = build_graph()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        for case in test_cases:
            print(f"[baseline/lending] {case['case_id']} ({case['facts']['channel']}, {case['tier']}, "
                  f"trap={case['is_trap']}) ...", file=sys.stderr)
            out = graph.invoke({"message": case["message"], "facts": case["facts"]})
            record = {
                "case_id": case["case_id"],
                "channel": case["facts"]["channel"],
                "true_tier": case["tier"],
                "is_trap": case["is_trap"],
                "predicted_tier": out["assessment"]["tier"],
                "confidence": out["assessment"]["confidence"],
                "rationale": out["assessment"]["rationale"],
                "auto_approved": out["auto_approved"],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"wrote {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
