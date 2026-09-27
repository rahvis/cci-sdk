"""Run the baseline (no-CCI) pipeline over the held-out test cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_cases
from common.env import load_env

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "baseline.jsonl"


def main() -> None:
    load_env()
    from baseline.graph import build_graph

    _, test_cases = load_cases()
    graph = build_graph()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        for case in test_cases:
            print(f"[baseline] {case['case_id']} ({case['urgency']}, trap={case['is_trap']}) ...", file=sys.stderr)
            out = graph.invoke({"message": case["message"], "facts": case["facts"]})
            record = {
                "case_id": case["case_id"],
                "true_urgency": case["urgency"],
                "is_trap": case["is_trap"],
                "predicted_urgency": out["assessment"]["urgency"],
                "confidence": out["assessment"]["confidence"],
                "explanation": out["assessment"]["explanation"],
                "auto_sent": out["auto_sent"],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"wrote {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
