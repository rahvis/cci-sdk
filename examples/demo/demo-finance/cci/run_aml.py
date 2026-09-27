"""Run the CCI-guarded AML pipeline over the held-out test cases,
against the real hosted API, using CCI_API_KEY. Run build_calibration.py first."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_aml_cases
from common.env import load_env

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "cci_aml.jsonl"


def main() -> None:
    load_env()
    from cli_sdk import CLIClient

    from cci.aml_graph import build_graph

    _, test_cases = load_aml_cases()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLIClient() as client:
        graph = build_graph(client)
        with RESULTS_PATH.open("w", encoding="utf-8") as fh:
            for case in test_cases:
                print(f"[cci/aml] {case['case_id']} (warranted={case['warranted']}, trap={case['is_trap']}) ...",
                      file=sys.stderr)
                out = graph.invoke({"message": case["message"], "facts": case["facts"]})
                record = {
                    "case_id": case["case_id"],
                    "true_warranted": case["warranted"],
                    "is_trap": case["is_trap"],
                    "gate_decision": out["gate_decision"],
                    "gate_guarantee": out["gate_guarantee"],
                    "auto_held": out["auto_held"],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
    print(f"wrote {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
