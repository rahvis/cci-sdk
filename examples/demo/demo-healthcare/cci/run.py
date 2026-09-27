"""Run the CCI-guarded pipeline over the held-out test cases, against the
real hosted API (https://cci.gitdate.ink/api/v1), using CCI_API_KEY.

Run cci/build_calibration.py first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_cases
from common.env import load_env

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "cci.jsonl"


def main() -> None:
    load_env()
    from cli_sdk import CLIClient

    from cci.graph import build_graph

    _, test_cases = load_cases()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLIClient() as client:
        graph = build_graph(client)
        with RESULTS_PATH.open("w", encoding="utf-8") as fh:
            for case in test_cases:
                print(f"[cci] {case['case_id']} ({case['urgency']}, trap={case['is_trap']}) ...", file=sys.stderr)
                out = graph.invoke({"message": case["message"], "facts": case["facts"]})
                record = {
                    "case_id": case["case_id"],
                    "true_urgency": case["urgency"],
                    "is_trap": case["is_trap"],
                    "predicted_urgency": out["set_top"],
                    "predicted_set": out.get("set_full", []),
                    "set_is_singleton": out["set_is_singleton"],
                    "set_guarantee": out["set_guarantee"],
                    "gate_decision": out.get("gate_decision"),
                    "gate_guarantee": out.get("gate_guarantee"),
                    "explanation": out.get("explanation", ""),
                    "retained_claims": out.get("retained_claims", []),
                    "dropped_claims": out.get("dropped_claims", []),
                    "claim_guarantee": out.get("claim_guarantee"),
                    "auto_sent": out["auto_sent"],
                    "final_text": out["final_text"],
                    "escalation_reason": out["escalation_reason"],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
    print(f"wrote {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
