"""Create and populate the three calibration profiles against the real hosted CCI API.

Run once before cci/run.py. Only creates profiles and uploads raw labelled
examples (fast, no model calls); the hosted API scores them for real,
lazily, the first time cci/run.py's graph actually evaluates each query
(see services/stats-api/app/evaluate.py's just-in-time calibration).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_cases, load_claim_calibration
from common.env import load_env

LEVELS = ("self_care", "primary_care", "urgent_care", "emergency")


def adjacent_wrong(level: str) -> str:
    i = LEVELS.index(level)
    return LEVELS[i - 1] if i > 0 else LEVELS[i + 1]


def main() -> None:
    load_env()
    from cli_sdk import CLIClient

    from cci.queries import (
        CLAIM_ALPHA,
        CLAIM_PROFILE,
        GATE_PROFILE,
        GATE_TARGET,
        SET_ALPHA,
        SET_PROFILE,
    )

    calibration_cases, _ = load_cases()
    claim_examples_raw = load_claim_calibration()
    rng = random.Random(7)

    backend = {"provider": "anthropic", "model": "claude-sonnet-5"}

    with CLIClient() as client:
        print(f"creating profiles against {client.base_url} ...", file=sys.stderr)

        client.calibration_profiles.create(name=SET_PROFILE, backend=backend, method="APS", alpha=SET_ALPHA)
        set_examples = [
            {"context": {"message": c["message"], "facts": c["facts"]}, "label": c["urgency"]}
            for c in calibration_cases
        ]
        client.calibration_profiles.add_examples(SET_PROFILE, examples=set_examples)
        print(f"  {SET_PROFILE}: {len(set_examples)} examples", file=sys.stderr)

        client.calibration_profiles.create(name=GATE_PROFILE, backend=backend, method="CRC", alpha=GATE_TARGET)
        gate_examples = []
        for c in calibration_cases:
            propose_correct = rng.random() < 0.6
            proposed = c["urgency"] if propose_correct else adjacent_wrong(c["urgency"])
            gate_examples.append({
                "context": {"message": c["message"], "facts": c["facts"], "proposed_urgency": proposed},
                "label": proposed == c["urgency"],
            })
        client.calibration_profiles.add_examples(GATE_PROFILE, examples=gate_examples)
        print(f"  {GATE_PROFILE}: {len(gate_examples)} examples", file=sys.stderr)

        client.calibration_profiles.create(name=CLAIM_PROFILE, backend=backend, method="conformal-factuality", alpha=CLAIM_ALPHA)
        client.calibration_profiles.add_examples(CLAIM_PROFILE, examples=claim_examples_raw)
        print(f"  {CLAIM_PROFILE}: {len(claim_examples_raw)} examples", file=sys.stderr)

    print("done. Profiles are registered; each will calibrate for real on its first evaluate() call.", file=sys.stderr)


if __name__ == "__main__":
    main()
