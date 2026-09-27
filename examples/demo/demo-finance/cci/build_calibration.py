"""Create and populate the three calibration profiles against the real hosted CCI API.

Run once before cci/run_lending.py and cci/run_aml.py. Only registers
profiles and uploads raw labelled examples (fast, no model calls); the
hosted API scores them for real, lazily, the first time each pipeline
actually evaluates a query (just-in-time calibration).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_aml_cases, load_lending_cases
from common.env import load_env

BACKEND = {"provider": "anthropic", "model": "claude-sonnet-5"}


def main() -> None:
    load_env()
    from cli_sdk import CLIClient

    from cci.queries import (
        AML_GATE_DELTA,
        AML_GATE_PROFILE,
        AML_GATE_TARGET,
        LENDING_GATE_PROFILE,
        LENDING_GATE_TARGET,
        LENDING_SET_ALPHA,
        LENDING_SET_PROFILE,
    )

    lending_calibration, _ = load_lending_cases()
    aml_calibration, _ = load_aml_cases()

    with CLIClient() as client:
        print(f"creating profiles against {client.base_url} ...", file=sys.stderr)

        # --- Lending: Set with Mondrian groups ---
        client.calibration_profiles.create(
            name=LENDING_SET_PROFILE, backend=BACKEND, method="APS", alpha=LENDING_SET_ALPHA, group_by="channel"
        )
        set_examples = [
            {
                "context": {"message": c["message"], "facts": c["facts"], "channel": c["facts"]["channel"]},
                "label": c["tier"],
                "group": c["facts"]["channel"],
            }
            for c in lending_calibration
        ]
        client.calibration_profiles.add_examples(LENDING_SET_PROFILE, examples=set_examples)
        print(f"  {LENDING_SET_PROFILE}: {len(set_examples)} examples", file=sys.stderr)

        # --- Lending: Gate(risk) on "is approving the proposed tier correct?" ---
        client.calibration_profiles.create(
            name=LENDING_GATE_PROFILE, backend=BACKEND, method="CRC", alpha=LENDING_GATE_TARGET
        )
        # Proposal mix: 65% propose the true tier (label True), 35% propose an
        # adjacent tier (label False), same pattern as demo-healthcare's Gate.
        tiers_order = ["prime", "near_prime", "subprime", "decline"]
        gate_examples = []
        for i, c in enumerate(lending_calibration):
            propose_correct = (i % 3) != 0
            if propose_correct:
                proposed = c["tier"]
            else:
                idx = tiers_order.index(c["tier"])
                proposed = tiers_order[idx - 1] if idx > 0 else tiers_order[idx + 1]
            gate_examples.append({
                "context": {"message": c["message"], "facts": c["facts"], "proposed_tier": proposed},
                "label": proposed == c["tier"],
            })
        client.calibration_profiles.add_examples(LENDING_GATE_PROFILE, examples=gate_examples)
        print(f"  {LENDING_GATE_PROFILE}: {len(gate_examples)} examples", file=sys.stderr)

        # --- AML: Gate(fdr) on "is holding the account correct?" ---
        client.calibration_profiles.create(
            name=AML_GATE_PROFILE, backend=BACKEND, method="LTT", alpha=AML_GATE_TARGET
        )
        aml_examples = [
            {"context": {"message": c["message"], "facts": c["facts"]}, "label": c["warranted"]}
            for c in aml_calibration
        ]
        client.calibration_profiles.add_examples(AML_GATE_PROFILE, examples=aml_examples)
        print(f"  {AML_GATE_PROFILE}: {len(aml_examples)} examples", file=sys.stderr)

    print("done. Profiles are registered; each will calibrate for real on its first evaluate() call.", file=sys.stderr)


if __name__ == "__main__":
    main()
