"""Synthetic AML account-hold dataset generator.

SYNTHETIC DATA FOR DEMONSTRATION ONLY. Not a real AML/compliance system.
Seeded and deterministic; no network access.

Ground truth is whether holding the account is actually warranted. Traps
run in both directions: alerts that read as routine but hide a real
pattern (should be held), and alerts that read as suspicious on the
surface but are legitimate (should be cleared) — AML has a real cost
on both sides.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SEED = 20260929
OUT_DIR = Path(__file__).resolve().parent
N_CALIBRATION = 40
N_TEST = 12


def _base_facts(rng: random.Random) -> dict[str, Any]:
    return {
        "transaction_amount": round(rng.uniform(500, 50000), -2),
        "country_risk": rng.choice(["low", "medium", "high"]),
        "customer_tenure_years": round(rng.uniform(0.2, 12), 1),
        "prior_sar_count": 0,
        "structuring_pattern": False,
        "pep_status": False,
        "unusual_velocity_change": False,
        "business_type": rng.choice(["retail", "import_export", "consulting", "individual", "ecommerce"]),
    }


def make_warranted_case(rng: random.Random, subtle: bool) -> dict[str, Any]:
    facts = _base_facts(rng)
    kind = rng.choice(["structuring", "pep_high_risk", "velocity_spike"])
    if kind == "structuring":
        facts["structuring_pattern"] = True
        facts["transaction_amount"] = round(rng.uniform(8500, 9900), -1)  # just under a $10k reporting line
        note = "several transactions just under the reporting threshold this week"
    elif kind == "pep_high_risk":
        facts["pep_status"] = True
        facts["country_risk"] = "high"
        note = "politically exposed person, high-risk jurisdiction"
    else:
        facts["unusual_velocity_change"] = True
        facts["customer_tenure_years"] = round(rng.uniform(3, 10), 1)
        note = "account dormant for months, then a sudden burst of activity"

    if subtle:
        # Phrase it casually / low-key, the way a written-up alert summary
        # sometimes undersells a real pattern.
        message = (
            f"Routine-looking alert, {facts['business_type']} customer, account open "
            f"{facts['customer_tenure_years']} years. Nothing dramatic, but flagged automatically: {note}."
        )
    else:
        message = (
            f"AML alert: {facts['business_type']} customer, ${facts['transaction_amount']:,.0f} transaction, "
            f"{facts['country_risk']} country risk. {note.capitalize()}."
        )
    return {"facts": facts, "message": message, "warranted": True, "is_trap": subtle}


def make_clear_case(rng: random.Random, alarming_surface: bool) -> dict[str, Any]:
    facts = _base_facts(rng)
    if alarming_surface:
        facts["transaction_amount"] = round(rng.uniform(20000, 60000), -2)
        facts["country_risk"] = "high"
        facts["business_type"] = "import_export"
        facts["customer_tenure_years"] = round(rng.uniform(5, 15), 1)
        message = (
            f"AML alert: established {facts['business_type']} customer of "
            f"{facts['customer_tenure_years']} years, ${facts['transaction_amount']:,.0f} international wire to a "
            f"{facts['country_risk']}-risk country. Matches this customer's known supplier-payment pattern on file; "
            f"documentation for the underlying trade is attached and consistent with prior payments."
        )
    else:
        message = (
            f"AML alert: {facts['business_type']} customer, ${facts['transaction_amount']:,.0f} transaction, "
            f"{facts['country_risk']} country risk, {facts['customer_tenure_years']} years on file. No prior SARs, "
            "no structuring, no PEP status, no unusual velocity change."
        )
    return {"facts": facts, "message": message, "warranted": False, "is_trap": alarming_surface}


def generate(rng: random.Random, n: int, trap_fraction: float) -> list[dict[str, Any]]:
    n_trap_each = max(1, round(n * trap_fraction / 2))
    cases = []
    remaining = n
    for _ in range(n_trap_each):
        cases.append(make_warranted_case(rng, subtle=True))
        remaining -= 1
    for _ in range(n_trap_each):
        cases.append(make_clear_case(rng, alarming_surface=True))
        remaining -= 1
    half = remaining // 2
    for _ in range(half):
        cases.append(make_warranted_case(rng, subtle=False))
    for _ in range(remaining - half):
        cases.append(make_clear_case(rng, alarming_surface=False))
    rng.shuffle(cases)
    return cases


def main() -> None:
    rng = random.Random(SEED)
    calibration = generate(rng, N_CALIBRATION, trap_fraction=0.2)
    test = generate(rng, N_TEST, trap_fraction=0.5)

    path = OUT_DIR / "aml_cases.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i, case in enumerate(calibration):
            fh.write(json.dumps({"case_id": f"AC-{i + 1:04d}", "split": "calibration", **case}) + "\n")
        for i, case in enumerate(test):
            fh.write(json.dumps({"case_id": f"AT-{i + 1:04d}", "split": "test", **case}) + "\n")

    n_warranted = sum(1 for c in calibration if c["warranted"])
    print(f"wrote {len(calibration)} calibration ({n_warranted} warranted) + {len(test)} test cases to {path}")


if __name__ == "__main__":
    main()
