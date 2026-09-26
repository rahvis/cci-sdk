"""Generate ``trial_screening.jsonl``: synthetic clinical-trial pre-screening records.

Every patient, record, value and the protocol itself are invented for this
demonstration. The protocol below is fictional and simplified. Nothing here
is medical advice, and the data must not be used to decide anyone's care or
trial eligibility.

Each output line is ``{"id": ..., "context": {...}, "label": ...}``:

- ``context`` is exactly what the calibrated guard evaluates: the written
  protocol plus the pre-screening record, produced by ``screening_context()``.
  The example agent builds its inference-time context with the same
  function, so calibration and serving contexts have identical structure
  and wording.
- ``label`` is ``True`` when the patient was confirmed eligible at the
  screening visit (every inclusion criterion met, no exclusion criterion).

How labels are made
-------------------
The screening visit re-measures eGFR and HbA1c and confirms the medication
history, and the protocol applies to those visit values. The pre-screening
record only holds the most recent values on file, so a patient whose eGFR is
59 or whose HbA1c is 10.4 may or may not qualify at the visit: those cases
are genuinely ambiguous from the record alone, which is what keeps the
calibration realistic instead of trivially separable. About 1.5% of patients
also turn out to have an exclusion that was not in the record.

Deterministic (seeded) and standard library only::

    python examples/agents/data/generate_trial_screening.py            # writes trial_screening.jsonl
    python examples/agents/data/generate_trial_screening.py --stats    # also prints label balance
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional

SEED = 20260925
DEFAULT_COUNT = 320
OUTPUT = Path(__file__).resolve().parent / "trial_screening.jsonl"

TRIAL_ID = "SYN-CKD-201"

# The protocol travels inside every context, word for word, so the evidence
# model reads the same criteria at calibration and at serving time.
PROTOCOL = [
    f"Screening protocol {TRIAL_ID} (fictional, for demonstration only): adults with type 2 "
    "diabetes and stage 3 chronic kidney disease.",
    "Criteria apply to the values measured at the screening visit.",
    "Inclusion criteria (all must be met):",
    "I1. Age 18 to 75 years inclusive.",
    "I2. Diagnosis of type 2 diabetes mellitus.",
    "I3. eGFR 30 to 59 mL/min/1.73 m2 inclusive.",
    "I4. HbA1c 7.0% to 10.5% inclusive.",
    "Exclusion criteria (any one excludes):",
    "E1. Any SGLT2 inhibitor taken within the 12 weeks before screening.",
    "E2. Type 1 diabetes, current dialysis, or a kidney transplant.",
    "E3. ALT above 3 times the upper limit of normal.",
    "E4. Pregnant, breastfeeding, or planning pregnancy during the study.",
    "E5. Participation in another interventional study within the last 30 days.",
]

ELIGIBILITY_QUESTION = (
    "Does this patient meet every inclusion criterion and no exclusion criterion of the protocol "
    "in the context?"
)

SGLT2 = ["empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin"]
OTHER_DRUGS = ["metformin", "insulin glargine", "sitagliptin", "semaglutide", "lisinopril",
               "atorvastatin", "amlodipine", "losartan"]
COMORBIDITIES = ["hypertension", "hyperlipidemia", "obesity", "osteoarthritis", "hypothyroidism",
                 "gout", "sleep apnea"]


def screening_context(record: dict[str, Any]) -> dict[str, Any]:
    """The context the guard evaluates for one patient, at calibration and at serving time."""
    return {
        "protocol": list(PROTOCOL),
        "patient": {
            "patient_id": record["patient_id"],
            "age_years": record["age_years"],
            "sex": record["sex"],
            "diagnoses": list(record["diagnoses"]),
            "labs_most_recent": dict(record["labs_most_recent"]),
            "medications": [dict(m) for m in record["medications"]],
            "dialysis_or_kidney_transplant": record["dialysis_or_kidney_transplant"],
            "pregnant_breastfeeding_or_planning": record["pregnant_breastfeeding_or_planning"],
            "interventional_study_last_30_days": record["interventional_study_last_30_days"],
        },
    }


def criteria_met(record: dict[str, Any], egfr: float, hba1c: float, sglt2_weeks: Optional[float]) -> bool:
    """The protocol, applied to one set of values (the record's or the visit's)."""
    diagnoses = record["diagnoses"]
    inclusion = (
        18 <= record["age_years"] <= 75
        and "type 2 diabetes mellitus" in diagnoses
        and 30 <= egfr <= 59
        and 7.0 <= hba1c <= 10.5
    )
    exclusion = (
        (sglt2_weeks is not None and sglt2_weeks < 12)
        or "type 1 diabetes mellitus" in diagnoses
        or record["dialysis_or_kidney_transplant"]
        or record["labs_most_recent"]["alt_x_uln"] > 3.0
        or record["pregnant_breastfeeding_or_planning"]
        or record["interventional_study_last_30_days"]
    )
    return inclusion and not exclusion


def sglt2_weeks_since(record: dict[str, Any]) -> Optional[float]:
    """Weeks since the most recent SGLT2 inhibitor dose (0 = still taking), or None."""
    weeks = [m["weeks_since_last_dose"] for m in record["medications"] if m["drug_class"] == "SGLT2 inhibitor"]
    return min(weeks) if weeks else None


def eligible_on_record(record: dict[str, Any]) -> bool:
    labs = record["labs_most_recent"]
    return criteria_met(record, labs["egfr_ml_min_1_73m2"], labs["hba1c_percent"], sglt2_weeks_since(record))


# ---------------------------------------------------------------------------
# synthetic patients
# ---------------------------------------------------------------------------


def _meds(rng: random.Random, sglt2_weeks: Optional[int]) -> list[dict[str, Any]]:
    meds = [{"drug": d, "drug_class": "other", "weeks_since_last_dose": 0}
            for d in rng.sample(OTHER_DRUGS, rng.randint(1, 4))]
    if sglt2_weeks is not None:
        meds.append({"drug": rng.choice(SGLT2), "drug_class": "SGLT2 inhibitor",
                     "weeks_since_last_dose": sglt2_weeks})
    return meds


def _base(rng: random.Random) -> dict[str, Any]:
    sex = rng.choice(["female", "male"])
    return {
        "age_years": rng.randint(32, 74),
        "sex": sex,
        "diagnoses": ["type 2 diabetes mellitus", "chronic kidney disease stage 3"]
        + rng.sample(COMORBIDITIES, rng.randint(0, 3)),
        "labs_most_recent": {"egfr_ml_min_1_73m2": rng.randint(33, 56),
                             "hba1c_percent": round(rng.uniform(7.3, 10.1), 1),
                             "alt_x_uln": round(rng.uniform(0.4, 1.8), 1)},
        "medications": _meds(rng, rng.choice([None, None, None, rng.randint(20, 104)])),
        "dialysis_or_kidney_transplant": False,
        "pregnant_breastfeeding_or_planning": False,
        "interventional_study_last_30_days": False,
    }


def _clear_eligible(rng):
    return _base(rng)


def _recent_sglt2(rng):
    r = _base(rng)
    r["medications"] = _meds(rng, rng.choice([0, 0, rng.randint(1, 10)]))
    return r


def _labs_out_of_range(rng):
    r = _base(rng)
    labs = r["labs_most_recent"]
    which = rng.choice(["egfr_low", "egfr_high", "a1c_low", "a1c_high"])
    if which == "egfr_low":
        labs["egfr_ml_min_1_73m2"] = rng.randint(15, 26)
    elif which == "egfr_high":
        labs["egfr_ml_min_1_73m2"] = rng.randint(64, 88)
        r["diagnoses"] = [d for d in r["diagnoses"] if d != "chronic kidney disease stage 3"]
    elif which == "a1c_low":
        labs["hba1c_percent"] = round(rng.uniform(5.6, 6.7), 1)
    else:
        labs["hba1c_percent"] = round(rng.uniform(10.9, 12.8), 1)
    return r


def _other_exclusion(rng):
    r = _base(rng)
    which = rng.choice(["alt", "dialysis", "type1", "pregnancy", "study"])
    if which == "alt":
        r["labs_most_recent"]["alt_x_uln"] = round(rng.uniform(3.4, 6.5), 1)
    elif which == "dialysis":
        r["dialysis_or_kidney_transplant"] = True
    elif which == "type1":
        r["diagnoses"] = ["type 1 diabetes mellitus"] + r["diagnoses"][1:]
    elif which == "pregnancy":
        r["sex"] = "female"
        r["age_years"] = rng.randint(24, 44)
        r["pregnant_breastfeeding_or_planning"] = True
    else:
        r["interventional_study_last_30_days"] = True
    return r


def _age_out_of_range(rng):
    r = _base(rng)
    r["age_years"] = rng.randint(77, 86)
    return r


def _near_cutoff(rng):
    r = _base(rng)
    labs = r["labs_most_recent"]
    which = rng.choice(["egfr_top", "egfr_bottom", "a1c_top", "a1c_bottom", "washout"])
    if which == "egfr_top":
        labs["egfr_ml_min_1_73m2"] = rng.randint(57, 62)
    elif which == "egfr_bottom":
        labs["egfr_ml_min_1_73m2"] = rng.randint(28, 32)
    elif which == "a1c_top":
        labs["hba1c_percent"] = round(rng.uniform(10.2, 10.8), 1)
    elif which == "a1c_bottom":
        labs["hba1c_percent"] = round(rng.uniform(6.8, 7.2), 1)
    else:
        r["medications"] = _meds(rng, rng.randint(11, 13))
    return r


ARCHETYPES = [
    (_clear_eligible, 0.40), (_recent_sglt2, 0.12), (_labs_out_of_range, 0.14),
    (_other_exclusion, 0.10), (_age_out_of_range, 0.04), (_near_cutoff, 0.20),
]


def screening_visit_eligible(rng: random.Random, record: dict[str, Any]) -> bool:
    """Eligibility at the screening visit, which re-measures labs and confirms medication dates."""
    labs = record["labs_most_recent"]
    egfr = round(labs["egfr_ml_min_1_73m2"] + rng.gauss(0.0, 2.0))
    hba1c = round(labs["hba1c_percent"] + rng.gauss(0.0, 0.15), 1)
    weeks = sglt2_weeks_since(record)
    if weeks is not None and weeks > 0:
        weeks = weeks + rng.uniform(-1.0, 1.0)  # the record rounds to whole weeks
    eligible = criteria_met(record, egfr, hba1c, weeks)
    if eligible and rng.random() < 0.015:
        eligible = False  # an exclusion found at the visit that was not in the record
    return eligible


def make_record(rng: random.Random, patient_id: str) -> tuple[dict[str, Any], bool]:
    builder = rng.choices([a for a, _ in ARCHETYPES], weights=[w for _, w in ARCHETYPES])[0]
    record = builder(rng)
    record["patient_id"] = patient_id
    return record, screening_visit_eligible(rng, record)


def generate(count: int = DEFAULT_COUNT, *, seed: int = SEED, first_id: int = 1001,
             prefix: str = "P") -> list[dict[str, Any]]:
    """``count`` labelled examples, ``{"id", "context", "label"}``, deterministically from ``seed``."""
    rng = random.Random(seed)
    out = []
    for i in range(count):
        patient_id = f"{prefix}-{first_id + i}"
        record, label = make_record(rng, patient_id)
        out.append({"id": patient_id, "context": screening_context(record), "label": label})
    return out


def write(path: Path = OUTPUT, count: int = DEFAULT_COUNT, seed: int = SEED) -> list[dict[str, Any]]:
    rows = generate(count, seed=seed)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic trial pre-screening calibration set.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--stats", action="store_true", help="print label balance against the record-only rule")
    args = parser.parse_args(argv)
    rows = write(args.out, args.count, args.seed)
    eligible = sum(r["label"] for r in rows)
    print(f"wrote {len(rows)} records to {args.out} ({eligible} eligible at the visit, {len(rows) - eligible} not)")
    if args.stats:
        table = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
        for r in rows:
            table[(eligible_on_record(r["context"]["patient"]), bool(r["label"]))] += 1
        for (on_record, at_visit), n in sorted(table.items()):
            print(f"  eligible on record={on_record!s:<5}  eligible at visit={at_visit!s:<5}  {n:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
