"""Generate ``adverse_events.jsonl``: synthetic adverse-event case reports with seriousness grades.

Every record is synthetic: no real patient, reporter, product or safety
database is involved, and ``CLX-101`` is a fictitious product. The grading
guide below is invented for this example. It is loosely modelled on
five-level severity scales, and it is not a clinical, pharmacovigilance or
regulatory standard. In real pharmacovigilance, seriousness (for example
the ICH E2A criteria) and severity are different concepts, and expedited
reporting also depends on expectedness and causality. Nothing here is
medical or regulatory advice.

Each line is ``{"id": ..., "context": {...}, "label": ...}``:

- ``context`` is exactly what the calibrated guard evaluates, built by
  ``case_context`` (the example imports the same function, so calibration
  and inference contexts are identical in keys and rendering);
- ``label`` is the grade an assessor recorded by following ``GRADING_GUIDE``.

Two sources of realistic disagreement are built in:

- observation stays under 24 hours are left to the assessor's judgement by
  the guide, and the narrative wording only partly settles them;
- about 3% of cases carry a one-grade coding difference (never into or out
  of ``grade_5``), as double-coded safety data does.

Run (stdlib only, deterministic)::

    python examples/agents/data/generate_adverse_events.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

SEED = 8001
N_CASES = 300
CODING_NOISE_RATE = 0.03
OUTPUT = Path(__file__).resolve().parent / "adverse_events.jsonl"

GRADES = ("grade_1", "grade_2", "grade_3", "grade_4", "grade_5")
GRADE_NAMES = {"grade_1": "mild", "grade_2": "moderate", "grade_3": "severe",
               "grade_4": "life-threatening", "grade_5": "fatal"}
SUSPECT_PRODUCT = "CLX-101 (synthetic)"

OUTCOMES = ("recovered", "recovering", "not yet recovered", "recovered with lasting effects", "fatal")
HOSPITALIZATION = (
    "none",
    "emergency department visit, not admitted",
    "observation stay under 24 hours",
    "admitted to hospital",
    "existing hospital stay prolonged",
)
TREATMENT = (
    "none",
    "over-the-counter medicine",
    "prescription medicine",
    "intravenous therapy or procedure",
    "intensive care (ventilation or vasopressors)",
)
LAB_CHANGES = ("none", "mild abnormality", "moderate abnormality", "severe abnormality",
               "life-threatening abnormality")
DAILY_ACTIVITIES = ("not affected", "some activities limited", "unable to perform self-care")

GRADING_GUIDE = (
    "Synthetic seriousness grading guide (demonstration only; not a clinical or regulatory standard).",
    "Assign the highest grade whose criteria any finding in the case meets.",
    "grade_5 (fatal): the outcome is fatal.",
    "grade_4 (life-threatening): intensive care (ventilation or vasopressors), or a life-threatening "
    "laboratory abnormality.",
    "grade_3 (severe): hospital admission or a prolonged existing stay, intravenous therapy or a procedure, "
    "a severe laboratory abnormality, inability to perform self-care, or recovery with lasting effects.",
    "grade_2 (moderate): an emergency department visit without admission, prescription medicine, a moderate "
    "laboratory abnormality, or some activities limited.",
    "grade_1 (mild): none of the above; no treatment or over-the-counter medicine only.",
    "An observation stay under 24 hours is left to the assessor: grade_3 when the narrative shows the stay "
    "was medically necessary, grade_2 when it was precautionary.",
    "In this synthetic workflow, grades 3 to 5 qualify for an expedited report; grades 1 and 2 go to "
    "routine periodic reporting.",
)

# Grade index (0 = grade_1) each structured finding supports under the guide.
OUTCOME_GRADE = {"recovered": 0, "recovering": 0, "not yet recovered": 0,
                 "recovered with lasting effects": 2, "fatal": 4}
HOSPITALIZATION_GRADE = {HOSPITALIZATION[0]: 0, HOSPITALIZATION[1]: 1, HOSPITALIZATION[2]: 1,
                         HOSPITALIZATION[3]: 2, HOSPITALIZATION[4]: 2}
TREATMENT_GRADE = {TREATMENT[0]: 0, TREATMENT[1]: 0, TREATMENT[2]: 1, TREATMENT[3]: 2, TREATMENT[4]: 3}
LAB_GRADE = {LAB_CHANGES[0]: 0, LAB_CHANGES[1]: 0, LAB_CHANGES[2]: 1, LAB_CHANGES[3]: 2, LAB_CHANGES[4]: 3}
ACTIVITY_GRADE = {DAILY_ACTIVITIES[0]: 0, DAILY_ACTIVITIES[1]: 1, DAILY_ACTIVITIES[2]: 2}

# Observation-stay wording, with the probability an assessor records grade_3.
OBSERVATION_CUES = {
    "Kept overnight as a precaution; no treatment needed during the stay.": 0.15,
    "Kept overnight for monitoring.": 0.50,
    "Kept overnight because symptoms had not settled.": 0.85,
}

EVENT_TERMS = (
    ("injection-site pain", "headache", "rash", "nausea", "fatigue"),
    ("urticaria", "vomiting", "dizziness", "hepatic enzyme increased", "palpitations"),
    ("pancreatitis", "acute kidney injury", "dehydration", "syncope", "pneumonitis"),
    ("anaphylaxis", "ventricular arrhythmia", "neutropenic sepsis", "status epilepticus"),
    ("cardiac arrest", "respiratory failure"),
)


def case_context(case: Mapping[str, Any]) -> dict[str, Any]:
    """The evaluation context for one case: the same keys and formatting everywhere.

    ``case`` holds ``case_id``, ``event_term``, ``onset_days_after_dose`` (int),
    ``hospitalization``, ``treatment``, ``lab_changes``, ``daily_activities``,
    ``outcome`` and ``narrative``. Unknown values raise ``ValueError``, which
    the guard turns into an escalation.
    """
    for key, allowed in (("hospitalization", HOSPITALIZATION), ("treatment", TREATMENT),
                         ("lab_changes", LAB_CHANGES), ("daily_activities", DAILY_ACTIVITIES),
                         ("outcome", OUTCOMES)):
        if case[key] not in allowed:
            raise ValueError(f"unknown {key} {case[key]!r}")
    onset = int(case["onset_days_after_dose"])
    if onset < 0:
        raise ValueError("onset_days_after_dose must be >= 0")
    return {
        "case_id": str(case["case_id"]),
        "suspect_product": SUSPECT_PRODUCT,
        "event_term": str(case["event_term"]),
        "onset_days_after_dose": onset,
        "hospitalization": case["hospitalization"],
        "treatment": case["treatment"],
        "lab_changes": case["lab_changes"],
        "daily_activities": case["daily_activities"],
        "outcome": case["outcome"],
        "narrative": " ".join(str(case["narrative"]).split()),
    }


def structured_grade(case: Mapping[str, Any]) -> int:
    """Highest grade index the structured findings support (observation stays count as grade_2)."""
    return max(OUTCOME_GRADE[case["outcome"]], HOSPITALIZATION_GRADE[case["hospitalization"]],
               TREATMENT_GRADE[case["treatment"]], LAB_GRADE[case["lab_changes"]],
               ACTIVITY_GRADE[case["daily_activities"]])


def _values_up_to(values: tuple[str, ...], grades: Mapping[str, int], limit: int) -> list[str]:
    return [v for v in values if grades[v] <= limit]


def _case(rng: random.Random, case_id: str) -> dict[str, Any]:
    target = rng.choices(range(5), weights=(0.30, 0.30, 0.22, 0.12, 0.06), k=1)[0]
    ceiling = min(target, 3)  # structured findings never reach grade_5 except through the outcome
    case: dict[str, Any] = {
        "case_id": case_id,
        "event_term": rng.choice(EVENT_TERMS[target]),
        "onset_days_after_dose": rng.randint(0, 21),
        "hospitalization": rng.choice(_values_up_to(HOSPITALIZATION, HOSPITALIZATION_GRADE, ceiling)),
        "treatment": rng.choice(_values_up_to(TREATMENT, TREATMENT_GRADE, max(0, ceiling - 1))),
        "lab_changes": rng.choice(_values_up_to(LAB_CHANGES, LAB_GRADE, max(0, ceiling - 1))),
        "daily_activities": rng.choice(_values_up_to(DAILY_ACTIVITIES, ACTIVITY_GRADE, max(0, ceiling - 1))),
        "outcome": rng.choice(("recovered", "recovering", "not yet recovered")),
    }
    # One finding at the target grade drives it, as the guide requires.
    if target == 4:
        case["outcome"] = "fatal"
    elif target == 3:
        driver = rng.choice(("treatment", "lab_changes"))
        case[driver] = TREATMENT[4] if driver == "treatment" else LAB_CHANGES[4]
        case["hospitalization"] = rng.choice(HOSPITALIZATION[3:])
    elif target == 2:
        driver = rng.choice(("hospitalization", "treatment", "lab_changes", "daily_activities", "outcome"))
        case[driver] = {"hospitalization": rng.choice(HOSPITALIZATION[3:]), "treatment": TREATMENT[3],
                        "lab_changes": LAB_CHANGES[3], "daily_activities": DAILY_ACTIVITIES[2],
                        "outcome": "recovered with lasting effects"}[driver]
    elif target == 1:
        driver = rng.choice(("hospitalization", "treatment", "lab_changes", "daily_activities"))
        case[driver] = {"hospitalization": rng.choice(HOSPITALIZATION[1:3]), "treatment": TREATMENT[2],
                        "lab_changes": LAB_CHANGES[2], "daily_activities": DAILY_ACTIVITIES[1]}[driver]
    # Borderline cases: a share of moderate and severe cases involve an observation stay.
    if target in (1, 2) and rng.random() < 0.35:
        case["hospitalization"] = HOSPITALIZATION[2]
        if target == 2 and structured_grade(case) < 2:
            case["treatment"] = TREATMENT[2]
    case["narrative"] = _narrative(rng, case)
    return case


def _narrative(rng: random.Random, case: Mapping[str, Any]) -> str:
    parts = [f"{case['event_term'].capitalize()} reported {case['onset_days_after_dose']} days after a dose "
             f"of {SUSPECT_PRODUCT}."]
    if case["hospitalization"] == HOSPITALIZATION[2]:
        parts.append(rng.choice(tuple(OBSERVATION_CUES)))
    elif case["hospitalization"] != HOSPITALIZATION[0]:
        parts.append(f"Hospital course: {case['hospitalization']}.")
    if case["treatment"] != TREATMENT[0]:
        parts.append(f"Treated with {case['treatment']}.")
    parts.append(f"Outcome at last follow-up: {case['outcome']}.")
    return " ".join(parts)


def assessor_grade(rng: random.Random, case: Mapping[str, Any]) -> int:
    """The grade an assessor records by following the guide (with its one judgement call)."""
    grade = structured_grade(case)
    if case["hospitalization"] == HOSPITALIZATION[2] and grade < 2:
        cue = next((c for c in OBSERVATION_CUES if c in case["narrative"]), None)
        if cue is not None and rng.random() < OBSERVATION_CUES[cue]:
            grade = 2
    return grade


def generate() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    rows = []
    for i in range(1, N_CASES + 1):
        case_id = f"CAL-AE-{i:04d}"
        case = _case(rng, case_id)
        grade = assessor_grade(rng, case)
        if grade < 4 and rng.random() < CODING_NOISE_RATE:
            grade = min(3, max(0, grade + rng.choice((-1, 1))))
        rows.append({"id": case_id, "context": case_context(case), "label": GRADES[grade]})
    return rows


def main() -> int:
    rows = generate()
    with OUTPUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    labels = Counter(row["label"] for row in rows)
    serious = sum(labels[g] for g in GRADES[2:])
    print(f"wrote {len(rows)} synthetic adverse-event cases to {OUTPUT.name}")
    print("grades:  " + ", ".join(f"{g}={labels[g]}" for g in GRADES))
    print(f"serious (grade_3 to grade_5): {serious}, non-serious: {len(rows) - serious}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
