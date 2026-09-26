"""Generate ``discharge_claims.jsonl``: synthetic discharge charts, draft summaries and claim labels.

All data is synthetic. No real patient, clinician or hospital is represented,
and nothing here is medical advice or a clinical protocol.

Each line is one draft patient-friendly discharge summary::

    {"id": "D-1001",
     "context": {"chart": {...}, "answer": "<draft summary>"},
     "label": [{"text": "<claim>", "supported": true}, ...]}

``context`` is exactly what the Claim check evaluates at run time, built by
``summary_context`` below; the LangGraph example imports it, so live drafts
and calibration drafts are rendered by the same function. The draft is one
sentence per claim, which is how the claims are split at run time.

Labelling protocol (written, applied by ``_claims`` below)
    A claim is *supported* when the chart states it, or when it restates a
    chart entry in plain language without changing any drug, dose,
    frequency, duration, status (new, continued, changed, held, stopped),
    appointment, date, test or symptom. A claim that adds a fact the chart
    does not contain, or changes one, is *unsupported*. Typical unsupported
    claims: a wrong dose, an invented dose change, "stop" for a continued
    drug, "continue" for a held drug, an invented or wrong follow-up date,
    a wrong follow-up interval or antibiotic duration, an invented pending
    test, an invented diagnosis.

About half of the drafts contain one or two unsupported claims. Some
supported claims are plain-language paraphrases that share few words with
the chart, and about 1% of claim labels are flipped to model annotator
disagreement, so the calibrated support threshold is realistic.

Run ``python generate_discharge_claims.py`` (stdlib only, seeded) to rewrite
``discharge_claims.jsonl`` next to this file; the output is byte-identical
on every run.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

SEED = 20260925
N_EXAMPLES = 240
OUTPUT = Path(__file__).resolve().parent / "discharge_claims.jsonl"

CHART_FIELDS = (
    "encounter_id", "discharge_date", "principal_diagnosis", "secondary_diagnoses", "medications",
    "follow_up", "pending_results", "instructions", "return_precautions",
)


def summary_context(chart: dict[str, Any], draft: str) -> dict[str, Any]:
    """The exact context the Claim check evaluates: the chart and the draft summary."""
    return {"chart": {field: chart.get(field) for field in CHART_FIELDS}, "answer": draft}


# ---------------------------------------------------------------------------
# synthetic clinical content (illustrative only)
# ---------------------------------------------------------------------------

CONDITIONS: list[dict[str, Any]] = [
    {
        "diagnosis": "community-acquired pneumonia",
        "meds": [{"name": "amoxicillin-clavulanate", "dose": 875, "unit": "mg", "frequency": "twice daily",
                  "status": "new", "duration": "5 more days"}],
        "clinic": "pulmonology clinic",
        "pending": ["blood cultures", "sputum culture"],
        "instructions": ["Finish every dose of the antibiotic.", "Rest and drink plenty of fluids."],
        "precautions": ["a temperature above 38.5 C", "worsening shortness of breath", "chest pain"],
        "precaution_paraphrase": "Get help right away if your breathing gets harder.",
    },
    {
        "diagnosis": "acute decompensated heart failure",
        "meds": [{"name": "furosemide", "dose": 40, "unit": "mg", "frequency": "twice daily", "status": "changed",
                  "previous": "40 mg once daily"}],
        "clinic": "cardiology clinic",
        "pending": ["echocardiogram"],
        "instructions": ["Weigh yourself every morning and write it down.", "Limit salt to 2 grams a day."],
        "precautions": ["weight gain of more than 1 kg in a day", "swelling in the legs",
                        "worsening shortness of breath"],
        "precaution_paraphrase": "Call the clinic if the scale goes up quickly from one morning to the next.",
    },
    {
        "diagnosis": "cellulitis of the left lower leg",
        "meds": [{"name": "cephalexin", "dose": 500, "unit": "mg", "frequency": "four times daily", "status": "new",
                  "duration": "7 more days"}],
        "clinic": "wound care clinic",
        "pending": ["wound culture"],
        "instructions": ["Keep the leg raised when sitting.", "Mark the edge of the redness each day."],
        "precautions": ["redness spreading past the marked line", "a fever above 38.5 C", "new blisters"],
        "precaution_paraphrase": "Come back if the red area grows beyond the pen line.",
    },
    {
        "diagnosis": "COPD exacerbation",
        "meds": [{"name": "prednisone", "dose": 40, "unit": "mg", "frequency": "once daily", "status": "new",
                  "duration": "4 more days"},
                 {"name": "tiotropium inhaler", "dose": 18, "unit": "mcg", "frequency": "once daily",
                  "status": "continued"}],
        "clinic": "pulmonology clinic",
        "pending": ["sputum culture"],
        "instructions": ["Use the rescue inhaler before walking longer distances.", "Do not smoke."],
        "precautions": ["rescue inhaler use more often than every 4 hours", "lips turning blue", "new confusion"],
        "precaution_paraphrase": "Seek care if your lips look bluish.",
    },
    {
        "diagnosis": "atrial fibrillation with rapid heart rate",
        "meds": [{"name": "apixaban", "dose": 5, "unit": "mg", "frequency": "twice daily", "status": "new",
                  "duration": "until your cardiologist reviews it"},
                 {"name": "diltiazem extended-release", "dose": 120, "unit": "mg", "frequency": "once daily",
                  "status": "new", "duration": "until your cardiologist reviews it"}],
        "clinic": "cardiology clinic",
        "pending": ["thyroid function tests"],
        "instructions": ["Do not skip apixaban doses.", "Avoid heavy alcohol use."],
        "precautions": ["fainting", "black or bloody stools", "a racing heartbeat that does not settle"],
        "precaution_paraphrase": "Get checked if you pass out or nearly pass out.",
    },
    {
        "diagnosis": "acute pyelonephritis",
        "meds": [{"name": "ciprofloxacin", "dose": 500, "unit": "mg", "frequency": "twice daily", "status": "new",
                  "duration": "6 more days"}],
        "clinic": "urology clinic",
        "pending": ["urine culture"],
        "instructions": ["Drink at least 2 liters of water a day.", "Take ciprofloxacin 2 hours before antacids."],
        "precautions": ["a fever above 38.5 C", "vomiting that stops you keeping fluids down",
                        "back pain getting worse"],
        "precaution_paraphrase": "Return if you cannot keep water down.",
    },
    {
        "diagnosis": "hyperglycemia in type 2 diabetes",
        "meds": [{"name": "insulin glargine", "dose": 18, "unit": "units", "frequency": "at bedtime", "status": "new",
                  "duration": "until your diabetes team adjusts it"}],
        "clinic": "endocrinology clinic",
        "pending": ["hemoglobin A1c"],
        "instructions": ["Check your blood sugar before meals and at bedtime.", "Carry glucose tablets."],
        "precautions": ["blood sugar below 4 mmol/L that does not improve", "blood sugar above 20 mmol/L", "vomiting"],
        "precaution_paraphrase": "Get help if your sugar stays low after you treat it.",
    },
    {
        "diagnosis": "acute kidney injury from dehydration",
        "meds": [{"name": "hydrochlorothiazide", "dose": 25, "unit": "mg", "frequency": "once daily",
                  "status": "stopped"}],
        "clinic": "nephrology clinic",
        "pending": ["kidney ultrasound"],
        "instructions": ["Drink fluids steadily through the day.", "Avoid ibuprofen and naproxen."],
        "precautions": ["very little urine", "dizziness when standing", "swelling in the legs"],
        "precaution_paraphrase": "Call if you are hardly peeing.",
    },
]

BACKGROUND_MEDS: list[dict[str, Any]] = [
    {"name": "metformin", "dose": 1000, "unit": "mg", "frequency": "twice daily"},
    {"name": "atorvastatin", "dose": 40, "unit": "mg", "frequency": "at bedtime"},
    {"name": "levothyroxine", "dose": 75, "unit": "mcg", "frequency": "once daily"},
    {"name": "amlodipine", "dose": 5, "unit": "mg", "frequency": "once daily"},
    {"name": "omeprazole", "dose": 20, "unit": "mg", "frequency": "once daily"},
    {"name": "sertraline", "dose": 50, "unit": "mg", "frequency": "once daily"},
    {"name": "lisinopril", "dose": 10, "unit": "mg", "frequency": "once daily"},
    {"name": "metoprolol succinate", "dose": 50, "unit": "mg", "frequency": "once daily"},
]
SECONDARY = ["type 2 diabetes", "hypertension", "hypothyroidism", "chronic kidney disease stage 3", "depression",
             "hyperlipidemia"]
OTHER_DIAGNOSES = ["a urinary tract infection", "pulmonary embolism", "a stroke", "pancreatitis", "anemia",
                   "sepsis"]
OTHER_TESTS = ["MRI", "biopsy", "sleep study", "colonoscopy", "CT scan"]
OTHER_CLINICS = ["neurology clinic", "rheumatology clinic", "dermatology clinic", "orthopedic clinic"]


def _dose(med: dict[str, Any], dose: Optional[float] = None) -> str:
    value = med["dose"] if dose is None else dose
    return f"{value:g} {med['unit']}"


# ---------------------------------------------------------------------------
# claim templates: (text, supported)
# ---------------------------------------------------------------------------


def _med_claim(med: dict[str, Any], rng: random.Random) -> tuple[str, bool]:
    name, dose, freq = med["name"], _dose(med), med["frequency"]
    status = med["status"]
    if status == "new":
        duration = med["duration"] if med["duration"].startswith("until") else f"for {med['duration']}"
        return f"Start {name} {dose} {freq} {duration}.", True
    if status == "changed":
        return f"Your {name} dose is now {dose} {freq}, up from {med['previous']}.", True
    if status == "held":
        if rng.random() < 0.3:
            return f"Pause {name} for now; your doctor will tell you when to restart it.", True
        return f"Do not take {name} until your follow-up visit.", True
    if status == "stopped":
        return f"Stop taking {name}.", True
    return f"Continue {name} {dose} {freq} as before.", True


def _unsupported_med_claim(med: dict[str, Any], rng: random.Random) -> tuple[str, bool]:
    name, freq, status = med["name"], med["frequency"], med["status"]
    kind = rng.random()
    if status == "continued":
        if kind < 0.35:
            return f"Your {name} dose was increased to {_dose(med, med['dose'] * 2)} {freq}.", False
        if kind < 0.60:
            return f"Stop taking {name}.", False
        wrong = med["dose"] * rng.choice([0.5, 2, 4])
        return f"Continue {name} {_dose(med, wrong)} {freq} as before.", False
    if status in ("held", "stopped"):
        if kind < 0.6:
            return f"Continue {name} {_dose(med)} {freq} as before.", False
        return f"Your {name} dose was lowered to {_dose(med, med['dose'] / 2)} {freq}.", False
    if status == "new":
        if kind < 0.5:
            return f"Start {name} {_dose(med)} {freq} for 14 more days.", False
        return f"Start {name} {_dose(med, med['dose'] * 2)} {freq} for {med['duration']}.".replace(
            "for until", "until"), False
    return f"Your {name} dose is now {_dose(med, med['dose'] * 2)} {freq}, up from {med['previous']}.", False


def _chart(rng: random.Random, index: int) -> dict[str, Any]:
    condition = rng.choice(CONDITIONS)
    discharge = date(2026, 1, 5) + timedelta(days=rng.randrange(0, 230))
    meds = [dict(m) for m in condition["meds"]]
    for med in rng.sample(BACKGROUND_MEDS, rng.choice([1, 2, 2, 3])):
        med = dict(med)
        med["status"] = rng.choices(["continued", "held", "stopped"], weights=[75, 15, 10])[0]
        if med["status"] == "held":
            med["reason"] = "held until the follow-up visit"
        meds.append(med)
    follow_up = [{"with": "primary care", "when": f"within {rng.choice([3, 5, 7, 10])} days"}]
    if rng.random() < 0.8:
        follow_up.append({"with": condition["clinic"],
                          "when": (discharge + timedelta(days=rng.randrange(7, 29))).isoformat()})
    return {
        "encounter_id": f"D-{1001 + index}",
        "discharge_date": discharge.isoformat(),
        "principal_diagnosis": condition["diagnosis"],
        "secondary_diagnoses": sorted(rng.sample(SECONDARY, rng.choice([0, 1, 2]))),
        "medications": meds,
        "follow_up": follow_up,
        "pending_results": rng.sample(condition["pending"], 1) if rng.random() < 0.7 else [],
        "instructions": list(condition["instructions"]),
        "return_precautions": list(condition["precautions"]),
        "_condition": condition,
    }


def _claims(chart: dict[str, Any], rng: random.Random) -> list[tuple[str, bool]]:
    """Supported claims from the chart, then zero to two unsupported ones, per the protocol."""
    condition = chart["_condition"]
    meds = chart["medications"]
    claims: list[tuple[str, bool]] = [(f"You were in the hospital for {chart['principal_diagnosis']}.", True)]
    for med in meds[: rng.choice([1, 2, 3])]:
        claims.append(_med_claim(med, rng))
    visit = rng.choice(chart["follow_up"])
    if "-" in visit["when"]:
        claims.append((f"You have an appointment at the {visit['with']} on {visit['when']}.", True))
    elif rng.random() < 0.25:
        days = visit["when"].split()[1]
        claims.append((f"Book a visit with your family doctor in the next {days} days.", True))
    else:
        claims.append((f"See your primary care doctor {visit['when']}.", True))
    optional = []
    if chart["pending_results"]:
        optional.append((f"The results of your {chart['pending_results'][0]} are still pending; "
                         "the team will call you with them.", True))
    optional.append((rng.choice(chart["instructions"]), True))
    if rng.random() < 0.3:
        optional.append((condition["precaution_paraphrase"], True))
    else:
        optional.append((f"Come back to the emergency department if you notice "
                         f"{rng.choice(chart['return_precautions'])}.", True))
    claims.extend(rng.sample(optional, min(len(optional), rng.choice([1, 2, 2, 3]))))

    roll = rng.random()
    n_unsupported = 2 if roll < 0.12 else (1 if roll < 0.52 else 0)
    for _ in range(n_unsupported):
        kind = rng.random()
        if kind < 0.40:
            bad = _unsupported_med_claim(rng.choice(meds), rng)
        elif kind < 0.60:
            clinic = rng.choice([condition["clinic"], rng.choice(OTHER_CLINICS)])
            booked = {visit["when"] for visit in chart["follow_up"]}
            when = date.fromisoformat(chart["discharge_date"]) + timedelta(days=rng.randrange(3, 40))
            while when.isoformat() in booked:
                when += timedelta(days=1)
            bad = (f"You have an appointment at the {clinic} on {when.isoformat()}.", False)
        elif kind < 0.72:
            bad = (f"See your primary care doctor within {rng.choice([14, 21, 30])} days.", False)
        elif kind < 0.86:
            bad = (f"The results of your {rng.choice(OTHER_TESTS)} are still pending; "
                   "the team will call you with them.", False)
        else:
            bad = (f"You were also treated for {rng.choice(OTHER_DIAGNOSES)}.", False)
        if bad[0] in {text for text, _ in claims}:
            continue
        claims.insert(rng.randrange(1, len(claims) + 1), bad)
    return claims


def build_examples(seed: int = SEED, n: int = N_EXAMPLES) -> list[dict[str, Any]]:
    """The labelled calibration examples, deterministically."""
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        chart = _chart(rng, i)
        claims = _claims(chart, rng)
        chart.pop("_condition")
        labels = []
        for text, supported in claims:
            if rng.random() < 0.01:
                supported = not supported          # annotator disagreement
            labels.append({"text": text, "supported": supported})
        draft = " ".join(text for text, _ in claims)
        examples.append({"id": chart["encounter_id"], "context": summary_context(chart, draft), "label": labels})
    return examples


def write(path: Optional[Path] = None) -> Path:
    path = path or OUTPUT
    with path.open("w", encoding="utf-8") as fh:
        for example in build_examples():
            fh.write(json.dumps(example, sort_keys=True) + "\n")
    return path


if __name__ == "__main__":
    out = write()
    rows = build_examples()
    claims = [c for r in rows for c in r["label"]]
    unsupported = sum(1 for c in claims if not c["supported"])
    with_false = sum(1 for r in rows if any(not c["supported"] for c in r["label"]))
    print(f"wrote {len(rows)} drafts ({len(claims)} claims) to {out.name}: {unsupported} unsupported claims; "
          f"{with_false} drafts ({with_false / len(rows):.0%}) contain at least one")
