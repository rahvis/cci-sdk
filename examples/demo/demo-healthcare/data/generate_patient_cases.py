"""Synthetic patient-portal triage dataset generator.

SYNTHETIC DATA FOR DEMONSTRATION ONLY. Not real patients, not medical
advice, not a medical device. Seeded and deterministic; no network access.

Produces two files in this directory:
- patient_cases.jsonl: one case per line (message, structured facts,
  ground-truth urgency, calibration/test split, is_trap flag).
- claim_calibration.jsonl: synthetic drafted patient replies with
  per-claim supported/not labels, for calibrating the Claim primitive.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SEED = 20260927
N_CALIBRATION = 48
N_TEST = 16
OUT_DIR = Path(__file__).resolve().parent

LEVELS = ("self_care", "primary_care", "urgent_care", "emergency")

# ---------------------------------------------------------------------------
# Case templates. Each template is a (facts_fn, message_fn) pair producing
# one case at a given urgency level. "casual" templates use minimizing,
# offhand language; "trap" cases pair casual language with facts that are
# actually red-flag (emergency-level) combinations.
# ---------------------------------------------------------------------------


def _self_care(rng: random.Random) -> dict[str, Any]:
    symptom = rng.choice(["a runny nose and mild sore throat", "a mild headache", "a small paper cut on my finger",
                          "occasional sneezing", "a slightly upset stomach after a big meal"])
    duration = rng.choice(["since this morning", "for about a day", "since last night"])
    facts = {"age": rng.randint(19, 55), "symptoms": [symptom], "duration": duration,
             "vitals": None, "red_flags": []}
    message = f"I've had {symptom} {duration}. Nothing else going on. Should I be worried?"
    return {"facts": facts, "message": message, "urgency": "self_care"}


def _primary_care(rng: random.Random) -> dict[str, Any]:
    symptom = rng.choice(["a low-grade fever that comes and goes", "a dry cough", "a rash on my arm that itches",
                          "mild joint pain in my knee", "trouble sleeping and feeling tired"])
    duration = rng.choice(["for about 5 days", "for over a week", "for 4 days now"])
    facts = {"age": rng.randint(20, 65), "symptoms": [symptom], "duration": duration,
             "vitals": {"temp_f": round(rng.uniform(99.0, 100.3), 1)}, "red_flags": []}
    message = f"I've had {symptom} {duration}. It's not getting worse but it's not going away either."
    return {"facts": facts, "message": message, "urgency": "primary_care"}


def _urgent_care(rng: random.Random) -> dict[str, Any]:
    symptom = rng.choice(["I twisted my ankle and it's swollen and painful to walk on",
                          "I've been vomiting since last night and can't keep water down",
                          "I cut my hand on a knife and it won't stop bleeding with direct pressure",
                          "I have a fever of 103 and chills"])
    facts = {"age": rng.randint(18, 60), "symptoms": [symptom], "duration": "since yesterday",
             "vitals": {"temp_f": 103.0} if "fever" in symptom else None,
             "red_flags": ["moderate_acute_injury_or_illness"]}
    message = f"{symptom.capitalize()}. What should I do?"
    return {"facts": facts, "message": message, "urgency": "urgent_care"}


_EMERGENCY_CASES = [
    {"symptoms": ["crushing chest pain radiating to my left arm", "shortness of breath", "sweating"],
     "red_flags": ["chest_pain", "dyspnea", "diaphoresis"]},
    {"symptoms": ["the worst headache of my life, sudden onset", "neck stiffness"],
     "red_flags": ["thunderclap_headache"]},
    {"symptoms": ["my face is drooping on one side", "slurred speech", "weakness in my right arm"],
     "red_flags": ["stroke_signs"]},
    {"symptoms": ["severe abdominal pain", "fever", "vomiting", "pain is worse when I move"],
     "red_flags": ["acute_abdomen", "fever"]},
    {"symptoms": ["my throat is closing up after eating peanuts", "difficulty breathing", "hives all over"],
     "red_flags": ["anaphylaxis"]},
    {"symptoms": ["coughing up blood", "chest pain", "shortness of breath"],
     "red_flags": ["hemoptysis", "chest_pain", "dyspnea"]},
]


def _emergency(rng: random.Random, casual: bool) -> dict[str, Any]:
    case = rng.choice(_EMERGENCY_CASES)
    facts = {"age": rng.randint(25, 75), "symptoms": list(case["symptoms"]), "duration": "started about an hour ago",
             "vitals": None, "red_flags": list(case["red_flags"])}
    if not casual:
        message = "I'm having " + ", ".join(case["symptoms"]) + ". This started about an hour ago."
        return {"facts": facts, "message": message, "urgency": "emergency", "is_trap": False}
    # Trap: same red-flag facts, minimizing/casual phrasing that could read as low-acuity.
    casual_phrasing = {
        "chest_pain": "a bit of tightness in my chest",
        "dyspnea": "kind of out of breath, probably just out of shape",
        "diaphoresis": "sweating more than usual",
        "thunderclap_headache": "a really bad headache that hit out of nowhere",
        "stroke_signs": "my face feels a little off on one side and my words came out weird for a bit",
        "acute_abdomen": "my stomach really hurts",
        "fever": "feeling warm",
        "anaphylaxis": "my throat feels tight after lunch, might just be allergies",
        "hemoptysis": "noticed a little blood when I coughed",
    }
    phrases = [casual_phrasing.get(rf, s) for rf, s in zip(case["red_flags"], case["symptoms"])]
    message = ("Probably nothing but just to mention it: " + ", ".join(phrases) +
              ". Started about an hour ago, figured I'd check before it's a whole thing.")
    return {"facts": facts, "message": message, "urgency": "emergency", "is_trap": True}


def make_case(rng: random.Random, level: str, casual_trap: bool = False) -> dict[str, Any]:
    if level == "self_care":
        out = _self_care(rng)
    elif level == "primary_care":
        out = _primary_care(rng)
    elif level == "urgent_care":
        out = _urgent_care(rng)
    else:
        out = _emergency(rng, casual=casual_trap)
    out.setdefault("is_trap", False)
    return out


def generate_cases(rng: random.Random, n: int, trap_fraction: float) -> list[dict[str, Any]]:
    n_trap = max(1, round(n * trap_fraction)) if trap_fraction > 0 else 0
    per_level = max(1, n // len(LEVELS))
    plan: list[tuple[str, bool]] = []
    for level in LEVELS:
        count = per_level
        for i in range(count):
            trap = level == "emergency" and i < n_trap
            plan.append((level, trap))
    while len(plan) < n:
        plan.append((rng.choice(LEVELS), False))
    rng.shuffle(plan)
    return [make_case(rng, level, casual_trap=trap) for level, trap in plan[:n]]


# ---------------------------------------------------------------------------
# Claim calibration data: synthetic drafted replies with per-claim labels.
# ---------------------------------------------------------------------------

def make_claim_example(rng: random.Random, case: dict[str, Any]) -> dict[str, Any]:
    facts = case["facts"]
    urgency_advice = {
        "self_care": "Rest, stay hydrated, and monitor your symptoms at home.",
        "primary_care": "Please schedule a visit with your primary care provider in the next few days.",
        "urgent_care": "Please visit an urgent care clinic today.",
        "emergency": "Please call 911 or go to the nearest emergency room right away.",
    }[case["urgency"]]
    true_claims = [
        f"You reported {facts['symptoms'][0]}.",
        urgency_advice,
    ]
    labels = [{"text": c, "supported": True} for c in true_claims]
    # About 40% of drafts include one fabricated claim not grounded in the facts.
    if rng.random() < 0.4:
        fabricated = rng.choice([
            "Your prior lab results confirm this is not serious.",
            "This is consistent with a diagnosis you were given last year.",
            "You mentioned this has happened many times before with no issues.",
            "Your vitals are completely normal.",
        ])
        labels.append({"text": fabricated, "supported": False})
    draft = " ".join(c["text"] for c in labels)
    return {
        "context": {"facts": facts, "guidance": urgency_advice, "answer": draft},
        "label": labels,
    }


def main() -> None:
    rng = random.Random(SEED)
    calibration = generate_cases(rng, N_CALIBRATION, trap_fraction=0.15)
    test = generate_cases(rng, N_TEST, trap_fraction=0.35)

    cases_path = OUT_DIR / "patient_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as fh:
        for i, case in enumerate(calibration):
            case = {"case_id": f"C-{i + 1:04d}", "split": "calibration", **case}
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
        for i, case in enumerate(test):
            case = {"case_id": f"T-{i + 1:04d}", "split": "test", **case}
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")

    claim_path = OUT_DIR / "claim_calibration.jsonl"
    with claim_path.open("w", encoding="utf-8") as fh:
        for case in calibration:
            fh.write(json.dumps(make_claim_example(rng, case), ensure_ascii=False) + "\n")

    print(f"wrote {len(calibration)} calibration + {len(test)} test cases to {cases_path}")
    print(f"wrote {len(calibration)} claim-calibration examples to {claim_path}")


if __name__ == "__main__":
    main()
