#!/usr/bin/env python3
"""Generate ``triage_messages.jsonl``: synthetic patient-portal messages labelled with a triage level.

Every record in the output is synthetic. The patients, messages and the
triage protocol are invented for the LangChain triage example
(``examples/agents/langchain/healthcare_patient_triage.py``). The protocol is
a simplified teaching example: it is not clinical guidance, not medical
advice, and must not be used to triage real patients.

What one line holds
-------------------
::

    {"id": "triage-0001",
     "context": {"protocol": TRIAGE_PROTOCOL, "message": {...intake form and free text...}},
     "label": "routine_appointment"}

``context`` is exactly what the triage guard evaluates: this generator and
the example both build it with ``triage_context()`` below, so calibration
scores and live scores come from the same scoring function. ``label`` is
one of the keys of ``TRIAGE_LEVELS``.

How labels are made
-------------------
``protocol_level()`` applies the written protocol rule by rule. Two kinds of
cases keep the data from being unrealistically clean:

- Vague messages (about 4%, complaint ``unclear``): rule 5 hands these to a
  nurse, so the label is the nurse's recorded call (a seeded choice between
  ``self_care`` and ``routine_appointment``), which the message cannot settle.
- Reviewer inconsistency (1.5%): the label moves one level up or down at
  random, as in any human-labelled export.

The generator is deterministic (seeded, standard library only)::

    python examples/agents/data/generate_triage_messages.py          # rewrite triage_messages.jsonl
    python examples/agents/data/generate_triage_messages.py --check  # verify the committed file
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

SEED = 20260925
N_EXAMPLES = 320
OUTPUT = Path(__file__).resolve().parent / "triage_messages.jsonl"

TRIAGE_LEVELS = {
    "self_care": "home-care advice with clear instructions on when to seek care",
    "routine_appointment": "book a routine appointment within 1 to 3 days",
    "urgent_care": "same-day urgent care visit",
    "emergency": "call emergency services or go to the emergency department now",
}
LEVEL_ORDER = tuple(TRIAGE_LEVELS)

RED_FLAGS = (
    "face_drooping", "arm_or_leg_weakness", "slurred_speech", "chest_pain_with_shortness_of_breath",
    "severe_difficulty_breathing", "fainting", "vomiting_blood", "stiff_neck_with_fever",
    "sudden_worst_headache", "new_confusion",
)
URGENT_SIGNS = (
    "cannot_bear_weight", "fever_with_flank_pain", "blood_in_urine", "pain_worsening_quickly",
    "rash_with_fever", "signs_of_dehydration", "wheezing",
)
SELF_CARE_LIMIT_DAYS = {  # rule 3: longest duration that is still self-care, by complaint
    "cold_symptoms": 10, "sore_throat": 7, "cough": 21, "rash": 7, "abdominal_pain": 3,
    "headache": 7, "back_pain": 14, "ankle_injury": 5, "chest_discomfort": 0, "urinary_symptoms": 0,
    "unclear": 7,
}

TRIAGE_PROTOCOL = """\
Patient-portal message triage protocol (synthetic teaching example for adult patients, version 2026-09; \
not clinical guidance).
Assign the first level whose rule matches, checking the rules in order.
1. emergency: any red flag is reported (face drooping, arm or leg weakness, slurred speech, chest pain with \
shortness of breath, severe difficulty breathing, fainting, vomiting blood, stiff neck with fever, sudden \
worst headache, new confusion).
2. urgent_care (same-day visit): no red flag, and any of: an urgent sign (cannot bear weight, fever with flank \
pain, blood in urine, pain worsening quickly, rash with fever, signs of dehydration, wheezing); temperature \
39.0 C or higher; self-rated severity severe; age band 65+ with temperature 38.0 C or higher; chest discomfort \
in the 40-64 or 65+ age band.
3. routine_appointment (within 1 to 3 days): none of the above, and any of: self-rated severity moderate; \
symptoms lasting longer than the self-care limit for the complaint (cold symptoms 10 days, sore throat 7, cough \
21, rash 7, abdominal pain 3, headache 7, back pain 14, ankle injury 5); any urinary symptoms; any chest \
discomfort; temperature 38.0 C or higher for 3 days or more.
4. self_care: everything else (mild symptoms within the self-care limit).
5. A message too vague to apply rules 1 to 4 (complaint unclear) is triaged by a nurse."""


# ---------------------------------------------------------------------------
# The record format shared with the example (one definition, used by both)
# ---------------------------------------------------------------------------


def make_message(
    message_id: str,
    *,
    age_band: str,
    complaint: str,
    duration_days: int,
    self_rated_severity: str,
    text: str,
    temperature_c: Optional[float] = None,
    red_flags: Sequence[str] = (),
    urgent_signs: Sequence[str] = (),
) -> dict[str, Any]:
    """One portal message: the intake-form fields plus the patient's own words (same keys every time)."""
    return {
        "message_id": message_id,
        "age_band": age_band,
        "complaint": complaint,
        "duration_days": int(duration_days),
        "self_rated_severity": self_rated_severity,
        "temperature_c": None if temperature_c is None else round(float(temperature_c), 1),
        "red_flags": sorted(red_flags),
        "urgent_signs": sorted(urgent_signs),
        "text": text,
    }


def triage_context(message: dict[str, Any]) -> dict[str, Any]:
    """The evaluation context for one message: calibration and inference both use this."""
    return {"protocol": TRIAGE_PROTOCOL, "message": dict(message)}


# ---------------------------------------------------------------------------
# The written protocol, rule by rule
# ---------------------------------------------------------------------------


def protocol_level(message: dict[str, Any]) -> Optional[str]:
    """The level ``TRIAGE_PROTOCOL`` assigns, or None when rule 5 hands the message to a nurse."""
    temp = message["temperature_c"]
    older = message["age_band"] in ("40-64", "65+")
    if message["red_flags"]:
        return "emergency"
    if message["complaint"] == "unclear":
        return None
    if (message["urgent_signs"] or (temp is not None and temp >= 39.0)
            or message["self_rated_severity"] == "severe"
            or (message["age_band"] == "65+" and temp is not None and temp >= 38.0)
            or (message["complaint"] == "chest_discomfort" and older)):
        return "urgent_care"
    if (message["self_rated_severity"] == "moderate"
            or message["duration_days"] > SELF_CARE_LIMIT_DAYS[message["complaint"]]
            or message["complaint"] in ("urinary_symptoms", "chest_discomfort")
            or (temp is not None and temp >= 38.0 and message["duration_days"] >= 3)):
        return "routine_appointment"
    return "self_care"


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------

COMPLAINTS = (  # (complaint, weight, how the patient describes it)
    ("cold_symptoms", 0.14, "have had a runny nose, sneezing and a mild cough"),
    ("sore_throat", 0.10, "have had a sore throat"),
    ("cough", 0.10, "have had a cough"),
    ("rash", 0.08, "have had an itchy rash on my arms"),
    ("urinary_symptoms", 0.07, "have had burning when I pee and need to go more often"),
    ("abdominal_pain", 0.09, "have had stomach pain"),
    ("headache", 0.09, "have had headaches"),
    ("chest_discomfort", 0.08, "have had a tight feeling in my chest"),
    ("back_pain", 0.09, "have had lower back pain"),
    ("ankle_injury", 0.08, "have had a sore ankle since I twisted it"),
    ("unclear", 0.04, "have been feeling off in a way that is hard to describe"),
)
RELATED_RED_FLAGS = {
    "chest_discomfort": ("chest_pain_with_shortness_of_breath", "fainting"),
    "headache": ("sudden_worst_headache", "slurred_speech", "face_drooping", "stiff_neck_with_fever", "new_confusion"),
    "cough": ("severe_difficulty_breathing",),
    "abdominal_pain": ("vomiting_blood", "fainting"),
    "cold_symptoms": ("severe_difficulty_breathing", "stiff_neck_with_fever"),
    "unclear": ("new_confusion", "arm_or_leg_weakness", "face_drooping"),
}
RELATED_URGENT_SIGNS = {
    "ankle_injury": ("cannot_bear_weight",),
    "urinary_symptoms": ("blood_in_urine", "fever_with_flank_pain"),
    "rash": ("rash_with_fever",),
    "cough": ("wheezing",),
    "abdominal_pain": ("pain_worsening_quickly", "signs_of_dehydration"),
    "back_pain": ("pain_worsening_quickly",),
    "sore_throat": ("signs_of_dehydration",),
}
PHRASES = {
    "face_drooping": "one side of my face is drooping",
    "arm_or_leg_weakness": "my left arm feels weak",
    "slurred_speech": "my speech is slurred",
    "chest_pain_with_shortness_of_breath": "the chest pain comes with shortness of breath",
    "severe_difficulty_breathing": "I am struggling to breathe",
    "fainting": "I fainted this morning",
    "vomiting_blood": "I vomited blood",
    "stiff_neck_with_fever": "my neck is stiff and I have a fever",
    "sudden_worst_headache": "it came on suddenly and is the worst headache of my life",
    "new_confusion": "my family says I seem confused",
    "cannot_bear_weight": "I cannot put any weight on it",
    "fever_with_flank_pain": "I have a fever and pain in my side",
    "blood_in_urine": "there is blood in my urine",
    "pain_worsening_quickly": "the pain is getting worse quickly",
    "rash_with_fever": "I also have a fever",
    "signs_of_dehydration": "I can barely keep fluids down and feel dizzy",
    "wheezing": "I am wheezing",
}
AGE_WORDS = {"18-39": "I am 29", "40-64": "I am 52", "65+": "I am 71"}


def message_text(complaint_words: str, age_band: str, duration: int, severity: str,
                 temp: Optional[float], findings: Sequence[str]) -> str:
    """The patient's own words, rendered from the intake fields (what a real model would read)."""
    days = "since yesterday" if duration <= 1 else f"for {duration} days"
    parts = [f"{AGE_WORDS[age_band]} and I {complaint_words} {days}.", f"It feels {severity}."]
    parts.append("I have not taken my temperature." if temp is None else f"My temperature is {temp:.1f} C.")
    parts.extend(PHRASES[f][0].upper() + PHRASES[f][1:] + "." for f in findings)
    return " ".join(parts)


def portal_message(message_id: str, *, age_band: str, complaint: str, duration_days: int,
                   self_rated_severity: str, temperature_c: Optional[float] = None,
                   red_flags: Sequence[str] = (), urgent_signs: Sequence[str] = ()) -> dict[str, Any]:
    """A complete portal message: intake fields plus the free text rendered from them."""
    words = next(w for name, _, w in COMPLAINTS if name == complaint)
    text = message_text(words, age_band, duration_days, self_rated_severity, temperature_c,
                        list(red_flags) + list(urgent_signs))
    return make_message(message_id, age_band=age_band, complaint=complaint, duration_days=duration_days,
                        self_rated_severity=self_rated_severity, text=text, temperature_c=temperature_c,
                        red_flags=red_flags, urgent_signs=urgent_signs)


def _duration(rng: random.Random, complaint: str) -> int:
    limit = SELF_CARE_LIMIT_DAYS[complaint] or 5
    # Most messages sit near the self-care limit, where triage decisions change.
    return max(1, min(45, int(round(rng.triangular(1, limit * 1.8, limit * 0.7)))))


def generate(seed: int = SEED, n: int = N_EXAMPLES) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    names, weights, _ = zip(*COMPLAINTS)
    records = []
    for i in range(1, n + 1):
        k = rng.choices(range(len(names)), weights=weights)[0]
        complaint = names[k]
        age_band = rng.choices(("18-39", "40-64", "65+"), weights=(0.4, 0.4, 0.2))[0]
        severity = rng.choices(("mild", "moderate", "severe"), weights=(0.64, 0.27, 0.09))[0]
        duration = _duration(rng, complaint)
        temp = (None if rng.random() < 0.6 else
                round(rng.uniform(36.5, 37.9) if rng.random() < 0.6 else rng.uniform(38.0, 40.2), 1))
        red = ([rng.choice(RELATED_RED_FLAGS[complaint])]
               if complaint in RELATED_RED_FLAGS and rng.random() < 0.14 else [])
        urgent = ([rng.choice(RELATED_URGENT_SIGNS[complaint])]
                  if complaint in RELATED_URGENT_SIGNS and rng.random() < 0.15 else [])
        message = portal_message(f"PM-{30000 + i * 7}", age_band=age_band, complaint=complaint,
                                 duration_days=duration, self_rated_severity=severity, temperature_c=temp,
                                 red_flags=red, urgent_signs=urgent)
        level = protocol_level(message)
        if level is None:
            level = rng.choice(("self_care", "routine_appointment"))  # the nurse's recorded call
        if rng.random() < 0.015:  # reviewer inconsistency: one level up or down
            j = LEVEL_ORDER.index(level) + rng.choice((-1, 1))
            level = LEVEL_ORDER[min(max(j, 0), len(LEVEL_ORDER) - 1)]
        records.append({"id": f"triage-{i:04d}", "context": triage_context(message), "label": level})
    return records


def render(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in records)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic triage calibration set.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="exit 1 if the file differs from a fresh generation")
    args = parser.parse_args(argv)
    records = generate()
    text = render(records)
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != text:
            print(f"{args.output} is out of date; rerun without --check", file=sys.stderr)
            return 1
        print(f"{args.output} is up to date ({len(records)} records)")
        return 0
    args.output.write_text(text, encoding="utf-8")
    counts = {level: sum(r["label"] == level for r in records) for level in TRIAGE_LEVELS}
    print(f"wrote {len(records)} records to {args.output}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
