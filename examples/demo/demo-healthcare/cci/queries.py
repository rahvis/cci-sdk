"""The three calibrated queries this pipeline uses, and their profile names.

Demonstration-scale parameters: alpha=0.2 and Gate's risk target=0.15 are
looser than a production deployment would use (the docs' own examples use
alpha=0.10 with 240-1000+ calibration examples), chosen here so ~48
synthetic examples are enough to reach "serving" status and the whole
demo runs in a few minutes. The math CCI runs is identical either way;
only the sample size changes how tight the guarantee is.
"""

from __future__ import annotations

from cli_sdk import Claim, Gate, Set

URGENCY_LEVELS = {
    "self_care": "Rest and monitor at home; no clinical visit needed right now.",
    "primary_care": "Schedule a visit with a primary care provider in the next few days.",
    "urgent_care": "Visit an urgent care clinic today.",
    "emergency": "Call emergency services or go to the emergency room right now.",
}
LOW_ACUITY = ("self_care", "primary_care")

SET_PROFILE = "patient-triage-set-v1"
GATE_PROFILE = "patient-triage-gate-v1"
CLAIM_PROFILE = "patient-triage-claim-v3"

SET_ALPHA = 0.2
GATE_TARGET = 0.15
CLAIM_ALPHA = 0.3


def urgency_set_query() -> Set:
    return Set(
        instructions=(
            "Given this patient's portal message and structured intake facts, which urgency "
            "level is correct?"
        ),
        options=dict(URGENCY_LEVELS),
        calibration_profile=SET_PROFILE,
        alpha=SET_ALPHA,
        method="APS",
    )


def urgency_gate_query() -> Gate:
    return Gate(
        instructions=(
            "Given this patient's portal message, structured intake facts, and the proposed "
            "urgency level, is the proposed urgency level correct?"
        ),
        calibration_profile=GATE_PROFILE,
        guarantee="risk",
        target=GATE_TARGET,
    )


def claim_query() -> Claim:
    return Claim(
        instructions=(
            "Keep only the statements in this patient-facing reply that are supported by "
            "either the patient's intake facts or the assigned urgency level's guidance text."
        ),
        calibration_profile=CLAIM_PROFILE,
        alpha=CLAIM_ALPHA,
    )
