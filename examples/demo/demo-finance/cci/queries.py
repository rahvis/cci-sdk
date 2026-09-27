"""The calibrated queries for both finance scenarios.

Demonstration-scale parameters, same rationale as demo-healthcare: looser
than production so ~40-60 synthetic examples reach "serving" status and
the demo runs in a few minutes.
"""

from __future__ import annotations

from cli_sdk import Gate, Set

# --- Lending: Set with Mondrian groups + Gate(risk) ---

TIERS = {
    "prime": "Strong credit profile: low DTI, solid history, no defaults.",
    "near_prime": "Acceptable profile with one soft factor (moderate DTI or shorter history).",
    "subprime": "Weak profile: high DTI, thin history, or a prior default.",
    "decline": "Unemployed or repeat defaults: does not qualify.",
}

LENDING_SET_PROFILE = "finance-lending-tier-v2"
LENDING_GATE_PROFILE = "finance-lending-approve-v2"
LENDING_SET_ALPHA = 0.2
LENDING_GATE_TARGET = 0.15


def lending_tier_query() -> Set:
    return Set(
        instructions="Given this applicant's channel and financials, which credit tier is correct?",
        options=dict(TIERS),
        calibration_profile=LENDING_SET_PROFILE,
        alpha=LENDING_SET_ALPHA,
        method="APS",
        group_by="channel",
    )


def lending_approve_query() -> Gate:
    return Gate(
        instructions="Given this applicant and the proposed credit tier, is auto-approving at that tier correct?",
        calibration_profile=LENDING_GATE_PROFILE,
        guarantee="risk",
        target=LENDING_GATE_TARGET,
    )


# --- AML: Gate(fdr) ---

AML_GATE_PROFILE = "finance-aml-hold-v2"
AML_GATE_TARGET = 0.15
AML_GATE_DELTA = 0.15


def aml_hold_query() -> Gate:
    return Gate(
        instructions="Given this AML alert, is placing an account hold the correct action?",
        calibration_profile=AML_GATE_PROFILE,
        guarantee="fdr",
        target=AML_GATE_TARGET,
        delta=AML_GATE_DELTA,
    )
