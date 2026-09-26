"""Console output shared by the agent examples."""

from __future__ import annotations

import textwrap
from typing import Iterable

from cli_sdk.integrations import GuardDecision

DISCLAIMERS = {
    "healthcare": (
        "Synthetic data for demonstration only. Not medical advice and not a medical device. A "
        "calibrated guarantee bounds error rates on data like the calibration set; it does not make any "
        "single recommendation safe. Keep a qualified clinician in the loop and follow your "
        "institution's clinical governance."
    ),
    "finance": (
        "Synthetic data for demonstration only. Not financial, credit or legal advice. Calibrated "
        "guarantees support, but do not replace, your model-risk-management, fair-lending and "
        "compliance review."
    ),
    "insurance": (
        "Synthetic data for demonstration only. Not claims-handling or legal advice. Keep adjusters "
        "in the loop for every decision the guard escalates."
    ),
    "compliance": (
        "Synthetic data for demonstration only. Not legal or regulatory advice. Account actions and "
        "regulatory filings remain the responsibility of qualified compliance staff."
    ),
}

PRODUCTION_NOTE = (
    "The bundled calibration sets are small so the example runs quickly. For production, calibrate on "
    "at least the recommended number of labelled examples from your own traffic (about 1,000 for "
    "alpha=0.10), audit on held-out data, and monitor for drift."
)


def banner(title: str, domain: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    for line in textwrap.wrap(DISCLAIMERS[domain], 78):
        print(line)
    print()


def scenario(case_id: str, text: str) -> None:
    print("-" * 78)
    print(f"{case_id}: {text}")


def decisions(items: Iterable[GuardDecision]) -> None:
    for d in items:
        print(f"    guard: {d.action.upper():<9} {d.tool} -> {d.reason}")
        if d.guarantee:
            print(f"           guarantee: {d.guarantee}")


def footer() -> None:
    print("-" * 78)
    for line in textwrap.wrap(PRODUCTION_NOTE, 78):
        print(line)
