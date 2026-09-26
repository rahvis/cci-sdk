"""Generate ``credit_applications.jsonl``: synthetic consumer-loan applications with risk tiers.

Every record is synthetic. No real applicant, lender or credit bureau data
is used, and the tiering policy below is invented for this example: it is
not a real underwriting policy and not financial, credit or legal advice.

Each line is ``{"id": ..., "context": {...}, "label": ...}``:

- ``context`` is exactly what the calibrated guard evaluates, built by
  ``application_context`` (the example imports the same function, so the
  calibration and inference contexts are identical in keys and rendering);
- ``label`` is the tier the synthetic tiering policy assigns (``TIERING_POLICY``),
  except for about 4% of files where a documented underwriter override moved
  the recorded tier by one step for reasons that are not in the file. Those
  records are the label noise a real portfolio has, and they keep the
  calibrated thresholds realistic.

Features deliberately exclude protected characteristics (age, sex, race,
ethnicity, religion, national origin, marital status) and obvious proxies
such as ZIP code. The application channel is recorded so calibration can be
checked per channel; the policy never uses it.

The broker channel is intentionally small (40 of 320 files) so its
per-channel calibration status is visibly less certain than the others.

Run (stdlib only, deterministic)::

    python examples/agents/data/generate_credit_applications.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

SEED = 7001
CHANNEL_SIZES = {"branch": 140, "online": 140, "broker": 40}
OVERRIDE_RATE = 0.04
OUTPUT = Path(__file__).resolve().parent / "credit_applications.jsonl"

TIERS = ("A", "B", "C", "D", "E")
CREDIT_SCORE_BANDS = ("800+", "740-799", "670-739", "580-669", "below 580")
PAYMENT_HISTORY = (
    "no late payments",
    "one 30-day late payment",
    "two or more late payments, or one 60-day delinquency",
    "collection or charge-off",
)
INCOME_VERIFICATION = (
    "verified with payroll or tax documents",
    "verified with bank statements",
    "stated, not verified",
)
CHANNELS = tuple(CHANNEL_SIZES)

TIERING_POLICY = (
    "Synthetic tiering policy (demonstration only, not a real underwriting policy).",
    "1. Credit score band points: 800+ = 0, 740-799 = 1, 670-739 = 2, 580-669 = 4, below 580 = 6.",
    "2. Debt-to-income points: up to 20% = 0, 21-35% = 1, 36-43% = 2, 44-50% = 3, above 50% = 5.",
    "3. Payment history points (last 24 months): no late payments = 0, one 30-day late payment = 1, "
    "two or more late payments or one 60-day delinquency = 3, collection or charge-off = 5.",
    "4. Add 2 points for a thin credit file and 1 point when income is stated but not verified.",
    "5. Tier from total points: 0-1 = A, 2-3 = B, 4-5 = C, 6-8 = D, 9 or more = E.",
    "6. Overrides: a thin file is never tier A; stated income is never better than tier C; "
    "a collection or charge-off is never better than tier D.",
    "7. The application channel never changes the tier.",
)

BAND_POINTS = {"800+": 0, "740-799": 1, "670-739": 2, "580-669": 4, "below 580": 6}
HISTORY_POINTS = {PAYMENT_HISTORY[0]: 0, PAYMENT_HISTORY[1]: 1, PAYMENT_HISTORY[2]: 3, PAYMENT_HISTORY[3]: 5}
THIN_FILE = "thin (fewer than 3 tradelines or under 2 years of history)"
ESTABLISHED_FILE = "established"


def dti_points(dti_pct: int) -> int:
    if dti_pct <= 20:
        return 0
    if dti_pct <= 35:
        return 1
    if dti_pct <= 43:
        return 2
    if dti_pct <= 50:
        return 3
    return 5


def application_context(application: Mapping[str, Any]) -> dict[str, Any]:
    """The evaluation context for one application: the same keys and formatting everywhere.

    ``application`` holds the raw fields ``application_id``, ``channel``,
    ``credit_score_band``, ``debt_to_income_pct`` (int), ``payment_history``,
    ``thin_file`` (bool) and ``income_verification``. Unknown values raise
    ``ValueError``, which the guard turns into an escalation.
    """
    band = application["credit_score_band"]
    history = application["payment_history"]
    income = application["income_verification"]
    channel = application["channel"]
    dti = int(application["debt_to_income_pct"])
    if band not in CREDIT_SCORE_BANDS:
        raise ValueError(f"unknown credit score band {band!r}")
    if history not in PAYMENT_HISTORY:
        raise ValueError(f"unknown payment history {history!r}")
    if income not in INCOME_VERIFICATION:
        raise ValueError(f"unknown income verification {income!r}")
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}")
    if not 0 <= dti <= 100:
        raise ValueError(f"debt-to-income {dti} is outside 0-100%")
    return {
        "application_id": str(application["application_id"]),
        "channel": channel,
        "credit_score_band": band,
        "debt_to_income": f"{dti}%",
        "payment_history_24_months": history,
        "credit_file": THIN_FILE if application["thin_file"] else ESTABLISHED_FILE,
        "income_verification": income,
    }


def policy_tier(application: Mapping[str, Any]) -> str:
    """The tier ``TIERING_POLICY`` assigns to a raw application."""
    points = (BAND_POINTS[application["credit_score_band"]]
              + dti_points(int(application["debt_to_income_pct"]))
              + HISTORY_POINTS[application["payment_history"]])
    thin = bool(application["thin_file"])
    stated = application["income_verification"] == INCOME_VERIFICATION[2]
    points += 2 if thin else 0
    points += 1 if stated else 0
    if points <= 1:
        tier = 0
    elif points <= 3:
        tier = 1
    elif points <= 5:
        tier = 2
    elif points <= 8:
        tier = 3
    else:
        tier = 4
    if thin:
        tier = max(tier, 1)
    if stated:
        tier = max(tier, 2)
    if application["payment_history"] == PAYMENT_HISTORY[3]:
        tier = max(tier, 3)
    return TIERS[tier]


def _pick(rng: random.Random, values: tuple[str, ...], weights: tuple[float, ...]) -> str:
    return rng.choices(values, weights=weights, k=1)[0]


def _application(rng: random.Random, application_id: str, channel: str) -> dict[str, Any]:
    band_index = rng.choices(range(5), weights=(0.20, 0.27, 0.27, 0.16, 0.10), k=1)[0]
    dti_mean = (17, 23, 30, 37, 42)[band_index]
    dti = int(round(min(65, max(5, rng.gauss(dti_mean, 8)))))
    history_weights = (
        (0.93, 0.06, 0.01, 0.00),
        (0.82, 0.14, 0.03, 0.01),
        (0.62, 0.24, 0.11, 0.03),
        (0.35, 0.30, 0.25, 0.10),
        (0.15, 0.25, 0.30, 0.30),
    )[band_index]
    thin_rate = {"branch": 0.10, "online": 0.25, "broker": 0.12}[channel]
    income_weights = {"branch": (0.75, 0.20, 0.05), "online": (0.55, 0.35, 0.10),
                      "broker": (0.45, 0.25, 0.30)}[channel]
    return {
        "application_id": application_id,
        "channel": channel,
        "credit_score_band": CREDIT_SCORE_BANDS[band_index],
        "debt_to_income_pct": dti,
        "payment_history": _pick(rng, PAYMENT_HISTORY, history_weights),
        "thin_file": rng.random() < thin_rate,
        "income_verification": _pick(rng, INCOME_VERIFICATION, income_weights),
    }


def generate() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    channels = [c for c, n in CHANNEL_SIZES.items() for _ in range(n)]
    rng.shuffle(channels)
    rows = []
    for i, channel in enumerate(channels, start=1):
        application_id = f"CAL-{i:04d}"
        application = _application(rng, application_id, channel)
        tier = policy_tier(application)
        if rng.random() < OVERRIDE_RATE:
            # A documented underwriter override for reasons that are not in the file.
            index = TIERS.index(tier) + rng.choice((-1, 1))
            tier = TIERS[min(4, max(0, index))]
        rows.append({"id": application_id, "context": application_context(application), "label": tier})
    return rows


def main() -> int:
    rows = generate()
    with OUTPUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    labels = Counter(row["label"] for row in rows)
    channels = Counter(row["context"]["channel"] for row in rows)
    print(f"wrote {len(rows)} synthetic applications to {OUTPUT.name}")
    print("tiers:    " + ", ".join(f"{t}={labels[t]}" for t in TIERS))
    print("channels: " + ", ".join(f"{c}={channels[c]}" for c in CHANNELS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
