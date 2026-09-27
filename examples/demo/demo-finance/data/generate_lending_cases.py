"""Synthetic lending/credit-tiering dataset generator.

SYNTHETIC DATA FOR DEMONSTRATION ONLY. Not financial advice, not a real
underwriting system. Seeded and deterministic; no network access.

Ground-truth tier comes from a fixed, invented underwriting policy
(income, debt-to-income ratio, credit history length, prior defaults,
employment) applied consistently across channels — real channels differ
only in application-quality noise, so the ground truth itself is fair.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SEED = 20260928
OUT_DIR = Path(__file__).resolve().parent

CHANNELS = ("online", "branch", "partner")
TIERS = ("prime", "near_prime", "subprime", "decline")
N_CALIBRATION_PER_CHANNEL = 20
N_TEST_PER_CHANNEL = 4

# Partner-channel applications carry more noisy/incomplete self-reported
# data in this synthetic world — a real reason a blended coverage number
# could hide a weaker channel.
CHANNEL_NOISE = {"online": 0.05, "branch": 0.02, "partner": 0.18}


def tier_from_facts(income: float, dti: float, history_years: float, prior_defaults: int,
                     employment: str) -> str:
    """A fixed, rule-based underwriting policy (invented, but internally consistent
    with how a careful reviewer would actually reason about these same facts —
    unlike an additive score, a hard DTI/history/default cutoff should mostly
    agree with a reasonable model's own judgment, so the interesting gap is
    channel-conditional coverage and confidence calibration, not gross
    disagreement on the policy itself)."""
    if employment == "unemployed" or prior_defaults >= 2:
        return "decline"
    if prior_defaults >= 1:
        return "subprime" if dti < 0.5 else "decline"
    if dti > 0.5 or history_years < 1:
        return "subprime"
    if dti <= 0.35 and history_years >= 3 and income >= 35000:
        return "prime"
    if dti <= 0.45 and history_years >= 1.5:
        return "near_prime"
    return "subprime"


_TIER_ZONES = {
    # Facts drawn comfortably inside each zone (margin from any boundary)
    # so a careful reader has a clear, confidently-correct answer for
    # non-trap cases; boundaries only get tested deliberately, via traps.
    "prime": {"dti": (0.10, 0.28), "history": (4.0, 15.0), "income": (45000, 140000)},
    "near_prime": {"dti": (0.36, 0.42), "history": (2.0, 6.0), "income": (28000, 90000)},
    "subprime": {"dti": (0.52, 0.65), "history": (1.5, 8.0), "income": (22000, 70000)},
    "decline": {"dti": (0.20, 0.50), "history": (0.2, 3.0), "income": (18000, 50000)},
}


def make_case(rng: random.Random, channel: str, trap: bool = False) -> dict[str, Any]:
    employment = "employed"
    prior_defaults = 0
    requested_amount = rng.uniform(2000, 40000)

    if not trap:
        target_tier = rng.choice(list(_TIER_ZONES))
        zone = _TIER_ZONES[target_tier]
        income = rng.uniform(*zone["income"])
        history_years = rng.uniform(*zone["history"])
        dti = rng.uniform(*zone["dti"])
        if target_tier == "decline":
            employment = "unemployed"

    if trap:
        # Looks strong at a glance (good income, employed) but the policy
        # still marks it down for a real reason a skimming reader might miss.
        income = rng.uniform(80000, 140000)
        employment = "employed"
        dti = rng.uniform(0.15, 0.30)  # a comfortable DTI: the trap is defaults/history, not DTI
        trap_kind = rng.choice(["old_default", "thin_file_big_ask"])
        if trap_kind == "old_default":
            prior_defaults = 1
            history_years = rng.uniform(4, 10)
            note = "one default on record, a few years back, otherwise a strong file"
        else:
            history_years = rng.uniform(0.5, 1.2)
            requested_amount = income * rng.uniform(0.5, 0.8)
            note = "very short credit history relative to a large requested amount"
    else:
        note = "no unusual factors"

    # Channel-specific noise perturbs the *reported* facts a customer's
    # application shows, without changing the underlying true policy score.
    noise = CHANNEL_NOISE[channel]
    reported_income = income * (1 + rng.uniform(-noise, noise))

    tier = tier_from_facts(income, dti, history_years, prior_defaults, employment)

    facts = {
        "channel": channel,
        "reported_annual_income": round(reported_income, -2),
        "debt_to_income_ratio": round(dti, 2),
        "credit_history_years": round(history_years, 1),
        "prior_defaults": prior_defaults,
        "employment_status": employment,
        "requested_amount": round(requested_amount, -2),
    }
    message = (
        f"Application via the {channel} channel. Reported annual income "
        f"${facts['reported_annual_income']:,.0f}, debt-to-income ratio "
        f"{facts['debt_to_income_ratio']:.2f}, {facts['credit_history_years']} years of credit "
        f"history, {facts['prior_defaults']} prior default(s), {employment.replace('_', ' ')}. "
        f"Requesting ${facts['requested_amount']:,.0f}. ({note})"
    )
    return {"facts": facts, "message": message, "tier": tier, "is_trap": trap}


def generate(rng: random.Random, n_per_channel: int, trap_fraction: float) -> list[dict[str, Any]]:
    out = []
    for channel in CHANNELS:
        n_trap = max(1, round(n_per_channel * trap_fraction)) if trap_fraction > 0 else 0
        for i in range(n_per_channel):
            out.append(make_case(rng, channel, trap=(i < n_trap)))
    rng.shuffle(out)
    return out


def main() -> None:
    rng = random.Random(SEED)
    calibration = generate(rng, N_CALIBRATION_PER_CHANNEL, trap_fraction=0.15)
    test = generate(rng, N_TEST_PER_CHANNEL, trap_fraction=0.4)

    path = OUT_DIR / "lending_cases.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i, case in enumerate(calibration):
            fh.write(json.dumps({"case_id": f"LC-{i + 1:04d}", "split": "calibration", **case}) + "\n")
        for i, case in enumerate(test):
            fh.write(json.dumps({"case_id": f"LT-{i + 1:04d}", "split": "test", **case}) + "\n")

    from collections import Counter
    print(f"wrote {len(calibration)} calibration + {len(test)} test cases to {path}")
    print("calibration tier/channel counts:", Counter((c["facts"]["channel"], c["tier"]) for c in calibration))
    print("test tier/channel counts:", Counter((c["facts"]["channel"], c["tier"]) for c in test))


if __name__ == "__main__":
    main()
