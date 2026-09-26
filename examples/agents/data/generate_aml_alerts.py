"""Generate ``aml_alerts.jsonl``: synthetic anti-money-laundering alerts with hold decisions.

Every alert, customer, account and amount in this file is invented. The
investigation guideline below is a simplified, fictional example written for
this demonstration; it is not a regulatory standard and not legal advice.

Each output line is ``{"id": ..., "context": {...}, "label": ...}``:

- ``context`` is exactly what the calibrated guard evaluates: the written
  guideline plus the alert summary, produced by ``alert_context()``. The
  example agent builds its inference-time context with the same function, so
  calibration and serving contexts have identical structure and wording.
- ``label`` is ``True`` when a temporary account hold is warranted under the
  guideline, as decided by an investigator.

How labels are made
-------------------
Labels come from the written guideline (``hold_warranted()``), applied by an
investigator to the account's full transaction history. The alert summary in
the context is a slightly lossy view of that history: a deposit counted in
the 8,000 to 9,999 USD band, a percentage or a wire total can differ a little
(``full_history_view()``). Alerts far from every threshold are unaffected;
alerts that hinge on a single near-threshold pattern are genuinely ambiguous
from the summary alone, which is what keeps the calibration realistic
instead of trivially separable. On top of that, about 1.5% of decisions are
reviewer disagreement (flipped at random).

Run it from anywhere; it is deterministic (seeded) and uses only the
standard library::

    python examples/agents/data/generate_aml_alerts.py            # writes aml_alerts.jsonl
    python examples/agents/data/generate_aml_alerts.py --stats    # also prints label balance
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional

SEED = 20260924
DEFAULT_COUNT = 360
OUTPUT = Path(__file__).resolve().parent / "aml_alerts.jsonl"

GUIDELINE_ID = "SYN-AML-IG-7"

# The guideline travels inside every context, word for word, so the evidence
# model reads the same policy at calibration and at serving time.
GUIDELINE = [
    f"Investigation guideline {GUIDELINE_ID} (fictional, for demonstration only).",
    "A temporary account hold is warranted when the alert shows a red-flag pattern that the "
    "file does not explain.",
    "Red-flag patterns, all measured over the 30-day alert window:",
    "S. Structuring: three or more cash deposits between 8,000 and 9,999 USD.",
    "V. Rapid movement: at least 80% of deposited funds sent out within 48 hours, with 30-day "
    "deposits of 25,000 USD or more.",
    "J. High-risk jurisdictions: wires totalling 20,000 USD or more to counterparties in "
    "jurisdictions on the bank's high-risk list.",
    "P. Above profile: 30-day deposits of at least 10,000 USD and more than 3 times the expected "
    "monthly deposits recorded at onboarding.",
    "Rule 1. Pattern S warrants a hold on its own. Documentation of the source of funds does not "
    "explain structuring.",
    "Rule 2. Patterns V, J and P warrant a hold unless verified documentation accounts for the "
    "flagged funds.",
    "Rule 3. With only partial documentation, one of V, J or P alone does not warrant a hold; two "
    "or more do.",
    "Rule 4. With no red-flag pattern, no hold is warranted.",
]

HOLD_QUESTION = "Under the investigation guideline in the context, is a temporary hold on this account warranted?"


def alert_context(alert: dict[str, Any]) -> dict[str, Any]:
    """The context the guard evaluates for one alert, at calibration and at serving time.

    Only case facts go in; free text written by the agent (such as the
    ``reason`` argument of a hold) never does, because the calibration set
    was scored without it.
    """
    return {
        "guideline": list(GUIDELINE),
        "alert": {
            "alert_id": alert["alert_id"],
            "account_id": alert["account_id"],
            "monitoring_rule": alert["monitoring_rule"],
            "customer": dict(alert["customer"]),
            "activity_30d": dict(alert["activity_30d"]),
            "documentation": dict(alert["documentation"]),
            "prior_alerts_12m": alert["prior_alerts_12m"],
        },
    }


def red_flags(alert: dict[str, Any]) -> list[str]:
    """The guideline's red-flag patterns present in an alert (a subset of S, V, J, P)."""
    act = alert["activity_30d"]
    expected = alert["customer"]["expected_monthly_deposits_usd"]
    flags = []
    if act["cash_deposits_8000_to_9999"] >= 3:
        flags.append("S")
    if act["sent_out_within_48h_pct"] >= 80 and act["deposits_total_usd"] >= 25_000:
        flags.append("V")
    if act["wires_to_high_risk_jurisdictions_usd"] >= 20_000:
        flags.append("J")
    if act["deposits_total_usd"] >= 10_000 and act["deposits_total_usd"] > 3 * expected:
        flags.append("P")
    return flags


def hold_warranted(alert: dict[str, Any]) -> bool:
    """The written guideline, applied to the alert facts (rules 1 to 4)."""
    flags = red_flags(alert)
    status = alert["documentation"]["status"]
    if "S" in flags:
        return True
    others = [f for f in flags if f != "S"]
    if not others:
        return False
    if status == "verified":
        return False
    if status == "partial":
        return len(others) >= 2
    return True


# ---------------------------------------------------------------------------
# synthetic alert archetypes
# ---------------------------------------------------------------------------

RETAIL_JOBS = ["nurse", "teacher", "software engineer", "retired", "electrician", "sales associate",
               "graduate student", "accountant", "truck driver", "pharmacist"]
SMALL_BUSINESSES = ["consulting firm", "landscaping company", "dental practice", "online retailer",
                    "import-export trading company", "freight forwarder", "construction subcontractor"]
CASH_BUSINESSES = ["restaurant", "laundromat", "convenience store", "car wash", "used-car dealer",
                   "nail salon", "parking operator"]

NOTES = {
    "none": ["No explanation on file.", "Customer has not responded to the request for information.",
             "No outreach made yet."],
    "partial": ["Customer states the funds are business receipts; ledger requested, not yet provided.",
                "Two of five invoices provided for the wires; remaining invoices outstanding.",
                "Customer says the funds come from a family loan; no loan agreement on file.",
                "Supplier contract provided, but amounts do not match the wires."],
    "verified": ["Closing statement for a property sale on file; amount matches the deposits.",
                 "Probate court letter on file; deposits match the inheritance distribution.",
                 "Signed supply contracts and customs declarations on file for every wire.",
                 "Payroll agreement on file; outbound payments match the employer's payroll run.",
                 "Bill of sale for a business asset on file; buyer and amount verified."],
}

RULE_NAMES = {"S": "cash structuring pattern", "V": "rapid movement of funds",
              "J": "wires to high-risk jurisdictions", "P": "activity above expected profile"}


def _money(rng: random.Random, low: float, high: float, step: int = 50) -> int:
    return int(round(rng.uniform(low, high) / step) * step)


def _doc(rng: random.Random, weights: dict[str, float]) -> dict[str, str]:
    status = rng.choices(list(weights), weights=list(weights.values()))[0]
    return {"status": status, "note": rng.choice(NOTES[status])}


def _customer(rng: random.Random, kind: str) -> dict[str, Any]:
    if kind == "retail":
        return {"segment": "retail", "occupation_or_business": rng.choice(RETAIL_JOBS),
                "tenure_months": rng.randint(3, 240),
                "expected_monthly_deposits_usd": _money(rng, 2_500, 12_000, 250)}
    if kind == "small_business":
        return {"segment": "small business", "occupation_or_business": rng.choice(SMALL_BUSINESSES),
                "tenure_months": rng.randint(6, 180),
                "expected_monthly_deposits_usd": _money(rng, 20_000, 120_000, 1_000)}
    return {"segment": "cash-intensive business", "occupation_or_business": rng.choice(CASH_BUSINESSES),
            "tenure_months": rng.randint(12, 240),
            "expected_monthly_deposits_usd": _money(rng, 40_000, 140_000, 1_000)}


def _activity(rng: random.Random, *, in_range: int, other_cash: int, cash_low: float, cash_high: float,
              non_cash_usd: int, branches: int, sent_out_pct: int, hr_wires: int) -> dict[str, Any]:
    in_range_amounts = [_money(rng, 8_000, 9_950, 25) for _ in range(in_range)]
    other = [_money(rng, cash_low, cash_high, 25) for _ in range(other_cash)]
    other = [a if not 8_000 <= a <= 9_999 else a - 2_500 for a in other]
    cash_total = sum(in_range_amounts) + sum(other)
    deposits = cash_total + non_cash_usd
    largest = max(in_range_amounts + other + [non_cash_usd]) if deposits else 0
    return {
        "deposits_total_usd": deposits,
        "cash_deposits_count": in_range + other_cash,
        "cash_deposits_total_usd": cash_total,
        "cash_deposits_8000_to_9999": in_range,
        "largest_single_deposit_usd": largest,
        "branches_used": branches,
        "sent_out_within_48h_pct": sent_out_pct,
        "wires_to_high_risk_jurisdictions_usd": hr_wires,
    }


def _structuring(rng: random.Random) -> dict[str, Any]:
    kind = rng.choice(["retail", "retail", "small_business"])
    return {"customer": _customer(rng, kind),
            "activity_30d": _activity(rng, in_range=rng.randint(3, 8), other_cash=rng.randint(0, 3),
                                      cash_low=500, cash_high=4_000, non_cash_usd=_money(rng, 0, 6_000),
                                      branches=rng.randint(2, 5), sent_out_pct=rng.randint(10, 95),
                                      hr_wires=rng.choice([0, 0, 0, _money(rng, 3_000, 30_000, 500)])),
            "documentation": _doc(rng, {"none": 0.7, "partial": 0.2, "verified": 0.1})}


def _rapid_movement(rng: random.Random) -> dict[str, Any]:
    kind = rng.choice(["retail", "small_business"])
    return {"customer": _customer(rng, kind),
            "activity_30d": _activity(rng, in_range=rng.choice([0, 0, 1]), other_cash=rng.randint(0, 2),
                                      cash_low=500, cash_high=3_000,
                                      non_cash_usd=_money(rng, 28_000, 160_000, 500),
                                      branches=rng.randint(1, 2), sent_out_pct=rng.randint(84, 99),
                                      hr_wires=rng.choice([0, 0, _money(rng, 22_000, 90_000, 500)])),
            "documentation": _doc(rng, {"none": 0.6, "partial": 0.25, "verified": 0.15})}


def _high_risk_wires(rng: random.Random) -> dict[str, Any]:
    return {"customer": _customer(rng, rng.choice(["small_business", "small_business", "retail"])),
            "activity_30d": _activity(rng, in_range=0, other_cash=rng.randint(0, 3), cash_low=300,
                                      cash_high=3_000, non_cash_usd=_money(rng, 20_000, 200_000, 500),
                                      branches=1, sent_out_pct=rng.randint(20, 75),
                                      hr_wires=_money(rng, 23_000, 250_000, 500)),
            "documentation": _doc(rng, {"none": 0.45, "partial": 0.3, "verified": 0.25})}


def _documented_large_deposit(rng: random.Random) -> dict[str, Any]:
    customer = _customer(rng, "retail")
    return {"customer": customer,
            "activity_30d": _activity(rng, in_range=0, other_cash=rng.randint(0, 2), cash_low=200,
                                      cash_high=2_000, non_cash_usd=_money(rng, 60_000, 420_000, 1_000),
                                      branches=1, sent_out_pct=rng.randint(0, 40), hr_wires=0),
            "documentation": _doc(rng, {"verified": 0.75, "partial": 0.15, "none": 0.10})}


def _cash_business(rng: random.Random) -> dict[str, Any]:
    customer = _customer(rng, "cash")
    expected = customer["expected_monthly_deposits_usd"]
    in_range = rng.choices([0, 1, 2, 3], weights=[0.45, 0.3, 0.15, 0.10])[0]
    count = rng.randint(14, 32)
    target = expected * rng.uniform(0.7, 1.6)
    per = max(600.0, (target - in_range * 9_000) / max(1, count - in_range))
    return {"customer": customer,
            "activity_30d": _activity(rng, in_range=in_range, other_cash=count - in_range,
                                      cash_low=per * 0.6, cash_high=min(per * 1.4, 7_900),
                                      non_cash_usd=_money(rng, 0, 15_000), branches=rng.randint(1, 3),
                                      sent_out_pct=rng.randint(10, 60), hr_wires=0),
            "documentation": _doc(rng, {"none": 0.5, "partial": 0.3, "verified": 0.2})}


def _single_cash_deposit(rng: random.Random) -> dict[str, Any]:
    return {"customer": _customer(rng, "retail"),
            "activity_30d": _activity(rng, in_range=rng.choice([1, 1, 2]), other_cash=rng.randint(0, 3),
                                      cash_low=100, cash_high=1_500, non_cash_usd=_money(rng, 2_000, 9_000),
                                      branches=rng.randint(1, 2), sent_out_pct=rng.randint(0, 50), hr_wires=0),
            "documentation": _doc(rng, {"none": 0.6, "partial": 0.25, "verified": 0.15})}


def _partial_documentation(rng: random.Random) -> dict[str, Any]:
    customer = _customer(rng, rng.choice(["retail", "small_business"]))
    expected = customer["expected_monthly_deposits_usd"]
    wants = set(rng.sample(["V", "J", "P"], rng.choice([1, 2, 2])))
    non_cash = _money(rng, 26_000, 90_000, 500)
    if "P" in wants:
        non_cash = max(non_cash, int(expected * rng.uniform(3.3, 6.0)))
    else:
        non_cash = min(non_cash, int(expected * rng.uniform(1.2, 2.6)))
    return {"customer": customer,
            "activity_30d": _activity(rng, in_range=rng.choice([0, 1]), other_cash=rng.randint(0, 2),
                                      cash_low=300, cash_high=2_000, non_cash_usd=max(non_cash, 26_000),
                                      branches=rng.randint(1, 2),
                                      sent_out_pct=rng.randint(84, 97) if "V" in wants else rng.randint(10, 70),
                                      hr_wires=_money(rng, 22_000, 70_000, 500) if "J" in wants else 0),
            "documentation": _doc(rng, {"partial": 1.0})}


def _near_threshold(rng: random.Random) -> dict[str, Any]:
    customer = _customer(rng, rng.choice(["retail", "small_business"]))
    expected = customer["expected_monthly_deposits_usd"]
    non_cash = int(expected * rng.uniform(2.6, 3.4)) if rng.random() < 0.4 else _money(rng, 18_000, 34_000, 500)
    return {"customer": customer,
            "activity_30d": _activity(rng, in_range=rng.choice([2, 2, 3]), other_cash=rng.randint(0, 2),
                                      cash_low=300, cash_high=2_500, non_cash_usd=non_cash,
                                      branches=rng.randint(1, 3), sent_out_pct=rng.randint(74, 86),
                                      hr_wires=rng.choice([0, _money(rng, 16_000, 24_000, 250)])),
            "documentation": _doc(rng, {"none": 0.5, "partial": 0.5})}


ARCHETYPES = [
    (_structuring, 0.16), (_rapid_movement, 0.10), (_high_risk_wires, 0.10),
    (_documented_large_deposit, 0.13), (_cash_business, 0.12), (_single_cash_deposit, 0.16),
    (_partial_documentation, 0.13), (_near_threshold, 0.10),
]


def make_alert(rng: random.Random, alert_id: str) -> tuple[dict[str, Any], bool]:
    """One synthetic alert and its investigator decision."""
    builder = rng.choices([a for a, _ in ARCHETYPES], weights=[w for _, w in ARCHETYPES])[0]
    alert = builder(rng)
    flags = red_flags(alert)
    rule = RULE_NAMES[flags[0]] if flags else rng.choice(list(RULE_NAMES.values()))
    alert.update({"alert_id": alert_id, "account_id": f"ACC-{rng.randint(10_000, 99_999)}",
                  "monitoring_rule": rule, "prior_alerts_12m": rng.choices([0, 1, 2, 3], [0.6, 0.25, 0.1, 0.05])[0]})
    label = hold_warranted(full_history_view(rng, alert))
    if rng.random() < 0.015:
        label = not label  # reviewer disagreement
    return alert, label


def full_history_view(rng: random.Random, alert: dict[str, Any]) -> dict[str, Any]:
    """What the investigator saw: the alert facts, differing slightly from the summary."""
    act = dict(alert["activity_30d"])
    act["cash_deposits_8000_to_9999"] = max(0, act["cash_deposits_8000_to_9999"]
                                            + rng.choices([-1, 0, 1], weights=[0.2, 0.6, 0.2])[0])
    act["sent_out_within_48h_pct"] = min(100, max(0, act["sent_out_within_48h_pct"] + rng.randint(-4, 4)))
    act["wires_to_high_risk_jurisdictions_usd"] = int(act["wires_to_high_risk_jurisdictions_usd"]
                                                      * rng.uniform(0.9, 1.1))
    act["deposits_total_usd"] = int(act["deposits_total_usd"] * rng.uniform(0.93, 1.07))
    return {**alert, "activity_30d": act}


def generate(count: int = DEFAULT_COUNT, *, seed: int = SEED, first_id: int = 1001,
             prefix: str = "A") -> list[dict[str, Any]]:
    """``count`` labelled examples, ``{"id", "context", "label"}``, deterministically from ``seed``."""
    rng = random.Random(seed)
    out = []
    for i in range(count):
        alert_id = f"{prefix}-{first_id + i}"
        alert, label = make_alert(rng, alert_id)
        out.append({"id": alert_id, "context": alert_context(alert), "label": label})
    return out


def write(path: Path = OUTPUT, count: int = DEFAULT_COUNT, seed: int = SEED) -> list[dict[str, Any]]:
    rows = generate(count, seed=seed)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic AML alert calibration set.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--stats", action="store_true", help="print label balance by red-flag pattern")
    args = parser.parse_args(argv)
    rows = write(args.out, args.count, args.seed)
    warranted = sum(r["label"] for r in rows)
    print(f"wrote {len(rows)} alerts to {args.out} ({warranted} hold warranted, {len(rows) - warranted} not)")
    if args.stats:
        by_flags: dict[str, list[int]] = {}
        for r in rows:
            key = "+".join(red_flags(r["context"]["alert"])) or "none"
            key = f"{key} / {r['context']['alert']['documentation']['status']}"
            by_flags.setdefault(key, [0, 0])[int(r["label"])] += 1
        for key, (neg, pos) in sorted(by_flags.items()):
            print(f"  {key:<24} warranted {pos:>3}   not warranted {neg:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
