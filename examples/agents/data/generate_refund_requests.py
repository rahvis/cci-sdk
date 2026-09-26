#!/usr/bin/env python3
"""Generate ``refund_requests.jsonl``: synthetic, labelled refund decisions for calibration.

Every record in the output is synthetic. The orders, amounts, customers and
policy are invented for the LangChain refunds example
(``examples/agents/langchain/finance_refund_agent.py``); nothing here is
financial, credit or legal advice.

What one line holds
-------------------
::

    {"id": "refund-0001",
     "context": {"policy": REFUND_POLICY, "order": {...}, "proposed_refund": {"amount": ..., "reason": ...}},
     "label": true}

``context`` is exactly what the refunds guard evaluates: both this generator
and the example build it with ``refund_context()`` below, so the calibration
scores and the live scores come from the same scoring function (the
exchangeability the guarantee rests on). ``label`` is True when issuing the
proposed refund is correct under ``REFUND_POLICY``.

How labels are made
-------------------
``policy_violations()`` applies the written policy rule by rule, and the label
is True when no rule is violated. Two kinds of cases keep the data from being
unrealistically clean:

- Exception requests (about 5%): plus-tier customers up to 15 days past
  their window. Rule 8 leaves these to a refund manager, so the label is the
  manager's recorded decision (a seeded coin flip), which cannot be read off
  the record.
- Reviewer inconsistency (2%): labels flipped at random, as in any
  human-labelled export.

The generator is deterministic (seeded, standard library only)::

    python examples/agents/data/generate_refund_requests.py          # rewrite refund_requests.jsonl
    python examples/agents/data/generate_refund_requests.py --check  # verify the committed file
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

SEED = 20260924
N_EXAMPLES = 320
OUTPUT = Path(__file__).resolve().parent / "refund_requests.jsonl"

REFUND_POLICY = """\
Refund policy for card and e-commerce orders (synthetic example policy, version 2026-09).
Issue a refund automatically only when every rule below holds for the order record and the proposed refund.
1. Reason codes: damaged_item, wrong_item, not_delivered, changed_mind, duplicate_charge.
2. Window. damaged_item, wrong_item and changed_mind: within 30 days of delivery (45 days for plus-tier \
customers). not_delivered: the carrier has not confirmed delivery and the order was placed at least 10 and at \
most 90 days ago. duplicate_charge: within 120 days of purchase.
3. Amount. The refund is at most amount_paid minus refunded_to_date. For changed_mind the maximum is the item \
price: original shipping is not refunded.
4. Category. gift_card and digital_download orders, and final-sale items, are refunded only for duplicate_charge.
5. One refund per order. If refunded_to_date is above zero, no further refund is issued under this policy.
6. Holds. No refund while the order has an open fraud flag or an open chargeback.
7. Evidence. damaged_item and wrong_item refunds need a photo on file.
8. Exceptions. A refund manager may extend the window in rule 2 by up to 15 days for plus-tier customers; \
those requests are decided case by case and are never issued automatically."""

REASONS = ("damaged_item", "wrong_item", "not_delivered", "changed_mind", "duplicate_charge")
NON_REFUNDABLE_CATEGORIES = ("gift_card", "digital_download")
PHYSICAL_CATEGORIES = {  # category: (lowest, highest) item price in USD
    "apparel": (12, 240),
    "electronics": (25, 900),
    "home_goods": (8, 350),
    "books": (6, 60),
    "beauty": (6, 120),
    "toys": (8, 150),
}
SHIPPING_FEES = (0.0, 4.99, 7.99, 12.0)


# ---------------------------------------------------------------------------
# The record format shared with the example (one definition, used by both)
# ---------------------------------------------------------------------------


def make_order(
    order_id: str,
    *,
    category: str,
    item_price: float,
    shipping_fee: float = 0.0,
    final_sale: bool = False,
    purchased_days_ago: int,
    delivery_status: str,
    delivered_days_ago: Optional[int],
    customer_tier: str = "standard",
    photo_on_file: bool = False,
    refunded_to_date: float = 0.0,
    open_fraud_flag: bool = False,
    open_chargeback: bool = False,
) -> dict[str, Any]:
    """One order record, as the order system returns it (same keys for every order)."""
    return {
        "order_id": order_id,
        "category": category,
        "final_sale": final_sale,
        "item_price": round(float(item_price), 2),
        "shipping_fee": round(float(shipping_fee), 2),
        "amount_paid": round(float(item_price) + float(shipping_fee), 2),
        "payment_method": "card",
        "purchased_days_ago": int(purchased_days_ago),
        "delivery_status": delivery_status,
        "delivered_days_ago": None if delivered_days_ago is None else int(delivered_days_ago),
        "customer_tier": customer_tier,
        "photo_on_file": photo_on_file,
        "refunded_to_date": round(float(refunded_to_date), 2),
        "open_fraud_flag": open_fraud_flag,
        "open_chargeback": open_chargeback,
    }


def refund_context(order: dict[str, Any], amount: float, reason: str) -> dict[str, Any]:
    """The evaluation context for one proposed refund: calibration and inference both use this."""
    return {
        "policy": REFUND_POLICY,
        "order": dict(order),
        "proposed_refund": {"amount": round(float(amount), 2), "reason": reason},
    }


# ---------------------------------------------------------------------------
# The written policy, rule by rule
# ---------------------------------------------------------------------------


def return_window(order: dict[str, Any]) -> int:
    """Rule 2 window for delivered-item reasons, in days after delivery."""
    return 45 if order["customer_tier"] == "plus" else 30


def policy_violations(order: dict[str, Any], amount: float, reason: str) -> list[str]:
    """The rules of ``REFUND_POLICY`` this proposed refund breaks (empty: issuing it is correct).

    ``"rule8_manager_decides"`` marks a plus-tier request inside the 15-day
    exception band: the only problem is the window, and rule 8 hands it to a
    refund manager.
    """
    broken: list[str] = []
    remaining = round(order["amount_paid"] - order["refunded_to_date"], 2)
    window_problem = False
    if reason in ("damaged_item", "wrong_item", "changed_mind"):
        days = order["delivered_days_ago"]
        if order["delivery_status"] != "delivered" or days is None:
            broken.append("rule2_not_delivered_yet")
        elif days > return_window(order):
            window_problem = True
    elif reason == "not_delivered":
        if order["delivery_status"] == "delivered":
            broken.append("rule2_carrier_confirmed_delivery")
        if not 10 <= order["purchased_days_ago"] <= 90:
            broken.append("rule2_window")
    elif reason == "duplicate_charge":
        if order["purchased_days_ago"] > 120:
            broken.append("rule2_window")
    else:
        broken.append("rule1_unknown_reason")

    limit = order["item_price"] if reason == "changed_mind" else remaining
    if amount > min(limit, remaining) + 1e-9 or amount <= 0:
        broken.append("rule3_amount")
    if reason != "duplicate_charge" and (order["category"] in NON_REFUNDABLE_CATEGORIES or order["final_sale"]):
        broken.append("rule4_category")
    if order["refunded_to_date"] > 0:
        broken.append("rule5_prior_refund")
    if order["open_fraud_flag"] or order["open_chargeback"]:
        broken.append("rule6_hold")
    if reason in ("damaged_item", "wrong_item") and not order["photo_on_file"]:
        broken.append("rule7_no_photo")

    if window_problem:
        overdue = order["delivered_days_ago"] - return_window(order)
        if not broken and order["customer_tier"] == "plus" and overdue <= 15:
            broken.append("rule8_manager_decides")
        else:
            broken.append("rule2_window")
    return broken


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------

ARCHETYPES = (  # (name, weight)
    ("clean", 0.50),
    ("window", 0.08),
    ("amount_over", 0.05),
    ("changed_mind_shipping", 0.05),
    ("category", 0.06),
    ("prior_refund", 0.07),
    ("hold", 0.04),
    ("carrier_confirmed", 0.05),
    ("no_photo", 0.05),
    ("exception", 0.05),
)
LABEL_NOISE = 0.02


def _price(rng: random.Random, lo: float, hi: float) -> float:
    return max(round(rng.uniform(lo, hi)) - 0.01, 4.99)


def _baseline(rng: random.Random, order_id: str, reason: str) -> tuple[dict[str, Any], float]:
    """A request that satisfies the policy for ``reason``."""
    tier = "plus" if rng.random() < 0.25 else "standard"
    if reason == "duplicate_charge" and rng.random() < 0.25:
        category = rng.choice(NON_REFUNDABLE_CATEGORIES)
        item = _price(rng, 10, 150)
        shipping = 0.0
    else:
        category = rng.choice(sorted(PHYSICAL_CATEGORIES))
        item = _price(rng, *PHYSICAL_CATEGORIES[category])
        shipping = rng.choice(SHIPPING_FEES)

    if reason == "not_delivered":
        purchased, status, delivered = rng.randint(10, 90), rng.choice(["in_transit", "no_scan_since_label"]), None
    elif reason == "duplicate_charge":
        purchased = rng.randint(1, 120)
        if category in NON_REFUNDABLE_CATEGORIES:
            status, delivered = "digital_delivery", purchased
        elif purchased > 6:
            status, delivered = "delivered", purchased - rng.randint(2, 6)
        else:
            status, delivered = "in_transit", None
    else:
        window = 45 if tier == "plus" else 30
        delivered = rng.randint(0, window)
        purchased, status = delivered + rng.randint(2, 8), "delivered"

    order = make_order(
        order_id, category=category, item_price=item, shipping_fee=shipping,
        purchased_days_ago=purchased, delivery_status=status, delivered_days_ago=delivered,
        customer_tier=tier, photo_on_file=reason in ("damaged_item", "wrong_item") or rng.random() < 0.3,
    )
    if reason == "changed_mind":
        amount = order["item_price"]
    elif reason == "damaged_item" and rng.random() < 0.3:
        amount = round(order["item_price"] * rng.choice([0.25, 0.5]), 2)  # partial refund for minor damage
    elif reason == "duplicate_charge":
        amount = order["amount_paid"]
    else:
        amount = order["amount_paid"]
    return order, amount


def _mutate(rng: random.Random, archetype: str, order: dict[str, Any], amount: float, reason: str):
    """Apply one policy problem (or an exception request) to a clean baseline."""
    order = dict(order)
    if archetype == "window":
        if reason in ("damaged_item", "wrong_item", "changed_mind"):
            if order["customer_tier"] == "plus":
                order["delivered_days_ago"] = return_window(order) + rng.randint(16, 60)
            else:
                order["delivered_days_ago"] = return_window(order) + rng.choice([1, 2, 3, 5, 8, 14, 25, 40])
            order["purchased_days_ago"] = order["delivered_days_ago"] + rng.randint(2, 8)
        elif reason == "not_delivered":
            order["purchased_days_ago"] = rng.choice([rng.randint(2, 9), rng.randint(91, 150)])
        else:
            order["purchased_days_ago"] = rng.randint(121, 200)
            if order["delivery_status"] == "digital_delivery":
                order["delivered_days_ago"] = order["purchased_days_ago"]
            elif order["delivered_days_ago"] is not None:
                order["delivered_days_ago"] = order["purchased_days_ago"] - 3
    elif archetype == "amount_over":
        remaining = order["amount_paid"] - order["refunded_to_date"]
        amount = round(remaining * rng.choice([1.1, 1.25, 1.5, 2.0, 3.0]), 2)
    elif archetype == "changed_mind_shipping":
        reason = "changed_mind"
        if order["shipping_fee"] == 0:
            order["shipping_fee"] = rng.choice(SHIPPING_FEES[1:])
            order["amount_paid"] = round(order["item_price"] + order["shipping_fee"], 2)
        if order["delivery_status"] != "delivered" or order["delivered_days_ago"] is None:
            order["delivery_status"], order["delivered_days_ago"] = "delivered", rng.randint(1, 25)
            order["purchased_days_ago"] = order["delivered_days_ago"] + 4
        amount = order["amount_paid"]
    elif archetype == "category":
        if rng.random() < 0.5:
            order["final_sale"] = True
        else:
            order["category"] = rng.choice(NON_REFUNDABLE_CATEGORIES)
            order["shipping_fee"] = 0.0
            order["amount_paid"] = order["item_price"]
            amount = min(amount, order["amount_paid"])
    elif archetype == "prior_refund":
        prior = order["amount_paid"] if rng.random() < 0.6 else round(order["amount_paid"] * rng.choice([0.2, 0.5]), 2)
        order["refunded_to_date"] = prior
    elif archetype == "hold":
        order["open_fraud_flag" if rng.random() < 0.5 else "open_chargeback"] = True
    elif archetype == "carrier_confirmed":
        reason = "not_delivered"
        order["delivery_status"] = "delivered"
        order["purchased_days_ago"] = rng.randint(10, 60)
        order["delivered_days_ago"] = order["purchased_days_ago"] - rng.randint(2, 7)
        amount = order["amount_paid"]
    elif archetype == "no_photo":
        reason = rng.choice(["damaged_item", "wrong_item"])
        order["photo_on_file"] = False
        if order["delivery_status"] != "delivered" or order["delivered_days_ago"] is None:
            order["delivery_status"], order["delivered_days_ago"] = "delivered", rng.randint(0, 25)
            order["purchased_days_ago"] = order["delivered_days_ago"] + 3
        amount = min(amount, order["amount_paid"])
    elif archetype == "exception":
        reason = rng.choice(["damaged_item", "wrong_item", "changed_mind"])
        order["customer_tier"] = "plus"
        order["photo_on_file"] = True
        order["delivery_status"] = "delivered"
        order["delivered_days_ago"] = 45 + rng.randint(1, 15)
        order["purchased_days_ago"] = order["delivered_days_ago"] + rng.randint(2, 8)
        amount = order["item_price"] if reason == "changed_mind" else min(amount, order["amount_paid"])
    return order, amount, reason


def generate(seed: int = SEED, n: int = N_EXAMPLES) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    names, weights = zip(*ARCHETYPES)
    used_ids: set[str] = set()
    records = []
    for i in range(1, n + 1):
        order_id = f"ORD-{rng.randint(20000, 69999)}"
        while order_id in used_ids:
            order_id = f"ORD-{rng.randint(20000, 69999)}"
        used_ids.add(order_id)
        archetype = rng.choices(names, weights=weights)[0]
        reason = rng.choice(REASONS)
        if archetype in ("no_photo", "exception", "carrier_confirmed", "changed_mind_shipping"):
            reason = "damaged_item"  # the mutation picks its own reason
        order, amount = _baseline(rng, order_id, reason)
        order, amount, reason = _mutate(rng, archetype, order, amount, reason)
        broken = policy_violations(order, amount, reason)
        if broken == ["rule8_manager_decides"]:
            label = rng.random() < 0.5          # the refund manager's recorded call
        else:
            label = not broken
        if rng.random() < LABEL_NOISE:
            label = not label                   # reviewer inconsistency
        records.append({"id": f"refund-{i:04d}", "context": refund_context(order, amount, reason), "label": label})
    return records


def render(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in records)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic refund calibration set.")
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
    positives = sum(r["label"] for r in records)
    print(f"wrote {len(records)} records to {args.output} ({positives} refundable, {len(records) - positives} not)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
