"""Generate ``insurance_claims.jsonl``: synthetic claim-payment decisions labelled by a written guideline.

All data is synthetic. No real policyholder, claim or insurer is represented,
and nothing here is claims-handling or legal advice.

Each line is one historical decision::

    {"id": "C-1001", "context": {...}, "label": true}

``context`` is exactly what the payment guard evaluates at run time: the
claim file, the payment the agent proposed, and the guideline text, built by
``claim_context`` below. The LangGraph example imports ``claim_context`` from
this module, so calibration examples and live requests are rendered by the
same function (the calibrated guarantee only holds for the scoring function
it was calibrated with).

``label`` is True when paying the proposed amount is correct under the
guideline, which is written out in ``GUIDELINE``:

1. Policy period: the date of loss falls between policy inception and expiry.
2. Covered peril: sudden and accidental losses are covered; gradual leaks,
   mold, wear and tear, earth movement and flood are excluded. Sewer backup
   needs the "water backup" endorsement, flood the "flood" endorsement.
3. No referral trigger: a loss within 30 days of inception, two or more
   fraud indicators, or three or more prior claims in 36 months goes to the
   Special Investigations Unit before any payment.
4. Documentation: proof of loss, photos and a repair estimate or invoice;
   theft also needs a police report.
5. Amount: payable = min(assessed loss, sub-limit when one applies,
   coverage limit) - deductible, never below 0; the proposed amount must
   equal it within 1.00.

About 7% of claims carry an adjuster note that makes the case a genuine
judgment call (an unsigned estimate, low-resolution photos); their labels
follow the historical adjuster's call, which the note alone does not
determine. About 2% of labels are flipped to model historical labelling
errors. Both keep the calibrated thresholds realistic.

Run ``python generate_insurance_claims.py`` (stdlib only, seeded) to rewrite
``insurance_claims.jsonl`` next to this file; the output is byte-identical
on every run.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

SEED = 20260924
N_EXAMPLES = 320
OUTPUT = Path(__file__).resolve().parent / "insurance_claims.jsonl"

GUIDELINE = (
    "Claims payment guideline CPG-7 (synthetic, for demonstration only). Paying the proposed "
    "amount is correct only when every rule below holds. "
    "1. Policy period: the date of loss is on or after policy_inception and on or before "
    "policy_expiry. "
    "2. Covered peril: sudden and accidental direct physical loss is covered. Excluded: gradual "
    "leaks, mold, wear and tear, earth movement and flood. Sewer backup is covered only with the "
    "'water backup' endorsement, and flood only with the 'flood' endorsement. "
    "3. Referral: a loss within 30 days after policy_inception, two or more fraud_indicators, or "
    "three or more prior_claims_36m must be referred to the Special Investigations Unit before "
    "any payment is made. "
    "4. Documentation: 'proof of loss', 'photos' and 'repair estimate or invoice' are required; "
    "theft also requires 'police report'. "
    "5. Amount: payable = min(assessed_loss, sub_limit amount when a sub_limit applies, "
    "coverage_limit) - deductible, and never below 0. The proposed amount must equal the payable "
    "amount within 1.00."
)

REQUIRED_DOCUMENTS = ("proof of loss", "photos", "repair estimate or invoice")
THEFT_DOCUMENT = "police report"
JEWELRY_SUB_LIMIT = 1500.0
REFERRAL_DAYS = 30

# peril -> assessed-loss range, flags (theft, jewelry sub-limit, required endorsement, excluded), description
PERILS: dict[str, dict[str, Any]] = {
    "burst pipe (sudden water discharge)": {
        "range": (1200, 18000),
        "description": "Supply line under the kitchen sink burst; water damage to flooring and cabinets."},
    "kitchen fire": {
        "range": (2000, 42000),
        "description": "Grease fire on the stove; smoke and fire damage to the kitchen."},
    "wind damage to roof": {
        "range": (1500, 22000),
        "description": "Storm winds lifted shingles; roof decking and attic insulation damaged."},
    "hail damage to roof": {
        "range": (3000, 26000),
        "description": "Hailstorm dented the roof and gutters; inspection found cracked shingles."},
    "fallen tree on detached garage": {
        "range": (2500, 15000),
        "description": "A neighbor's oak fell on the detached garage during a storm."},
    "lightning surge to electronics": {
        "range": (600, 6000),
        "description": "Lightning strike nearby; surge damaged the television and a desktop computer."},
    "theft of electronics": {
        "range": (800, 7500), "theft": True,
        "description": "Laptop, tablet and camera taken during a daytime break-in."},
    "theft of jewelry": {
        "range": (900, 9000), "theft": True, "jewelry": True,
        "description": "Rings and a watch taken from the bedroom during a break-in."},
    "sewer backup": {
        "range": (1500, 14000), "endorsement": "water backup",
        "description": "Municipal sewer backed up into the finished basement."},
    "flood (surface water)": {
        "range": (4000, 60000), "endorsement": "flood",
        "description": "River overflow put 30 cm of surface water on the ground floor."},
    "gradual leak under sink (over several months)": {
        "range": (1000, 9000), "excluded": True,
        "description": "Slow drip from a corroded trap; subfloor rot discovered during a remodel."},
    "mold remediation": {
        "range": (1500, 12000), "excluded": True,
        "description": "Mold found behind bathroom tiles; remediation contractor engaged."},
    "wear and tear (aged roof)": {
        "range": (4000, 20000), "excluded": True,
        "description": "Roof is 27 years old; shingles brittle and curling, leaks in two rooms."},
    "earth movement (foundation settling)": {
        "range": (5000, 40000), "excluded": True,
        "description": "Cracks in the foundation and drywall from soil settling."},
}
PERIL_WEIGHTS = {
    "burst pipe (sudden water discharge)": 14, "kitchen fire": 8, "wind damage to roof": 12,
    "hail damage to roof": 9, "fallen tree on detached garage": 6, "lightning surge to electronics": 6,
    "theft of electronics": 9, "theft of jewelry": 9, "sewer backup": 7, "flood (surface water)": 3,
    "gradual leak under sink (over several months)": 4, "mold remediation": 3,
    "wear and tear (aged roof)": 3, "earth movement (foundation settling)": 2,
}

FRAUD_INDICATORS = (
    "loss reported more than 45 days after the date of loss",
    "receipts dated after the date of loss",
    "policy limits increased shortly before the loss",
    "inconsistent statements about the time of loss",
    "same contractor on three unrelated claims this year",
)
PLAIN_NOTES = (
    "", "", "", "Field inspection completed; damage consistent with the reported cause.",
    "Desk review; estimate reviewed against regional pricing.", "Insured cooperative; statement taken by phone.",
)
AMBIGUOUS_NOTES = (
    "Repair estimate is unsigned; the contractor confirmed the figure by phone.",
    "Photos are low resolution but appear to show the damaged area.",
    "Insured gives an approximate loss date (the same week); neighbors corroborate the storm.",
    "Proof of loss arrived by email; the signed original is still pending.",
)


# ---------------------------------------------------------------------------
# the shared context builder (used for calibration and by the live guard)
# ---------------------------------------------------------------------------

CLAIM_FIELDS = (
    "claim_id", "policy_form", "peril", "loss_description", "loss_date", "reported_date",
    "policy_inception", "policy_expiry", "endorsements", "coverage_limit", "sub_limit",
    "deductible", "assessed_loss", "prior_claims_36m", "documents", "fraud_indicators",
    "adjuster_note",
)


def claim_context(claim: dict[str, Any], amount: float) -> dict[str, Any]:
    """The exact context the payment guard evaluates: claim file, proposed payment, guideline."""
    return {
        "claim": {field: claim.get(field) for field in CLAIM_FIELDS},
        "proposed_payment": {"claim_id": claim["claim_id"], "amount": round(float(amount), 2)},
        "guideline": GUIDELINE,
    }


# ---------------------------------------------------------------------------
# the written guideline as code (labels only; never shown to the evidence model)
# ---------------------------------------------------------------------------


def payable_amount(claim: dict[str, Any]) -> float:
    cap = min(float(claim["assessed_loss"]), float(claim["coverage_limit"]))
    if claim.get("sub_limit"):
        cap = min(cap, float(claim["sub_limit"]["amount"]))
    return round(max(0.0, cap - float(claim["deductible"])), 2)


def guideline_violations(claim: dict[str, Any], amount: float) -> list[str]:
    """Every rule of CPG-7 the proposed payment breaks (empty list: paying it is correct)."""
    problems = []
    loss = date.fromisoformat(claim["loss_date"])
    inception = date.fromisoformat(claim["policy_inception"])
    expiry = date.fromisoformat(claim["policy_expiry"])
    if not (inception <= loss <= expiry):
        problems.append("loss outside the policy period")
    peril = PERILS[claim["peril"]]
    if peril.get("excluded"):
        problems.append("excluded peril")
    if peril.get("endorsement") and peril["endorsement"] not in claim["endorsements"]:
        problems.append(f"peril needs the '{peril['endorsement']}' endorsement")
    if 0 <= (loss - inception).days < REFERRAL_DAYS:
        problems.append("loss within 30 days of inception: refer to SIU")
    if len(claim["fraud_indicators"]) >= 2:
        problems.append("two or more fraud indicators: refer to SIU")
    if claim["prior_claims_36m"] >= 3:
        problems.append("three or more prior claims: refer to SIU")
    required = list(REQUIRED_DOCUMENTS) + ([THEFT_DOCUMENT] if peril.get("theft") else [])
    missing = [doc for doc in required if doc not in claim["documents"]]
    if missing:
        problems.append("missing documents: " + ", ".join(missing))
    if abs(float(amount) - payable_amount(claim)) > 1.0:
        problems.append(f"amount {amount:.2f} differs from payable {payable_amount(claim):.2f}")
    return problems


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _money(rng: random.Random, lo: float, hi: float) -> float:
    return float(round(rng.uniform(lo, hi) / 10.0) * 10)


def _claim(rng: random.Random, index: int) -> dict[str, Any]:
    peril_name = rng.choices(list(PERIL_WEIGHTS), weights=list(PERIL_WEIGHTS.values()))[0]
    peril = PERILS[peril_name]
    inception = date(2023, 6, 1) + timedelta(days=rng.randrange(0, 600))
    expiry = inception + timedelta(days=365)
    roll = rng.random()
    if roll < 0.08:
        loss = inception + timedelta(days=rng.randrange(1, REFERRAL_DAYS))       # early loss
    elif roll < 0.13:
        loss = expiry + timedelta(days=rng.randrange(5, 90))                     # policy lapsed
    else:
        loss = inception + timedelta(days=rng.randrange(REFERRAL_DAYS, 365))
    endorsements = sorted(e for e, p in (("water backup", 0.45), ("flood", 0.10)) if rng.random() < p)
    if peril.get("endorsement") and rng.random() < 0.5 and peril["endorsement"] not in endorsements:
        endorsements = sorted(endorsements + [peril["endorsement"]])
    documents = list(REQUIRED_DOCUMENTS) + ([THEFT_DOCUMENT] if peril.get("theft") else [])
    if rng.random() < 0.08:
        documents.remove(rng.choice(documents))
    indicators: list[str] = []
    roll = rng.random()
    if roll < 0.06:
        indicators = rng.sample(FRAUD_INDICATORS, 2)
    elif roll < 0.18:
        indicators = [rng.choice(FRAUD_INDICATORS)]
    prior = rng.choices([0, 1, 2, 3, 4], weights=[60, 22, 12, 4, 2])[0]
    report_lag = rng.randrange(0, 12) if "loss reported more than 45 days" not in " ".join(indicators) \
        else rng.randrange(46, 90)
    ambiguous = rng.random() < 0.07
    note = rng.choice(AMBIGUOUS_NOTES) if ambiguous else rng.choice(PLAIN_NOTES)
    return {
        "claim_id": f"C-{1001 + index}",
        "policy_form": "HO-3 homeowners (synthetic)",
        "peril": peril_name,
        "loss_description": peril["description"],
        "loss_date": loss.isoformat(),
        "reported_date": (loss + timedelta(days=report_lag)).isoformat(),
        "policy_inception": inception.isoformat(),
        "policy_expiry": expiry.isoformat(),
        "endorsements": endorsements,
        "coverage_limit": float(rng.choice([250000, 350000, 500000])),
        "sub_limit": ({"applies_to": "jewelry and watches (theft)", "amount": JEWELRY_SUB_LIMIT}
                      if peril.get("jewelry") else None),
        "deductible": float(rng.choice([500, 1000, 1000, 2500])),
        "assessed_loss": _money(rng, *peril["range"]),
        "prior_claims_36m": prior,
        "documents": documents,
        "fraud_indicators": indicators,
        "adjuster_note": note,
        "_ambiguous": ambiguous,
    }


def _proposed_amount(rng: random.Random, claim: dict[str, Any]) -> float:
    """What the (fallible) claims agent proposed to pay."""
    payable = payable_amount(claim)
    roll = rng.random()
    if claim.get("sub_limit") and roll < 0.40:
        return round(max(0.0, claim["assessed_loss"] - claim["deductible"]), 2)   # ignored the sub-limit
    if roll < 0.70:
        return payable
    if roll < 0.82:
        return round(float(claim["assessed_loss"]), 2)                           # forgot the deductible
    if roll < 0.92:
        return round(payable * rng.uniform(0.85, 1.15) / 10.0) * 10.0            # arithmetic slip
    return payable


def build_examples(seed: int = SEED, n: int = N_EXAMPLES) -> list[dict[str, Any]]:
    """The labelled calibration examples, deterministically."""
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        claim = _claim(rng, i)
        amount = _proposed_amount(rng, claim)
        label = not guideline_violations(claim, amount)
        if claim.pop("_ambiguous"):
            label = rng.random() < 0.5            # the historical adjuster's judgment call
        if rng.random() < 0.02:
            label = not label                     # historical labelling error
        examples.append({"id": claim["claim_id"], "context": claim_context(claim, amount), "label": label})
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
    positives = sum(1 for r in rows if r["label"])
    print(f"wrote {len(rows)} examples to {out.name}: {positives} correct payments, "
          f"{len(rows) - positives} incorrect ({positives / len(rows):.0%} positive)")
