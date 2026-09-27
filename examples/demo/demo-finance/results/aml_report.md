# Financial-crime compliance (AML): baseline vs. CCI-guarded — comparison report

**Synthetic demo data only. Not a real AML/compliance system.**

Held-out test set: 12 alerts, half designed to look routine while hiding a real pattern, half designed to look alarming while being legitimate — AML has a real cost in both directions.

## Headline numbers

| | Baseline (no CCI) | CCI-guarded |
|---|---|---|
| Decision accuracy | 92% (11/12) | n/a — CCI only decides hold vs. escalate, not a top-1 guess to score |
| Auto-actioned (held or cleared without a human) | 25% (3/12) | 0% (0/12) auto-held |
| Auto-actioned AND wrong | 0 | 0 |
| Genuinely-warranted alerts left for a human anyway | n/a | 6 |

## Why CCI escalated every case in this run

With a genuinely clean, deduplicated **n=40** calibration set, the Learn-then-Test procedure behind `guarantee="fdr"` found no confidence threshold it could clear at the 85%-confidence, 15%-target bar — so it escalated all 12 test cases rather than auto-hold anything it couldn't back with a real bound. That's the same fail-closed pattern the healthcare demo's `Claim` check showed on its own small calibration set: escalate rather than guess. It is a direct, honest consequence of demo-scale data (this method's own minimum-n floor is far higher than the CRC/risk Gate's), not a sign the mechanism doesn't work — the same procedure, run with production-scale calibration data (hundreds to low thousands of labelled alerts, per the product's own recommendation), would find real thresholds and start auto-deciding the clear-cut cases.

## What the numbers mean

The baseline auto-actioned 3 of 12 alerts based on a self-reported confidence crossing 0.8 — a number with no stated error-rate meaning. CCI auto-held nothing, since no case cleared the calibrated bound on this demo-scale calibration set (Learn-then-Test, a different algorithm than the CRC/risk Gate the lending and healthcare demos used — this is genuinely a second calibration method being exercised, not a repeat), and left 6 genuinely-warranted alerts for a human rather than clearing them — the conservative failure mode, not the dangerous one.

## Guarantee card actually returned (verbatim)

- Gate: With 85% confidence, at most 0.15 of auto-approved decisions are wrong (Learn-then-Test, fixed-sequence, n=40).

## Every case, side by side

| Case | Warranted? | Baseline pred (conf) | Baseline action | CCI decision |
|---|---|---|---|---|
| AT-0001 | False | clear (0.65) | escalated | escalate |
| AT-0002 | True | hold (0.65) | escalated | escalate |
| AT-0003 | True | hold (0.60) | escalated | escalate |
| AT-0004 | True | hold (0.60) | escalated | escalate |
| AT-0005 | True | hold (0.72) | escalated | escalate |
| AT-0006 | False | clear (0.82) | auto-actioned | escalate |
| AT-0007 | False | clear (0.83) | auto-actioned | escalate |
| AT-0008 | True | hold (0.70) | escalated | escalate |
| AT-0009 | False | clear (0.82) | auto-actioned | escalate |
| AT-0010 | True | hold (0.72) | escalated | escalate |
| AT-0011 | False | clear (0.68) | escalated | escalate |
| AT-0012 | False | hold (0.62) | escalated | escalate |
