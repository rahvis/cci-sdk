# Lending: baseline vs. CCI-guarded — comparison report

**Synthetic demo data only. Not financial or credit advice.**

Held-out test set: 12 applications across 3 channels (online/branch/partner), same Claude model in both arms.

## Headline numbers

| | Baseline (no CCI) | CCI-guarded |
|---|---|---|
| Tier accuracy (top guess / candidate-set coverage) | 58% (7/12) | 92% (11/12) |
| Candidate set was a single tier | n/a (always gives one guess) | 100% (12/12) |
| Auto-finalized (approved *or* auto-declined) | 25% (3/12) | 33% (4/12) |
| Auto-finalized AND wrong | 0 | 0 |

## Per-channel coverage — the point of Mondrian grouping

Coverage audited **separately per channel**, not blended into one overall number:

- **branch**: 4/4 held-out cases covered by the calibrated set
- **online**: 3/4 held-out cases covered by the calibrated set
- **partner**: 4/4 held-out cases covered by the calibrated set

This is the exact mechanism the product's own use-cases page names for lending: *"tier coverage is calibrated separately for each application channel, so a strong overall number cannot hide a weaker channel."* With only 20 calibration examples per channel here, per-channel coverage is imprecise (small-sample noise, not a flaw in the mechanism) — production use would calibrate on far more per-channel data.

## What the numbers mean

Both pipelines auto-finalized 0 wrong decisions in this run (0 baseline, 0 CCI) — but for different reasons. The baseline's decision to act is a hand-picked 0.8 cutoff on a number the model made up about itself. CCI's is a calibrated risk bound (CRC): "Expected rate of decisions that are auto-approved and wrong is at most 0.15 (conformal risk control, n=60)." One case makes the difference concrete: an unemployed applicant's file was a clear 'decline' by every measure. CCI's Set correctly narrowed to a singleton and its Gate auto-finalized the decline with a stated bound behind it. The baseline predicted the same correct tier but only reported 0.75 confidence — just under its own 0.8 bar — and left a clear-cut case sitting in a queue for no defensible reason. Confidence-based automation is exactly as arbitrary when it's *too* cautious as when it's *not cautious enough*; a calibrated bound acts exactly when the evidence supports it, no more and no less.

## Guarantee cards actually returned (verbatim)

- Set: Contains the correct answer at least 80% of the time on profile 'finance-lending-tier-v2' (n=20).
- Gate: Expected rate of decisions that are auto-approved and wrong is at most 0.15 (conformal risk control, n=60).

## Every case, side by side

| Case | Channel | True tier | Baseline pred (conf) | Baseline action | CCI set | CCI action |
|---|---|---|---|---|---|---|
| LT-0001 | partner | prime | prime (0.85) | auto-finalized | ['prime'] | auto-finalized |
| LT-0002 | online | subprime | near_prime (0.68) | escalated | ['subprime'] | escalated |
| LT-0003 | branch | subprime | subprime (0.60) | escalated | ['subprime'] | escalated |
| LT-0004 | branch | subprime | near_prime (0.70) | escalated | ['subprime'] | escalated |
| LT-0005 | partner | decline | subprime (0.60) | escalated | ['decline'] | escalated |
| LT-0006 | partner | subprime | near_prime (0.68) | escalated | ['subprime'] | escalated |
| LT-0007 | branch | prime | prime (0.85) | auto-finalized | ['prime'] | auto-finalized |
| LT-0008 | online | subprime | subprime (0.62) | escalated | ['subprime'] | escalated |
| LT-0009 | online | near_prime | subprime (0.60) | escalated | ['subprime'] | escalated |
| LT-0010 | partner | subprime | subprime (0.60) | escalated | ['subprime'] | escalated |
| LT-0011 | online | decline | decline (0.75) | escalated | ['decline'] | auto-finalized |
| LT-0012 | branch | prime | prime (0.90) | auto-finalized | ['prime'] | auto-finalized |
