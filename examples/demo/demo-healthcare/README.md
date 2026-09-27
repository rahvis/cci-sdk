# Patient-portal triage demo: with and without CCI

**Synthetic demo data only. Not medical advice, not a medical device.**

Two LangGraph pipelines for the same task — deciding whether an AI can
auto-reply to a patient's symptom message, or must send it to a human —
built against the same synthetic dataset and the same held-out test
cases, so the difference is measurable. See `plan.md` for the full
design, acceptance criteria, and success criteria.

- `baseline/` — Claude + LangGraph only. Self-reported confidence,
  hand-picked 0.8 threshold, no check on the reply text.
- `cci/` — the same shape of graph, but urgency routing goes through the
  real hosted CCI API (`https://cci.gitdate.ink/api/v1`) using `Set` +
  `Gate`, and the drafted reply is verified with `Claim` before it can be
  sent.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and CCI_API_KEY from claude_api.md
```

## Run it

```bash
python data/generate_patient_cases.py   # writes data/patient_cases.jsonl, data/claim_calibration.jsonl
python cci/build_calibration.py          # registers 3 calibration profiles on the real hosted API (fast, no model calls yet)
python baseline/run.py                   # -> results/baseline.jsonl
python cci/run.py                        # -> results/cci.jsonl (first case is slow: real just-in-time calibration; rest are fast)
python compare.py                        # prints + writes results/report.md
```

## What we actually found

See `results/report.md` for the full run this repo currently has
committed results from. Headline: the baseline auto-sent 8 of 16 replies
in that run — including all 4 emergency-severity cases — purely because
its own self-reported confidence cleared an arbitrary 0.8 cutoff, a
number with no stated error-rate meaning. CCI's guarded pipeline never
auto-sent anything above low-acuity (a categorical policy, not a
probability) and, for the low-acuity cases, its `Claim` check caught at
least one unsupported statement in every drafted reply on this
demo-scale (~48-example) calibration set, escalating all of them to a
human rather than guessing. That's the intended fail-closed behavior of
a properly calibrated system running on realistically small data — see
`results/report.md`'s "What the numbers mean" section for the full
explanation, including what changes with more calibration data.

## Re-running with a fresh calibration profile

If you change a query's `instructions`/`options` in `cci/queries.py`,
either bump the profile name (the pattern this repo already follows,
`-v1` → `-v2` → `-v3`) or just re-run `cci/build_calibration.py` — the
hosted API recalibrates automatically the first time `cci/run.py`
evaluates a query whose fingerprint no longer matches what's cached, so
you never need to manually trigger recalibration.
