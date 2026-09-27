# Finance demo: lending and AML, with and without CCI

**Synthetic demo data only. Not financial, credit, or compliance advice.**

Same architecture as `demo-healthcare`, covering the two domains named on
`/use-cases` this repo hadn't demonstrated yet. See `plan.md` for the
full design, acceptance criteria, and success criteria.

Two independent scenarios, each with a LangGraph baseline (Claude only,
self-reported confidence, hand-picked threshold) vs. a LangGraph pipeline
guarded by the real hosted CCI API (`https://cci.gitdate.ink/api/v1`,
your `CCI_API_KEY`):

- **Lending** (`baseline/lending_graph.py`, `cci/lending_graph.py`) —
  credit tiering with `Set(group_by="channel")` (Mondrian per-channel
  coverage) + `Gate(guarantee="risk")` for auto-approval.
- **AML** (`baseline/aml_graph.py`, `cci/aml_graph.py`) — account-hold
  decisions with `Gate(guarantee="fdr")` (Learn-then-Test — a different
  calibration algorithm than lending's or `demo-healthcare`'s Gate).

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and CCI_API_KEY from claude_api.md
```

## Run it

```bash
python data/generate_lending_cases.py
python data/generate_aml_cases.py
python cci/build_calibration.py           # registers all 3 profiles on the real hosted API

python baseline/run_lending.py            # -> results/baseline_lending.jsonl
python cci/run_lending.py                 # -> results/cci_lending.jsonl (first case is slow: real JIT calibration)
python compare_lending.py                 # -> results/lending_report.md

python baseline/run_aml.py                # -> results/baseline_aml.jsonl
python cci/run_aml.py                     # -> results/cci_aml.jsonl
python compare_aml.py                     # -> results/aml_report.md
```

Non-technical visual comparisons: `results/lending_comparison.html` and
`results/aml_comparison.html` — open directly in a browser.

## What we actually found

**Lending**: CCI's `Set` step, calibrated separately per channel
(Mondrian grouping), reached 92% candidate-set coverage vs. the
baseline's 58% top-1 accuracy, and — the more interesting result — one
case (an unemployed applicant, an unambiguous "decline") was
auto-finalized by CCI's calibrated Gate but sat unactioned by the
baseline, whose own self-reported confidence (0.75) fell just under its
arbitrary 0.8 cutoff. A calibrated bound isn't just more cautious than a
hand-picked number — it also acts decisively when the evidence actually
supports it.

**AML**: with a clean, deduplicated 40-example calibration set, the
Learn-then-Test procedure behind `guarantee="fdr"` couldn't clear its
85%-confidence/15%-target bar for any alert in this run, so it escalated
all 12 — the same fail-closed pattern seen throughout this demo series.
Meanwhile the baseline auto-actioned 3 alerts (all correct, by chance) on
a confidence number with no stated error-rate meaning. See
`results/aml_report.md` for why this is a demo-scale artifact of this
specific calibration method's minimum-data requirement, not a limitation
of the mechanism.

An earlier run of this pipeline did surface one real auto-hold miss
before a data-integrity bug (duplicated calibration examples from a
re-run) was found and fixed — worth knowing about since it's a good,
honest illustration of a probabilistic guarantee (a single miss in 4
trials is not a violation of an 85%-confidence bound; see the git
history of `compare_aml.py` and `results/aml_report.md` for that
analysis, preserved because it's instructive even though the clean run
superseded it).

## Re-running with a fresh calibration profile

Same convention as `demo-healthcare`: bump the profile name in
`cci/queries.py` (`-v1` → `-v2`) whenever a query's `instructions` change,
or when you need a guaranteed-clean recalibration (e.g. after
accidentally re-posting the same examples twice, as happened once during
this demo's own construction).
