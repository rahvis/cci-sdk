# Synthetic calibration datasets

> **Every record in this directory is synthetic.** No real patient,
> customer, claimant, applicant, investigator, account, product, clinician
> or institution is represented. Every policy, protocol, guideline and
> grading guide is invented for the examples, is simplified, and is not a
> clinical, underwriting, claims-handling, pharmacovigilance or regulatory
> standard. Nothing here is medical, financial, credit, legal or regulatory
> advice, and the data must not be used to make decisions about real
> people.

Each dataset is written by a deterministic generator next to it
(`generate_<dataset>.py`: seeded, standard library only) that labels every
record from an explicit written rule set, and then adds a small, stated
fraction of genuinely ambiguous or noisy records, the way real labelled
exports have them. Without that, the calibrated thresholds would be
unrealistically clean. The committed `.jsonl` files are the generators'
output; re-running a generator reproduces its file exactly.

## Format

One JSON object per line:

```json
{"id": "...", "context": {...}, "label": ...}
```

- `context` is **exactly** what the calibrated guard evaluates. Each
  generator defines the function that builds it (the "context builder"
  below), and the example imports that same function for its
  `GuardRule(context=...)`. Calibration contexts and live contexts therefore
  have the same keys, the same policy text and the same rendering, which is
  what the guarantee requires: it holds only for the scoring function it was
  calibrated with.
- `label` is the ground truth for the example's primitive: a boolean for a
  `Gate` or `Belief`, an option key for a `Set`, a level for an `Interval`,
  and a list of `{"text": ..., "supported": ...}` for a `Claim`.
- `id` identifies the record. The examples' mock evidence models see only
  the `context`, like a real evidence model: they read the case's features
  with deterministic noise keyed on the case identifier inside it, and they
  never read the label.

## Datasets

| File | Generator | Context builder | Used by | Records | Label |
| --- | --- | --- | --- | --- | --- |
| `refund_requests.jsonl` | `generate_refund_requests.py` | `refund_context(order, amount, reason)` | `langchain/finance_refund_agent.py` | 320 | `true` if issuing the proposed refund is correct under the policy (175 true, 145 false) |
| `triage_messages.jsonl` | `generate_triage_messages.py` | `triage_context(message)` | `langchain/healthcare_patient_triage.py` | 320 | triage level: `self_care` 84, `routine_appointment` 125, `urgent_care` 85, `emergency` 26 |
| `insurance_claims.jsonl` | `generate_insurance_claims.py` | `claim_context(claim, amount)` | `langgraph/insurance_claims_graph.py` | 320 | `true` if paying the proposed amount is correct under the guideline (154 true, 166 false) |
| `discharge_claims.jsonl` | `generate_discharge_claims.py` | `summary_context(chart, draft)` | `langgraph/clinical_summary_graph.py` | 240 drafts, 1,537 claims | per claim, `supported` true or false (156 unsupported claims, in 128 drafts) |
| `aml_alerts.jsonl` | `generate_aml_alerts.py` | `alert_context(alert)` | `google_adk/aml_account_hold_agent.py` | 360 | `true` if a temporary account hold is warranted (191 true, 169 false) |
| `trial_screening.jsonl` | `generate_trial_screening.py` | `screening_context(record)` | `google_adk/clinical_trial_screening_agent.py` | 320 | `true` if confirmed eligible at the screening visit (176 true, 144 false) |
| `credit_applications.jsonl` | `generate_credit_applications.py` | `application_context(application)` | `agent_framework/credit_underwriting_agent.py` | 320 | risk tier `A` 66, `B` 80, `C` 79, `D` 52, `E` 43 |
| `adverse_events.jsonl` | `generate_adverse_events.py` | `case_context(case)` | `agent_framework/pharmacovigilance_agent.py` | 300 | seriousness grade `grade_1` 106, `grade_2` 62, `grade_3` 90, `grade_4` 30, `grade_5` 12 |

The sizes (240 to 360) are chosen so the examples calibrate in seconds with
the mock, and in a few hundred requests with a real log-probability
evidence model (see the [examples README](../README.md#what-calibrating-with-a-real-model-costs)).
They are enough for each example's `alpha` or target, and below the size
recommended for a stable production guarantee (about 1,000 examples at
`alpha=0.10`, 1,500 at 0.05).

## refund_requests

Card and e-commerce refund decisions for the LangChain refunds agent.

- **Context**: `policy` (the written refund policy, `REFUND_POLICY`),
  `order` (the order record from the order system: category, final-sale
  flag, item price, shipping fee, amount paid, payment method, days since
  purchase and delivery, delivery status, customer tier, photo on file,
  amount already refunded, open fraud flag, open chargeback) and
  `proposed_refund` (`amount`, `reason`).
- **Label**: `true` when issuing the proposed refund violates no rule of
  the policy (`policy_violations()` applies it rule by rule).
- **Ambiguity and noise**: about 5% are exception requests (plus-tier
  customers up to 15 days past their window) that the policy leaves to a
  refund manager, labelled with the manager's recorded decision, which the
  record cannot settle; 2% of labels are flipped as reviewer inconsistency.
- **Regenerate**: `python examples/agents/data/generate_refund_requests.py`
  (`--check` exits 1 if the committed file differs from a fresh
  generation).

## triage_messages

Patient-portal messages for the LangChain triage agent.

- **Context**: `protocol` (the written triage protocol,
  `TRIAGE_PROTOCOL`) and `message` (message id, age band, complaint,
  duration in days, self-rated severity, temperature, reported red flags
  and urgent signs, and the patient's free text).
- **Label**: one of `self_care`, `routine_appointment`, `urgent_care`,
  `emergency`, from `protocol_level()`: the first protocol rule that
  matches, checked in order (any red flag is `emergency`).
- **Ambiguity and noise**: about 4% are vague messages (complaint
  `unclear`) that the protocol hands to a nurse, labelled with the nurse's
  recorded call; 1.5% of labels move one level up or down as reviewer
  inconsistency.
- **Regenerate**: `python examples/agents/data/generate_triage_messages.py`
  (`--check` as above).
- The protocol is a simplified teaching example for adult patients. It is
  not clinical guidance and must not be used to triage real patients.

## insurance_claims

Historical property-claim payment decisions for the LangGraph claims graph.

- **Context**: `claim` (the claim file: policy form, inception and expiry,
  loss and report dates, peril, loss description, assessed loss, coverage
  limit, sub-limit, deductible, endorsements, documents, fraud indicators,
  prior claims in 36 months, adjuster note), `guideline` (the written
  claims guideline CPG-7, `GUIDELINE`) and `proposed_payment` (`claim_id`,
  `amount`).
- **Label**: `true` when paying the proposed amount is correct under the
  guideline: the loss is in the policy period, the peril is covered (with
  the needed endorsement), there is no referral trigger (a loss within 30
  days of inception, two or more fraud indicators, or three or more prior
  claims), the documentation is complete (a police report for theft), and
  the amount equals min(assessed loss, sub-limit, coverage limit) minus the
  deductible, within 1.00.
- **Ambiguity and noise**: about 7% carry an adjuster note that makes the
  case a judgment call (an unsigned estimate, low-resolution photos),
  labelled with the historical adjuster's call; about 2% of labels are
  flipped as historical labelling errors.
- **Regenerate**: `python examples/agents/data/generate_insurance_claims.py`.

## discharge_claims

Synthetic discharge charts with machine-written, patient-friendly draft
summaries, for the LangGraph claim-verification node.

- **Context**: `chart` (encounter id, discharge date, principal and
  secondary diagnoses, medications, follow-up, pending results,
  instructions, return precautions) and `answer` (the draft summary, one
  sentence per claim).
- **Label**: one `{"text", "supported"}` per claim, from the written
  labelling protocol in the generator: a claim is supported when the chart
  states it, or restates a chart entry in plain language without changing
  any drug, dose, frequency, duration, status, appointment, date, test or
  symptom. A claim that adds or changes a fact (a wrong dose, an invented
  dose change, "stop" for a continued drug, a wrong follow-up date, an
  invented pending test or diagnosis) is unsupported.
- **Ambiguity and noise**: about half of the drafts contain one or two
  unsupported claims; some supported claims are paraphrases that share few
  words with the chart; about 1% of claim labels are flipped as annotator
  disagreement.
- **Regenerate**: `python examples/agents/data/generate_discharge_claims.py`.

## aml_alerts

Anti-money-laundering alerts with investigator hold decisions, for the
Google ADK investigator.

- **Context**: `guideline` (the fictional investigation guideline
  SYN-AML-IG-7, as a list of lines) and `alert` (alert and account ids,
  the monitoring rule that fired, a customer profile, 30-day activity,
  documentation on file, prior alerts in 12 months).
- **Label**: `true` when a temporary account hold is warranted under the
  guideline (`hold_warranted()`), as decided by an investigator looking at
  the account's full transaction history.
- **Ambiguity and noise**: the alert summary in the context is a slightly
  lossy view of that history (a deposit counted in the 8,000 to 9,999 USD
  band, a percentage or a wire total can differ a little), so alerts that
  hinge on one near-threshold pattern are genuinely ambiguous from the
  summary alone; about 1.5% of decisions are flipped as reviewer
  disagreement.
- **Regenerate**: `python examples/agents/data/generate_aml_alerts.py`
  (`--stats` prints the label balance by red-flag pattern; `--count`,
  `--seed` and `--out` change the output).

## trial_screening

Pre-screening records for a fictional trial protocol, SYN-CKD-201 (adults
with type 2 diabetes and stage 3 chronic kidney disease), for the Google
ADK pre-screening assistant.

- **Context**: `protocol` (the fictional protocol's inclusion and
  exclusion criteria, as a list of lines) and `patient` (age, sex,
  diagnoses, most recent labs, medications, dialysis or kidney transplant,
  recent interventional study, pregnancy, breastfeeding or planning).
- **Label**: `true` when the patient was confirmed eligible at the
  screening visit: every inclusion criterion met and no exclusion.
- **Ambiguity and noise**: the visit re-measures eGFR and HbA1c, and the
  record only holds the most recent values on file, so patients near a
  cutoff (an eGFR of 59, an HbA1c of 10.4) may or may not qualify; about
  1.5% of patients turn out to have an exclusion that was not in the record.
- **Regenerate**: `python examples/agents/data/generate_trial_screening.py`
  (`--stats`, `--count`, `--seed` and `--out` as for `aml_alerts`).

## credit_applications

Consumer-loan application files for the Agent Framework underwriting
assistant.

- **Context**: `application_id`, `channel` (`branch` 140, `online` 140,
  `broker` 40), `credit_score_band`, `debt_to_income`,
  `payment_history_24_months`, `credit_file` (thin or established) and
  `income_verification`.
- **Label**: the tier from `A` (lowest risk) to `E` (highest) assigned by
  the synthetic tiering policy (`TIERING_POLICY`): points for score band,
  debt-to-income and payment history, plus points for a thin file or
  unverified income, mapped to tiers, with overrides (a thin file is never
  tier A; stated income is never better than C; a collection or
  charge-off is never better than D). The channel never changes the tier.
- **Ambiguity and noise**: about 4% of files carry a documented
  underwriter override that moved the recorded tier by one step for
  reasons not in the file.
- **Fairness**: features exclude protected characteristics (age, sex,
  race, ethnicity, religion, national origin, marital status) and obvious
  proxies such as ZIP code. The channel is recorded only so calibration
  can be checked per channel (Mondrian); the broker channel is deliberately
  small, so its per-channel guarantee is visibly less certain. This is not
  a fair-lending analysis.
- **Regenerate**: `python examples/agents/data/generate_credit_applications.py`.

## adverse_events

Adverse-event case reports for a fictitious product, CLX-101, for the
Agent Framework drug-safety assistant.

- **Context**: `case_id`, `suspect_product`, `event_term`,
  `onset_days_after_dose`, `hospitalization`, `treatment`, `lab_changes`,
  `daily_activities`, `outcome` and a short `narrative`.
- **Label**: the grade an assessor recorded by following the synthetic
  grading guide (`GRADING_GUIDE`): the highest grade whose criteria any
  finding meets, from `grade_1` (mild) to `grade_5` (fatal). In this
  synthetic workflow, grades 3 to 5 qualify for an expedited report.
- **Ambiguity and noise**: the guide leaves observation stays under 24
  hours to the assessor (grade 3 when medically necessary, grade 2 when
  precautionary), and the narrative only partly settles them; about 3% of
  cases carry a one-grade coding difference (never into or out of
  `grade_5`), as double-coded safety data does.
- **Regenerate**: `python examples/agents/data/generate_adverse_events.py`.
- In real pharmacovigilance, seriousness and severity are different
  concepts, and expedited reporting also depends on expectedness and
  causality. This guide is loosely modelled on five-level severity scales
  and is not a regulatory standard.

## Using your own data

Replace a bundled file with a random sample of your own labelled cases in
the same format, with every `context` built by the example's context
builder (adapted to your records), and recalibrate with `--recalibrate`.
Label the cases the way the decision should have been made under your own
written policy, by qualified reviewers, and keep a held-out part for
`client.audit(...)`. See the [examples README](../README.md#bring-your-own-labelled-data).
