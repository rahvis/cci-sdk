# cli-sdk — Python SDK for Conformal Logit Inference

Conformal Logit Inference (CLI) turns the raw output of any LLM into
finite-sample statistical guarantees. This package contains:

- **`cli_sdk`** — sync and async clients for the hosted API, the seven
  guarantee-bearing query primitives, typed answers, calibration profiles,
  and drift monitors.
- **`cli_sdk.stats`** — the standalone statistics engine (conformal
  prediction, Venn-Abers calibration, e-values). It needs only NumPy and
  makes no network calls, so it runs inside air-gapped environments on
  precomputed score arrays.

- **`cli_sdk.local`** and **`cli_sdk.evidence`** — local mode: calibrate and
  evaluate the same queries in-process on your own model (OpenAI, Azure
  OpenAI, Anthropic, Gemini, or open weights such as Gemma on vLLM or
  SGLang), with calibration profiles stored as reviewable JSON files.
- **`cli_sdk.integrations`** — calibrated guardrails for agent tool calls in
  LangChain, LangGraph, Google ADK and Microsoft Agent Framework: allow,
  escalate to a human, or block, each with a stated guarantee.

Every guarantee is marginal or group-conditional over data exchangeable with
a stated calibration set. None is a promise about one decision in isolation.

## Install

```bash
pip install cli-sdk          # or: uv add cli-sdk
```

Optional extras for model providers and agent frameworks (Python 3.10+ for
the frameworks):

```bash
pip install "cli-sdk[openai]"            # OpenAI, Azure OpenAI, vLLM, SGLang evidence backends
pip install "cli-sdk[anthropic]"         # Claude evidence backend
pip install "cli-sdk[langchain,openai]"  # or [langgraph], [google-adk], [agent-framework], [all]
```

For development in this repository:

```bash
cd sdk/python
pip install -e ".[dev]"
pytest
```

## Authenticate

```bash
export CLI_API_KEY=sk_live_...
# optional, for a self-hosted deployment:
export CLI_BASE_URL=https://cli.internal.example.com/v1
```

A key is not required when `CLI_BASE_URL` points at `localhost`.

## Accessing the API

### Over HTTP

Every call is `POST https://api.cli.dev/v1/evaluate` with a bearer token:

```bash
curl https://api.cli.dev/v1/evaluate \
  -H "Authorization: Bearer $CLI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "context": {"ticket": "My payouts have been failing for 3 days."},
    "backend": {"provider": "openai", "model": "gpt-4.1-2025-04-14"},
    "queries": {
      "department": {
        "type": "set",
        "instructions": "Which team should handle this ticket?",
        "options": {"billing": null, "technical": null, "sales": null},
        "calibration_profile": "support-routing-v3",
        "alpha": 0.10
      }
    }
  }'
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/evaluate` | Evaluate a context against one or more queries |
| `POST /v1/calibration-profiles` | Create a calibration profile |
| `GET /v1/calibration-profiles/{name}` | Size, coverage interval, audit history |
| `POST /v1/calibration-profiles/{name}/examples` | Add labelled examples |
| `POST /v1/calibration-profiles/{name}/label-with-judge` | Judge-labelled pool plus a human sample, combined validly |
| `POST /v1/calibration-profiles/{name}/audit` | Audit against fresh labelled examples |
| `POST /v1/calibration-profiles/{name}/monitors` | Create an anytime-valid drift monitor |
| `GET /v1/calibration-profiles/{name}/monitors/{id}/alerts` | Poll monitor alerts |
| `GET /v1/backends` | Configured backends and their detected access level |

### With the SDK

```python
from cli_sdk import CLIClient, Set, Gate

with CLIClient() as client:
    result = client.evaluate(
        context={"ticket": "My payouts have been failing for 3 days."},
        backend={"provider": "openai", "model": "gpt-4.1-2025-04-14"},
        queries={
            "department": Set(
                instructions="Which team should handle this ticket?",
                options={"billing": None, "technical": None, "sales": None},
                calibration_profile="support-routing-v3",
                alpha=0.10,
            ),
            "route": Gate(
                instructions="Auto-route this ticket without human review?",
                calibration_profile="support-routing-v3",
                guarantee="fdr",
                target=0.05,
            ),
        },
    )

department = result.answers["department"]
print(department.set, department.guarantee.describe())

if result.answers["route"].approved:
    assign_queue(department.top)
else:
    send_to_human_triage(department.set)
```

The async client has the same interface, with `await` on each call (`monitors.poll` returns a list rather than a generator):

```python
from cli_sdk import AsyncCLIClient

async with AsyncCLIClient() as client:
    result = await client.evaluate(context=..., backend=..., queries=...)
```

## The seven primitives

| Primitive | Returns | Guarantee |
|---|---|---|
| `Belief` | Venn-Abers interval `[p0, p1]` for one statement | calibrated probability bracket |
| `Set` | a set of options | coverage >= 1 - alpha |
| `Interval` | a conformalized ordinal/continuous interval | coverage >= 1 - alpha |
| `Gate` | `auto_approve` / `escalate` / `abstain` | risk <= alpha, risk with probability 1 - delta, or batch FDR <= q |
| `Claim` | long-form output filtered to supported claims | P(all retained claims true) >= 1 - alpha |
| `Judge` | a verdict with an escalation cascade | human agreement >= 1 - alpha on accepted verdicts |
| `Route` | which backend in a cascade served the request | cost or accuracy bound |

Every answer carries `answer.guarantee`, a `Guarantee` with `type`,
`method`, `alpha`/`delta`/`target`, `calibration_profile`, `calibration_n`,
`coverage_ci`, and `last_audited`. `guarantee.is_heuristic` is `True` when an
answer carries no formal guarantee, for example because its profile is still
below the minimum size. Pass `strict_guarantees=True` to the client to raise
`InsufficientCalibrationError` instead of receiving heuristic answers.

## Typed responses

```python
from cli_sdk import EvaluateResponse, SetAnswer

class RoutingResponse(EvaluateResponse):
    department: SetAnswer

result = client.evaluate(..., response_model=RoutingResponse)
result.department  # a SetAnswer; a missing or mistyped answer raises at parse time
```

## Calibration profiles

```python
client.calibration_profiles.create(
    name="support-routing-v3",
    backend={"provider": "openai", "model": "gpt-4.1-2025-04-14"},
    method="APS",
    alpha=0.10,
    group_by="account_tier",       # optional Mondrian (per-group) calibration
)
client.calibration_profiles.add_examples(
    "support-routing-v3",
    examples=[{"context": {"ticket": "I was charged twice."}, "label": "billing"}],
)
profile = client.calibration_profiles.get("support-routing-v3")
profile.n, profile.minimum_n, profile.recommended_n, profile.realized_coverage_ci
```

| alpha | hard minimum n | recommended n |
|---|---|---|
| 0.20 | 4 | ~300 |
| 0.10 | 9 | ~1,000 |
| 0.05 | 19 | ~1,500 |
| 0.01 | 99 | ~2,500 |

Label-efficient calibration labels a pool with a judge and a random human
subset, combined so the result is valid regardless of judge quality:

```python
client.calibration_profiles.label_with_judge(
    "support-routing-v3",
    judge={"provider": "openai", "model": "gpt-4.1-2025-04-14"},
    unlabelled_examples=pool,
    human_labelled_sample_size=300,
)
```

## Drift monitoring

```python
client.calibration_profiles.monitors.create(
    "support-routing-v3", type="coverage", target=0.90,
    false_alarm_rate=0.05, labelled_sample_rate=0.02,
)
for alert in client.calibration_profiles.monitors.poll("support-routing-v3"):
    pause_auto_actioning()
```

Monitors are e-processes: the false-alarm rate holds no matter how often or
for how long you check them. `LocalMonitor` runs the same test in-process:

```python
from cli_sdk import LocalMonitor

monitor = LocalMonitor(type="coverage", target=0.90, false_alarm_rate=0.05)
for label, prediction_set in stream:
    if alert := monitor.update(label in prediction_set):
        page_oncall(alert)
```

## Backends

```python
from cli_sdk import OpenAIBackend, AnthropicBackend, VLLMBackend

OpenAIBackend(model="gpt-4.1-2025-04-14")                  # L1 on non-reasoning configs
AnthropicBackend(model="claude-sonnet-5", sample_count=20) # L0: sampling only
VLLMBackend(model="Qwen/Qwen3-8B", base_url="http://vllm:8000")  # up to L4
```

Plain dicts (`{"provider": "openai", "model": ...}`) work anywhere a backend
is accepted.

### Bring your own model

A `CustomBackend` runs your model locally and sends only the evidence
(probabilities, samples, scores) to CLI. Implement the methods your access
level supports:

```python
from cli_sdk import CLIClient, CustomBackend

class MyEngine(CustomBackend):
    access_level = "L1"

    def score_options(self, context, instructions, options):
        return my_model.option_probabilities(context, instructions, list(options))

    def sample(self, context, instructions, n):
        return [my_model.generate(context, instructions) for _ in range(n)]

client = CLIClient(backend=MyEngine())
```

## Local mode

`LocalCLIClient` runs the same queries fully in-process: it scores labelled
examples with a model you bring, stores the calibration profile as a JSON
file, and answers with the same typed answers and guarantee cards as the
hosted client. Nothing leaves your process except calls to your own model
provider.

```python
from cli_sdk import Gate, LocalCLIClient
from cli_sdk.evidence import OpenAIEvidenceBackend, vllm_backend

evidence = OpenAIEvidenceBackend("gpt-4.1-mini", api_key="YOUR_OPENAI_API_KEY")
# or open weights:  evidence = vllm_backend("google/gemma-4-12B-it", base_url="http://localhost:8000/v1")

client = LocalCLIClient(evidence, store=".cli_profiles")
refund = Gate(instructions="Is issuing this refund correct under the policy?",
              calibration_profile="refund-approvals-v1", guarantee="risk", target=0.05)
client.calibrate(refund, labelled_examples)   # [{"context": {...}, "label": True}, ...]
answer = client.evaluate(context=case, queries={"refund": refund}).answers["refund"]
print(answer.decision, answer.guarantee.describe())
```

Profiles are fingerprinted with the backend configuration and the query's
prompt: change either and answers fall back to their safe heuristic form
(escalate, every option, `[0, 1]`) until you recalibrate. See the
[local mode guide](https://docs.cli.dev/agents/local-mode).

## Agent frameworks

`ToolGuard` decides each proposed tool call from a calibrated answer and
maps it onto the framework's native human-in-the-loop mechanism:

```python
from cli_sdk.integrations import GuardRule, ToolGuard
from cli_sdk.integrations.langchain import cli_middleware

guard = ToolGuard(client, [GuardRule(tool="issue_refund", query=refund, context=refund_context)])
agent = create_agent(model, tools=[lookup_order, issue_refund],
                     middleware=cli_middleware(guard), checkpointer=InMemorySaver())
```

| Framework | Module | Hook | Human review |
|---|---|---|---|
| LangChain 1.x | `cli_sdk.integrations.langchain` | `wrap_tool_call` middleware | `HumanInTheLoopMiddleware` |
| LangGraph | `cli_sdk.integrations.langgraph` | guard node before `ToolNode` | `interrupt()` / `Command(resume=...)` |
| Google ADK 2.x | `cli_sdk.integrations.google_adk` | `before_tool_callback` or plugin | tool confirmation |
| Microsoft Agent Framework | `cli_sdk.integrations.agent_framework` | `FunctionMiddleware` | function approval requests |

Eight runnable examples in risk-sensitive domains (refunds, patient triage,
insurance claims, discharge summaries, AML holds, trial screening, credit
tiers, drug safety) are in `examples/agents/`. Each runs offline with
`--provider mock` and with your own keys for OpenAI, Azure, Anthropic,
Gemini, or Gemma on vLLM or SGLang. See the
[agent frameworks guide](https://docs.cli.dev/agents).

## Offline statistics engine

```python
from cli_sdk.stats.conformal import aps, crc
from cli_sdk.stats.venn_abers import ivap
from cli_sdk.stats.evalues import ebh, ppi, CoverageMonitor

q_hat = aps.calibrate(cal_probs, cal_labels, alpha=0.10)
sets = aps.predict(test_probs, q_hat)

p0, p1 = ivap.calibrate_and_predict(cal_scores, cal_labels, test_scores)

selected = ebh.select(e_values, q=0.10)
```

| Module | Methods |
|---|---|
| `stats.conformal` | LAC, APS, RAPS, CQR, CRC, RCPS, Learn-then-Test, Mondrian |
| `stats.venn_abers` | IVAP, CVAP, interval merging and width |
| `stats.evalues` | anytime-valid risk and coverage monitors, e-BH, prediction-powered inference |

## Command-line tool

```bash
cli calibration show support-routing-v3
cli calibration audit support-routing-v3 --examples fresh.jsonl --fail-below 0.88
cli calibration audit-local --covered outcomes.txt --target 0.90
```

`audit` exits with status 1 on failure, so it can block a deploy in CI.

## Errors

| Exception | When |
|---|---|
| `AuthenticationError` | 401: missing or invalid API key |
| `ValidationError` | 422: the request failed validation |
| `InsufficientCalibrationError` | a profile is too small for the guarantee (raised only with `strict_guarantees=True`) |
| `RateLimitError` | 429 after retries are exhausted |
| `BackendError` | 502 / 529: the model backend errored or reported a lower access level than required |
| `ConfigurationError` | client-side misconfiguration, raised before any request |

`429`, `502` and `529` are retried with full-jitter exponential backoff,
honoring `retry-after`. Configure with `CLIClient(retry=RetryConfig(max_attempts=5, base_delay=0.5))`.

## Package layout

```
src/cli_sdk/
  client/         sync + async clients, retry policy
  queries/        Belief, Set, Interval, Gate, Claim, Judge, Route
  answers/        typed answers, the Guarantee card, EvaluateResponse
  calibration/    profiles, examples, audits, label-efficient calibration
  monitoring/     hosted monitors, alerts, LocalMonitor
  backends/       one adapter per provider, plus CustomBackend
  stats/          offline engine: conformal/, venn_abers/, evalues/
  local/          LocalCLIClient: in-process calibration and evaluation
  evidence/       evidence backends: OpenAI, Azure, Anthropic, Gemini, vLLM, SGLang, mock
  integrations/   ToolGuard and adapters for LangChain, LangGraph, ADK, Agent Framework
  cli_tool.py     the `cli` command
examples/         runnable end-to-end examples (agents/: agent-framework examples)
tests/            engine and client tests
```
