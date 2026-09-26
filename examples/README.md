# Examples

Runnable end-to-end examples for the Conformal Logit Inference (CLI) Python
SDK. Each hosted example follows one walkthrough from the PRD (section 11)
and runs fully offline against a mock of the hosted API. The offline
calibration example uses the real statistics engine with no network at all.

| File | Scenario | Primitives and APIs |
|---|---|---|
| `support_routing_gate.py` | Support-ticket auto-routing with an FDR guarantee (PRD 11.1) | `Set`, `Gate(guarantee="fdr")`, profile create / add examples / audit, typed `response_model`, drift monitors |
| `rag_claim_filtering.py` | RAG answer factuality on Claude (PRD 11.2) | `Claim`, `AnthropicBackend` (L0, sampling) |
| `llm_judge_cascade.py` | LLM-as-judge with an escalation cascade (PRD 11.4) | `Judge` with a gpt-4.1-mini, gpt-4.1, human-queue cascade |
| `model_cascade_routing.py` | Cost routing between a vLLM tier and an OpenAI tier (PRD 11.5) | `Route(guarantee="cost_budget")`, `VLLMBackend`, `OpenAIBackend` |
| `custom_backend_evidence.py` | Bring your own model (PRD 8.4) | `CustomBackend` at L1 and L0; prints the exact request body sent |
| `offline_calibration_no_network.py` | Offline calibration, no hosted API (PRD 8.3, 11.6) | `stats.conformal` (APS, LAC, CRC, LTT), `stats.venn_abers` (IVAP), `stats.evalues` (e-BH, `CoverageMonitor`, PPI) |
| `_mock.py` | Offline mock of `https://api.cli.dev/v1` used by the hosted examples | `httpx.MockTransport` |

## Agent-framework examples

`agents/` holds eight runnable examples that guard agent tool calls with
calibrated decisions in LangChain, LangGraph, Google ADK and Microsoft Agent
Framework, across healthcare, finance, insurance and compliance. They run
offline with `--provider mock` and with your own keys for OpenAI, Azure
OpenAI, Anthropic, Gemini, or Gemma served by vLLM or SGLang. See
[`agents/README.md`](agents/README.md).

## Setup

From the repository root:

```bash
cd sdk/python
pip install -e .            # installs cli-sdk with its only dependencies, httpx and numpy
```

The examples import `_mock` from this directory. Python adds a script's own
directory to the import path, so run them by path from anywhere.

## Run the hosted examples offline

`--mock` is the default whenever `CLI_API_KEY` is unset, so these are
equivalent:

```bash
python examples/support_routing_gate.py
python examples/support_routing_gate.py --mock
```

All of them:

```bash
python examples/support_routing_gate.py --mock
python examples/support_routing_gate.py --mock --simulate-drift   # monitors report alerts; auto-routing pauses
python examples/rag_claim_filtering.py --mock
python examples/llm_judge_cascade.py --mock
python examples/model_cascade_routing.py --mock
python examples/custom_backend_evidence.py --mock
```

Each prints `[mock] ...` on its first line when it is using the mock.

## Run the offline calibration example

```bash
python examples/offline_calibration_no_network.py
```

It takes no flags and needs no key. Sockets are disabled for the whole run,
and the HTTP client is never imported. It takes a few seconds and prints
eight sections, each checked against held-out synthetic data whose true labels
are known:

1. An overconfident 5-way classifier (confidence 0.83, accuracy 0.66).
2. APS sets at alpha = 0.10: empirical coverage against the target, set sizes,
   a local coverage audit, and an uncalibrated baseline that falls short.
3. LAC against APS: average size, and size and coverage on easy and hard items.
4. Venn-Abers IVAP intervals for "is the top answer correct?": reliability by
   confidence bin, per-item interval widths, and width shrinking as n grows.
5. Gate-style accept thresholds from CRC (`guarantee="risk"`) and from
   Learn-then-Test (`guarantee="risk_high_probability"`).
6. Batch FDR control with conformal e-values and e-BH (`guarantee="fdr"`),
   repeated over 300 fresh draws.
7. A `CoverageMonitor` that stays quiet for 2,000 in-spec outcomes and then
   fires after an injected drift.
8. Prediction-powered inference: 200 human labels plus judge labels on 9,800
   items, against human-only and judge-only estimates.

## Run against the live API

```bash
export CLI_API_KEY=sk_live_...
python examples/rag_claim_filtering.py
```

Without `--mock` and with `CLI_API_KEY` set, the examples call the hosted API.
Some things to know:

- `support_routing_gate.py` creates `support-routing-v3` if it does not
  exist, and writes 1,200 synthetic labelled tickets into it. Use a scratch
  workspace.
- The other hosted examples expect their profiles to exist already:
  `rag-factuality-v1`, `pairwise-judge-v2`, `cost-routing-v1` and
  `toy-intent-v1`.
- The backends need provider connections in your workspace. That means Azure
  OpenAI, Anthropic and OpenAI, plus a vLLM server that the service can
  reach at the configured `base_url`.

## About the mock

`_mock.py` answers every endpoint the examples use: `/evaluate`,
`/calibration-profiles` (list, create, get), `/examples`, `/audit`,
`/label-with-judge`, `/monitors`, `/alerts` and `/backends`. It is
deterministic. Answers are computed from each request, using its query ids,
types, options, context text and any client-side evidence. Every answer
carries a guarantee card built from the named profile's state: method,
alpha or target, `calibration_n`, the coverage interval for that n, and the
last audit date. A profile below its minimum size gets a `heuristic` card and
a 424 response, so the fail-closed path can be exercised offline.

The mock shows request and response shapes; its numbers are not statistics.
For real computation, use `offline_calibration_no_network.py` or
`cli_sdk.stats` directly.

To use the mock in your own tests:

```python
from _mock import MockCLIServer, open_client

server = MockCLIServer()
with open_client(server, backend={"provider": "openai", "model": "gpt-4.1-2025-04-14"}) as client:
    ...
print(server.requests[-1]["body"])   # the last request body the SDK sent
```

## Reading the guarantees

Every guarantee in these examples is marginal, or group-conditional under
Mondrian calibration, over data exchangeable with the named calibration
profile. "At most 5% of auto-routed tickets are misrouted" is a statement
about the stream of tickets, not about any single ticket. The examples act on
answers accordingly. They fail closed on `heuristic` answers, send whole
prediction sets to people when a Gate escalates, and pause automation when a
drift monitor alerts.
