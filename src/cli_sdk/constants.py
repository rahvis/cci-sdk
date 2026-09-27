"""Shared constants for the Conformal Logit Inference client."""

from __future__ import annotations

DEFAULT_BASE_URL = "https://cci.gitdate.ink/api/v1"

ENV_API_KEY = "CLI_API_KEY"
ENV_BASE_URL = "CLI_BASE_URL"

DEFAULT_TIMEOUT_SECONDS = 60.0

# Guarantee types a `guarantee` block on an answer may carry. See
# apps/docs/pages/guarantees.mdx for the full reference.
GUARANTEE_COVERAGE = "coverage"
GUARANTEE_CALIBRATION = "calibration"
GUARANTEE_RISK = "risk"
GUARANTEE_RISK_HIGH_PROBABILITY = "risk_high_probability"
GUARANTEE_FDR = "fdr"
GUARANTEE_ANYTIME = "anytime"
GUARANTEE_COST_BUDGET = "cost_budget"
GUARANTEE_HEURISTIC = "heuristic"

# Access levels a backend adapter can report. See
# apps/docs/pages/concepts/access-ladder.mdx.
ACCESS_LEVEL_L0 = "L0"  # sampled text only
ACCESS_LEVEL_L1 = "L1"  # generated-token log-probabilities
ACCESS_LEVEL_L2 = "L2"  # scoring of text you supply
ACCESS_LEVEL_L3 = "L3"  # exact label-token probabilities / full logits
ACCESS_LEVEL_L4 = "L4"  # hidden states

USER_AGENT = "cci-sdk-python/0.1.0"
