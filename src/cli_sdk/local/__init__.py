"""Local mode: calibrate and evaluate CLI queries in-process, on your own model.

``LocalCLIClient`` runs the same statistics engine as the hosted API
(``cli_sdk.stats``) against an evidence backend you control
(``cli_sdk.evidence``), and stores calibration profiles as reviewable JSON
files. Use it to try CLI with nothing but a model API key, to run
self-hosted models (vLLM, SGLang) in air-gapped environments, and inside
agent frameworks (``cli_sdk.integrations``).
"""

from cli_sdk.local.client import LocalCLIClient
from cli_sdk.local.store import LocalProfileStore, ProfileRecord, query_fingerprint

__all__ = ["LocalCLIClient", "LocalProfileStore", "ProfileRecord", "query_fingerprint"]
