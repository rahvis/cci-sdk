"""Conformal Logit Inference (CLI) — Python SDK.

Turns the raw output of any LLM into finite-sample statistical guarantees
using conformal prediction, Venn-Abers calibration, and e-values.

Hosted API::

    from cli_sdk import CLIClient, Set, Gate

Offline statistics engine (no network access)::

    from cli_sdk.stats.conformal import aps
    from cli_sdk.stats.venn_abers import ivap
    from cli_sdk.stats.evalues import ebh, eprocess
"""

from cli_sdk.answers import (
    Answer,
    BeliefAnswer,
    ClaimAnswer,
    EvaluateResponse,
    GateAnswer,
    Guarantee,
    IntervalAnswer,
    JudgeAnswer,
    RouteAnswer,
    SetAnswer,
    Usage,
)
from cli_sdk.backends import (
    AnthropicBackend,
    AzureOpenAIBackend,
    BedrockBackend,
    CustomBackend,
    GeminiBackend,
    OpenAIBackend,
    OpenRouterBackend,
    SGLangBackend,
    VLLMBackend,
)
from cli_sdk.calibration import CalibrationExample, CalibrationProfile
from cli_sdk.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BackendError,
    CLIError,
    ConfigurationError,
    InsufficientCalibrationError,
    RateLimitError,
    ValidationError,
)
from cli_sdk.monitoring import Alert, LocalMonitor
from cli_sdk.queries import Belief, Claim, Gate, Interval, Judge, Query, Route, Set

__version__ = "0.1.1"

# The HTTP clients are imported lazily so that the offline statistics
# engine (``cli_sdk.stats``) and the pure-Python query/answer types never
# require the HTTP stack. ``from cli_sdk import CLIClient`` still works.
_LAZY_CLIENT_NAMES = {"CLIClient", "AsyncCLIClient", "RetryConfig"}
_LAZY_LOCAL_NAMES = {"LocalCLIClient"}


def __getattr__(name: str):
    if name in _LAZY_CLIENT_NAMES:
        from cli_sdk import client as _client

        return getattr(_client, name)
    if name in _LAZY_LOCAL_NAMES:
        from cli_sdk import local as _local

        return getattr(_local, name)
    raise AttributeError(f"module 'cli_sdk' has no attribute {name!r}")

__all__ = [
    # clients
    "CLIClient",
    "AsyncCLIClient",
    "RetryConfig",
    "LocalCLIClient",
    # queries
    "Query",
    "Belief",
    "Set",
    "Interval",
    "Gate",
    "Claim",
    "Judge",
    "Route",
    # answers
    "Answer",
    "Guarantee",
    "BeliefAnswer",
    "SetAnswer",
    "IntervalAnswer",
    "GateAnswer",
    "ClaimAnswer",
    "JudgeAnswer",
    "RouteAnswer",
    "EvaluateResponse",
    "Usage",
    # calibration and monitoring
    "CalibrationProfile",
    "CalibrationExample",
    "Alert",
    "LocalMonitor",
    # backends
    "OpenAIBackend",
    "AzureOpenAIBackend",
    "AnthropicBackend",
    "GeminiBackend",
    "BedrockBackend",
    "OpenRouterBackend",
    "VLLMBackend",
    "SGLangBackend",
    "CustomBackend",
    # errors
    "CLIError",
    "APIConnectionError",
    "AuthenticationError",
    "ValidationError",
    "InsufficientCalibrationError",
    "RateLimitError",
    "BackendError",
    "ConfigurationError",
    "__version__",
]
