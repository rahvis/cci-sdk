"""Exceptions raised by the Conformal Logit Inference client."""

from __future__ import annotations


class CLIError(Exception):
    """Base class for every error this SDK raises."""


class AuthenticationError(CLIError):
    """Raised on a 401 response: missing or invalid API key."""


class ValidationError(CLIError):
    """Raised on a 422 response: the request body failed validation."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


class InsufficientCalibrationError(CLIError):
    """Raised on a 424 response.

    A query named a ``calibration_profile`` that does not yet have enough
    labelled examples to support the requested guarantee. The response
    still contains a ``heuristic``-labelled answer; this exception is
    raised only when the caller has opted into strict mode (see
    ``CLIClient(strict_guarantees=True)``) rather than silently receiving
    a heuristic answer.
    """


class RateLimitError(CLIError):
    """Raised on a 429 response after retries are exhausted."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class BackendError(CLIError):
    """Raised on a 502 response: the connected model backend errored, or
    returned a lower access level than the query required."""


class ConfigurationError(CLIError, ValueError):
    """Raised for client-side misconfiguration (missing API key, an
    unregistered backend, a primitive missing a required field) that
    never reaches the network.

    Also a ``ValueError``, so code that validates arguments with
    ``except ValueError`` keeps working."""


class APIConnectionError(CLIError):
    """The API could not be reached (connection refused, DNS failure,
    timeout) after the configured retries. The request may or may not have
    been processed; every POST carries an ``Idempotency-Key`` so retrying
    it is safe."""
