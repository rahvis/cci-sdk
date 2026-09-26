"""Retry policy: exponential backoff with full jitter, honoring ``retry-after``."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class RetryConfig:
    """How the client retries transient failures.

    ``429 Too Many Requests``, ``502 Backend Error`` and ``529 Overloaded``
    are retried by default, and so are connection errors and timeouts
    (``retry_connection_errors``). Everything else (auth, validation,
    insufficient calibration) is returned immediately, because retrying
    cannot change the outcome. Every POST carries an ``Idempotency-Key``
    that is reused across its retries, so a retried write is applied once.

    A server ``retry-after`` is honoured in full up to ``max_retry_after``
    seconds; a longer one stops retrying, and the error (for a 429,
    ``RateLimitError`` with ``retry_after``) is raised to the caller.
    """

    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    max_retry_after: float = 60.0
    retry_connection_errors: bool = True
    retry_statuses: frozenset[int] = field(default_factory=lambda: frozenset({429, 502, 529}))

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be non-negative")

    def should_retry(self, status_code: int, attempt: int, retry_after: Optional[str] = None) -> bool:
        """``attempt`` is 1-based: the attempt that just failed."""
        if status_code not in self.retry_statuses or attempt >= self.max_attempts:
            return False
        wait = _seconds(retry_after)
        return wait is None or wait <= self.max_retry_after

    def should_retry_connection_error(self, attempt: int) -> bool:
        return self.retry_connection_errors and attempt < self.max_attempts

    def delay(self, attempt: int, retry_after: Optional[str] = None) -> float:
        """Seconds to wait before the next attempt.

        A numeric ``retry-after`` header wins (honoured in full, up to
        ``max_retry_after``); otherwise full-jitter exponential backoff:
        uniform in ``[0, min(max_delay, base_delay * 2**(attempt-1))]``.
        """
        wait = _seconds(retry_after)
        if wait is not None:
            return min(self.max_retry_after, wait)
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return random.uniform(0.0, ceiling)


def _seconds(retry_after: Optional[str]) -> Optional[float]:
    if retry_after is None:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return None


NO_RETRY = RetryConfig(max_attempts=1)
