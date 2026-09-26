"""Shared building blocks for anytime-valid e-value procedures.

``BettingMartingale`` is the core construction behind every sequential
monitor in this package: for a 0/1 stream X_1, X_2, ... with
E[X_i] <= null_rate under the null hypothesis,

    M_t(lambda) = prod_{i<=t} (1 + lambda * (X_i - null_rate))

is a nonnegative supermartingale under the null for any fixed lambda in
[0, 1 / null_rate], because E[1 + lambda*(X_i - null_rate)] =
1 + lambda*(E[X_i] - null_rate) <= 1. By Ville's inequality,
P(exists t : M_t >= 1/delta) <= delta — a false-alarm guarantee that
holds no matter how often, or for how long, the process is checked
(research.md, section 9.1-9.2). This is the "betting" formulation of
sequential testing (Shafer & Vovk; Waudby-Smith & Ramdas).

Because a convex combination of valid nonnegative supermartingales is
itself a valid nonnegative supermartingale, this implementation averages
M_t(lambda) over a geometric grid of bets, from 0.5% to 80% of the maximum
bet 1 / null_rate. Small bets grow under small, sustained drift (for
example coverage 0.87 against a 0.90 target); large bets react quickly to
large drift. Each component is kept in log space, so a long healthy
stream never underflows a component to zero. Detecting a drift of size d
still takes on the order of null_rate / d**2 observations: the monitor
controls false alarms exactly and detects drift as fast as the data allow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, ~1e-9 accuracy)."""
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1)")
    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


_BET_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8)


def _lambda_grid(null_rate: float) -> list[float]:
    cap = 1.0 / max(null_rate, 1e-6)
    return [cap * fraction for fraction in _BET_FRACTIONS]


@dataclass
class BettingMartingale:
    """Detects a 0/1 stream's mean rising above ``null_rate``, anytime-validly.

    Call ``update(indicator)`` once per observation. Returns the current
    combined e-value; compare it against ``1 / false_alarm_rate`` to
    decide whether to alarm (or use ``triggered`` for convenience).
    """

    null_rate: float
    false_alarm_rate: float = 0.05
    n: int = 0
    _lambdas: list[float] = field(default_factory=list, repr=False)
    _log_wealth: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not (0 < self.null_rate < 1):
            raise ValueError("null_rate must be in (0, 1)")
        if not (0 < self.false_alarm_rate < 1):
            raise ValueError("false_alarm_rate must be in (0, 1)")
        self._lambdas = _lambda_grid(self.null_rate)
        self._log_wealth = [0.0 for _ in self._lambdas]

    @property
    def log_e_value(self) -> float:
        """log of the mixture e-value (log-mean-exp of the components)."""
        peak = max(self._log_wealth)
        return peak + math.log(sum(math.exp(w - peak) for w in self._log_wealth) / len(self._log_wealth))

    @property
    def e_value(self) -> float:
        log_e = self.log_e_value
        return math.exp(log_e) if log_e < 700 else math.inf

    @property
    def triggered(self) -> bool:
        return self.log_e_value >= math.log(1.0 / self.false_alarm_rate)

    def update(self, indicator: float) -> float:
        """``indicator`` should be 0 or 1 (or a value in [0, 1]). Returns the new e-value."""
        if not (0.0 <= indicator <= 1.0):
            raise ValueError("indicator must be in [0, 1]")
        self.n += 1
        for i, lam in enumerate(self._lambdas):
            # lam * null_rate < 1 on the whole grid, so the factor stays positive.
            self._log_wealth[i] += math.log1p(lam * (indicator - self.null_rate))
        return self.e_value
