"""e-BH: false-discovery-rate control from e-values (Wang & Ramdas, 2022).

Unlike Benjamini-Hochberg on p-values, e-BH controls FDR at the target
level ``q`` under *arbitrary dependence* between the simultaneous tests —
exactly the setting where several ``Gate`` or ``Judge`` decisions in one
batch may be correlated (drawn from related inputs, or scored by the same
underlying model). Powers the ``Gate`` primitive's ``"fdr"`` guarantee
mode when decisions cannot be assumed independent. See research.md,
section 2.3 and 9.1.
"""

from __future__ import annotations

import numpy as np


def select(e_values: np.ndarray, q: float) -> np.ndarray:
    """Return the indices of e-values selected (rejected) at FDR level q.

    Procedure: sort e-values descending; find the largest k such that the
    k-th largest e-value is >= m / (q * k); select the top k. If no such
    k exists, select none.
    """
    e_values = np.asarray(e_values, dtype=float)
    m = e_values.shape[0]
    if m == 0:
        return np.array([], dtype=int)
    if not (0 < q < 1):
        raise ValueError("q must be in (0, 1)")

    order = np.argsort(-e_values, kind="stable")
    sorted_e = e_values[order]

    k = 0
    for i in range(1, m + 1):
        if sorted_e[i - 1] >= m / (q * i):
            k = i
    return np.sort(order[:k])


def calibrator_from_p_value(p_value: float, kappa: float = 0.5) -> float:
    """Convert a valid p-value into a valid e-value.

    Uses the admissible calibrator family e = kappa * p^(kappa - 1) for
    kappa in (0, 1) (Vovk & Wang, 2021), valid for any p-value that is
    super-uniform under the null. The naive e = 1/p is not a valid
    calibrator: for a uniform p-value, E[1/p] is infinite.
    """
    if not (0 < p_value <= 1):
        raise ValueError("p_value must be in (0, 1]")
    if not (0 < kappa < 1):
        raise ValueError("kappa must be in (0, 1)")
    return kappa * p_value ** (kappa - 1)
