"""Pool-adjacent-violators (PAV) isotonic regression, no external dependency.

Given values in a fixed order, returns the non-decreasing sequence that
minimizes weighted sum of squared error to the input — the building block
Venn-Abers predictors fit at every calibration step.
"""

from __future__ import annotations

import numpy as np


def pool_adjacent_violators(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Isotonic (non-decreasing) least-squares fit of ``y``, in its given order.

    Uses the standard stack-based PAV algorithm: scan left to right,
    pooling (weighted-averaging) any block whose value exceeds the next
    one, repeatedly, until the sequence of pooled blocks is non-decreasing.
    Runs in O(n).
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)

    # Each stack entry: [value, weight, start_index, end_index]
    stack: list[list[float]] = []
    for i in range(n):
        block = [y[i], w[i], float(i), float(i)]
        while stack and stack[-1][0] > block[0]:
            prev = stack.pop()
            total_w = prev[1] + block[1]
            merged_value = (prev[0] * prev[1] + block[0] * block[1]) / total_w
            block = [merged_value, total_w, prev[2], block[3]]
        stack.append(block)

    result = np.empty(n)
    for value, _weight, start, end in stack:
        result[int(start) : int(end) + 1] = value
    return result
