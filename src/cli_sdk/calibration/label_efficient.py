"""Label-efficient calibration: few human labels, many judge labels.

The hosted ``label_with_judge`` endpoint runs this end to end. The helpers
here let you do the same estimate locally, for instance to decide how many
human labels you need before paying for them.

Validity never depends on the judge being good: a poor judge only widens
the interval. What a poor judge cannot do is save labels. If the judge is
no more accurate than the model being evaluated, no debiasing method can
cut the required human labels by more than about half (research.md,
section 10.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from cli_sdk.stats.evalues.ppi import PPIResult, estimate_mean


@dataclass(frozen=True)
class JudgeQuality:
    agreement_with_humans: float
    correlation: float
    effective_label_multiplier: float
    """Roughly how many human labels each human label is worth once the
    judge-labelled pool is added (1.0 means the judge adds nothing)."""


def judge_quality(human_labels: Sequence[float], judge_labels_on_same: Sequence[float],
                  pool_size: Optional[int] = None) -> JudgeQuality:
    """Diagnose a judge on the human-labelled subset it also labelled.

    ``pool_size`` is the number of judge-only items (N). With it, the
    multiplier is the exact large-sample gain of power-tuned PPI,
    ``1 / (1 - rho^2 * N / (n + N))``, which can never exceed ``(n + N) / n``.
    Without it, the ``N >> n`` limit ``1 / (1 - rho^2)`` is reported, which
    grows without bound as ``rho`` approaches 1: pass ``pool_size`` for a
    realistic number.
    """
    human = np.asarray(human_labels, dtype=float)
    judge = np.asarray(judge_labels_on_same, dtype=float)
    agreement = float(np.mean(human == judge)) if human.size else 0.0
    if human.size > 1 and np.std(human) > 0 and np.std(judge) > 0:
        rho = float(np.corrcoef(human, judge)[0, 1])
    else:
        rho = 0.0
    # Power-tuned prediction-powered inference: the variance of the estimate
    # shrinks by a factor 1 - rho^2 * N / (n + N) relative to human labels alone.
    n = int(human.size)
    shrink = rho**2 if pool_size is None else rho**2 * pool_size / max(1, n + pool_size)
    multiplier = 1.0 / max(1e-9, 1.0 - shrink)
    return JudgeQuality(agreement_with_humans=agreement, correlation=rho, effective_label_multiplier=multiplier)


def estimate_rate_with_judge(
    human_labels: Sequence[float],
    judge_labels_on_human_sample: Sequence[float],
    judge_labels_on_pool: Sequence[float],
    alpha: float = 0.05,
) -> PPIResult:
    """Prediction-powered estimate of a rate (e.g. coverage or error rate).

    ``human_labels`` and ``judge_labels_on_human_sample`` must be aligned:
    the same items, labelled once by humans and once by the judge.
    ``judge_labels_on_pool`` are judge labels on the unlabelled pool.
    """
    if len(human_labels) != len(judge_labels_on_human_sample):
        raise ValueError("human and judge labels on the human-labelled sample must be aligned")
    return estimate_mean(
        labelled_predictions=np.asarray(judge_labels_on_human_sample, dtype=float),
        labelled_labels=np.asarray(human_labels, dtype=float),
        unlabelled_predictions=np.asarray(judge_labels_on_pool, dtype=float),
        alpha=alpha,
    )
