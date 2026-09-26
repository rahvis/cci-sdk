"""Per-primitive calibration and prediction for local mode.

Each primitive has two halves:

``evidence``  model calls for one context (``cli_sdk.local.scoring``)
``answer``    turn test evidence plus stored calibration records into an
              answer payload with a guarantee card

Answers are built as the same JSON payloads the hosted API returns and then
parsed by ``cli_sdk.answers``, so local and hosted answers are the same
types with the same fail-closed behavior.

Methods (see research.md for derivations and references)

========  ===========================  =========================================
Query     Method                       Guarantee
========  ===========================  =========================================
Set       LAC / APS / RAPS (+Mondrian) P(label in set) >= 1 - alpha
Belief    IVAP (Venn-Abers)            calibrated probability interval [p0, p1]
Interval  ordinal APS (contiguous)     P(level in interval) >= 1 - alpha
Gate      CRC                          E[approved and wrong] <= target
Gate      RCPS (Hoeffding-Bentkus)     P(risk <= target) >= 1 - delta
Gate      LTT selective (fdr)          wrong / approved <= target, w.p. 1 - delta
Claim     conformal factuality         P(all retained claims true) >= 1 - alpha
Judge     trust-or-escalate cascade    agreement among non-escalated >= 1 - alpha
Route     conformal cascade            P(served answer wrong) <= alpha
========  ===========================  =========================================
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import numpy as np

from cli_sdk.backends.base import CustomBackend
from cli_sdk.exceptions import ConfigurationError
from cli_sdk.local import scoring
from cli_sdk.local.scoring import UsageCounter
from cli_sdk.local.store import ProfileRecord
from cli_sdk.stats.conformal import aps as aps_mod
from cli_sdk.stats.conformal import crc, rcps
from cli_sdk.stats.conformal import mondrian as mondrian_mod
from cli_sdk.stats.conformal._quantile import (
    conformal_quantile,
    coverage_confidence_interval,
    minimum_calibration_size,
)
from cli_sdk.stats.conformal import raps as raps_mod
from cli_sdk.stats.conformal.raps import _penalized_cumulative_mass
from cli_sdk.stats.venn_abers import ivap

STATS_VERSION = "cli-local/0.1"
DEFAULT_ALPHA = 0.10
DEFAULT_DELTA = 0.10
RAPS_K_REG = 1
RAPS_LAMBDA = 0.01
CALIBRATION_SEED = 0
MIN_VENN_ABERS_N = 20
# A fixed, data-independent grid of confidence thresholds for the
# Learn-then-Test procedures (Gate fdr, Judge). Fixing the grid before
# seeing calibration data is what keeps the Bonferroni correction valid.
LTT_GRID = tuple(round(float(x), 2) for x in np.linspace(0.0, 1.0, 101))
LTT_STARTS = (0.99, 0.95, 0.90, 0.80, 0.70)

Resolver = Callable[[str], CustomBackend]


class Heuristic(Exception):
    """Internal: calibration cannot support a guarantee; answer fails closed."""


# ---------------------------------------------------------------------------
# guarantee cards
# ---------------------------------------------------------------------------


def _card(record: ProfileRecord, gtype: str, method: str, *, n: Optional[int] = None, **fields: Any) -> dict[str, Any]:
    card: dict[str, Any] = {
        "type": gtype,
        "method": method,
        "calibration_profile": record.name,
        "calibration_n": record.n if n is None else n,
        "stats_version": STATS_VERSION,
    }
    card.update({k: v for k, v in fields.items() if v is not None})
    return card


def heuristic_card(profile: str, reason: str, n: int = 0) -> dict[str, Any]:
    return {
        "type": "heuristic",
        "calibration_profile": profile,
        "calibration_n": n,
        "statement": f"No guarantee: {reason}",
        "stats_version": STATS_VERSION,
    }


def _require_n(n: int, needed: int, what: str) -> None:
    if n < needed:
        raise Heuristic(f"{what} needs at least {needed} calibration examples; the profile has {n}")


def minimum_examples(query_payload: dict[str, Any]) -> int:
    """The smallest profile size for which ``query_payload`` can return a guaranteed answer.

    Mirrors the ``_require_n`` checks each answer function applies, so a
    profile's reported ``minimum_n`` and ``status`` match what evaluation
    actually does (for example, RCPS at target=0.05, delta=0.10 needs 47
    examples, not the 19 a split-conformal quantile would need).
    """
    from types import SimpleNamespace

    qtype = query_payload.get("type")
    if qtype == "belief":
        return MIN_VENN_ABERS_N
    if qtype == "gate":
        target = float(query_payload["target"])
        guarantee = query_payload.get("guarantee") or "risk"
        if guarantee == "risk_high_probability":
            return math.ceil(math.log(1.0 / float(query_payload["delta"])) / target)
        if guarantee == "fdr":
            delta = query_payload.get("delta") or DEFAULT_DELTA
            return math.ceil(math.log(len(LTT_STARTS) / float(delta)) / target)
        return math.ceil(1.0 / target - 1.0)
    alpha = float(query_payload.get("alpha") or DEFAULT_ALPHA)
    shape = SimpleNamespace(cascade=query_payload.get("cascade"))
    if qtype == "judge":
        stages = len(judge_stages(shape))
        return math.ceil(math.log(len(LTT_STARTS) * stages / DEFAULT_DELTA) / alpha)
    if qtype == "route":
        tiers = max(1, len(route_tiers(shape) if shape.cascade else []))
        return minimum_calibration_size(alpha / tiers)
    return minimum_calibration_size(alpha)


def _labels_index(records: Sequence[dict[str, Any]], keys: list[str]) -> np.ndarray:
    index = {key: i for i, key in enumerate(keys)}
    try:
        return np.array([index[scoring.canonical(r["label"])] for r in records], dtype=int)
    except KeyError as exc:
        raise ConfigurationError(f"calibration label {exc.args[0]!r} is not one of the options {keys}") from exc


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------


def _set_scores(method: str, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(CALIBRATION_SEED)
    if method == "LAC":
        return 1.0 - probs[np.arange(len(labels)), labels]
    if method == "APS":
        return np.array([aps_mod._score_true_class(probs[i], int(labels[i]), rng) for i in range(len(labels))])
    if method == "RAPS":
        return np.array([raps_mod.true_class_score(probs[i], int(y), RAPS_K_REG, RAPS_LAMBDA, rng.uniform())
                         for i, y in enumerate(labels)])
    raise ConfigurationError(f"unsupported Set method {method!r}")


def _set_predict(method: str, probs: np.ndarray, q_hat: float) -> list[int]:
    if method == "LAC":
        return [int(j) for j in np.flatnonzero(1.0 - probs <= q_hat)]
    if method == "APS":
        order, cum = aps_mod._sorted_cumulative_mass(probs)
    else:
        order, cum = _penalized_cumulative_mass(probs, RAPS_K_REG, RAPS_LAMBDA)
    cutoff = int(np.searchsorted(cum, q_hat, side="left"))
    included = order[: cutoff + 1] if cutoff < len(order) else order
    return sorted(int(j) for j in included)


def _venn_abers_per_option(cal: np.ndarray, labels: np.ndarray, test: np.ndarray) -> list[tuple[float, float]]:
    intervals = []
    for j in range(cal.shape[1]):
        p0, p1 = ivap.calibrate_and_predict(cal[:, j], (labels == j).astype(float), np.array([test[j]]))
        intervals.append((float(p0[0]), float(p1[0])))
    return intervals


def set_evidence(backend, query, context, sample_count, usage, resolve=None) -> dict[str, Any]:
    probs = scoring.option_probabilities(backend, context, query.instructions, query.options, sample_count, usage)
    return {"probs": [probs[k] for k in query.options]}


def set_answer(query, record: ProfileRecord, evidence, group: Optional[str], warnings: list[str]) -> dict[str, Any]:
    keys = list(query.options)
    test = np.asarray(evidence["probs"], dtype=float)
    probabilities = {k: float(p) for k, p in zip(keys, test)}
    alpha = query.alpha if query.alpha is not None else DEFAULT_ALPHA
    method = query.method or "APS"
    try:
        n = record.n
        _require_n(n, minimum_calibration_size(alpha), f"alpha={alpha}")
        cal = np.array([r["evidence"]["probs"] for r in record.records], dtype=float)
        labels = _labels_index(record.records, keys)
        scores = _set_scores(method, cal, labels)
        group_n = None
        if query.group_by:
            groups = np.array([scoring.canonical(r.get("group")) for r in record.records])
            fit = mondrian_mod.calibrate(scores, groups, alpha)
            key = scoring.canonical(group)
            q_hat = fit.threshold_for(key)
            group_n = fit.group_sizes.get(key, 0)
            if key in fit.underpowered_groups or key not in fit.group_thresholds:
                warnings.append(
                    f"{record.name}: group {key!r} has {group_n} calibration examples; "
                    "using the marginal threshold (coverage is marginal, not group-conditional)"
                )
                group_n = None
        else:
            q_hat = conformal_quantile(scores, alpha)
        members = _set_predict(method, test, q_hat)
        va = _venn_abers_per_option(cal, labels, test)
        effective_n = group_n if group_n is not None else n
        card = _card(
            record, "coverage", method, n=effective_n, alpha=alpha,
            coverage_ci=list(coverage_confidence_interval(effective_n, alpha)),
            group=scoring.canonical(group) if group_n is not None else None,
        )
        return {
            "type": "set",
            "set": [keys[j] for j in members],
            "probabilities": probabilities,
            "venn_abers": {keys[j]: list(va[j]) for j in range(len(keys))},
            "guarantee": card,
        }
    except Heuristic as reason:
        return {"type": "set", "set": keys, "probabilities": probabilities,
                "guarantee": heuristic_card(record.name, str(reason), record.n)}


# ---------------------------------------------------------------------------
# Belief
# ---------------------------------------------------------------------------


def binary_evidence(backend, query, context, sample_count, usage, resolve=None) -> dict[str, Any]:
    return {"score": scoring.probability_true(backend, context, query.instructions, sample_count, usage)}


def _binary_labels(records: Sequence[dict[str, Any]]) -> np.ndarray:
    out = []
    for r in records:
        label = r["label"]
        if isinstance(label, str):
            if label.lower() not in ("true", "false", "yes", "no", "1", "0"):
                raise ConfigurationError(f"binary label must be true/false, got {label!r}")
            label = label.lower() in ("true", "yes", "1")
        out.append(1.0 if bool(label) else 0.0)
    return np.array(out)


def _ivap(record: ProfileRecord, score: float) -> tuple[float, float]:
    cal = np.array([r["evidence"]["score"] for r in record.records], dtype=float)
    labels = _binary_labels(record.records)
    p0, p1 = ivap.calibrate_and_predict(cal, labels, np.array([score]))
    return float(p0[0]), float(p1[0])


def belief_answer(query, record: ProfileRecord, evidence, group, warnings) -> dict[str, Any]:
    score = float(evidence["score"])
    if record.n < MIN_VENN_ABERS_N:
        return {"type": "belief", "probability": score, "venn_abers": [0.0, 1.0],
                "guarantee": heuristic_card(
                    record.name, f"Venn-Abers needs at least {MIN_VENN_ABERS_N} examples; the profile has {record.n}",
                    record.n)}
    p0, p1 = _ivap(record, score)
    merged = float(ivap.merge_to_probability(np.array([p0]), np.array([p1]))[0])
    card = _card(record, "calibration", "IVAP", statement=(
        f"Venn-Abers pair [{p0:.3f}, {p1:.3f}]: the probability computed under the true label is "
        f"calibrated on data exchangeable with the {record.n} calibration examples; a wide pair means "
        "the calibration data cannot pin the probability down."))
    return {"type": "belief", "probability": merged, "venn_abers": [p0, p1], "raw_score": score, "guarantee": card}


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _selective_threshold(conf: np.ndarray, correct: np.ndarray, alpha: float, delta: float) -> float:
    """Smallest grid threshold whose error rate *among accepted* is <= alpha w.p. 1 - delta.

    Learn-then-Test (Angelopoulos, Bates, Candes, Jordan & Lei) with
    multi-start fixed-sequence testing. H0(lambda): P(wrong | accepted at
    lambda) > alpha, tested with a Hoeffding-Bentkus p-value on the accepted
    points. From each of a few fixed starting thresholds, thresholds are
    tested from conservative to permissive until the first failure; a
    Bonferroni split of delta across the starts controls the family-wise
    error. Multiple starts matter because the most conservative thresholds
    accept too few points to test on their own.
    """
    if conf.shape[0] == 0:
        return math.inf
    per_start = delta / len(LTT_STARTS)

    def rejected(lam: float) -> bool:
        accepted = conf >= lam
        n_acc = int(accepted.sum())
        if n_acc == 0:
            return False
        risk = float(np.mean(1.0 - correct[accepted]))
        return rcps._hoeffding_bentkus_p_value(risk, n_acc, alpha) <= per_start

    valid: list[float] = []
    for start in LTT_STARTS:
        for lam in (g for g in reversed(LTT_GRID) if g <= start):
            if not rejected(lam):
                break
            valid.append(lam)
    return min(valid, default=math.inf)


def gate_answer(query, record: ProfileRecord, evidence, group, warnings) -> dict[str, Any]:
    score = float(evidence["score"])
    target = query.target
    try:
        n = record.n
        cal = np.array([r["evidence"]["score"] for r in record.records], dtype=float)
        correct = _binary_labels(record.records) if n else np.array([])
        losses = 1.0 - correct
        if query.guarantee == "risk":
            _require_n(n, math.ceil(1.0 / target - 1.0), f"target={target}")
            lam = crc.calibrate(cal, losses, target)
            card = _card(record, "risk", "CRC", target=target, statement=(
                f"Expected rate of decisions that are auto-approved and wrong is at most {target:g} "
                f"(conformal risk control, n={n})."))
        elif query.guarantee == "risk_high_probability":
            _require_n(n, math.ceil(math.log(1.0 / query.delta) / target), f"target={target}, delta={query.delta}")
            lam = rcps.calibrate(cal, losses, target, query.delta)
            card = _card(record, "risk_high_probability", "RCPS", target=target, delta=query.delta, statement=(
                f"With {1 - query.delta:.0%} confidence, the rate of decisions that are auto-approved and "
                f"wrong is at most {target:g} (RCPS, Hoeffding-Bentkus, n={n})."))
        else:  # fdr
            delta = query.delta if query.delta is not None else DEFAULT_DELTA
            _require_n(n, math.ceil(math.log(len(LTT_STARTS) / delta) / target), f"target={target}, delta={delta}")
            lam = _selective_threshold(cal, correct, target, delta)
            card = _card(record, "fdr", "LTT-selective", target=target, delta=delta, statement=(
                f"With {1 - delta:.0%} confidence, at most {target:g} of auto-approved decisions are wrong "
                f"(Learn-then-Test, fixed-sequence, n={n})."))
        decision = "auto_approve" if score >= lam else "escalate"
        payload: dict[str, Any] = {"type": "gate", "decision": decision, "confidence": score,
                                   "threshold": None if math.isinf(lam) else lam, "guarantee": card}
        if n >= MIN_VENN_ABERS_N:
            payload["venn_abers"] = list(_ivap(record, score))
        if math.isinf(lam):
            warnings.append(
                f"{record.name}: no threshold meets target={target:g} on this calibration set; every "
                "request escalates. Add calibration examples or relax the target."
            )
        return payload
    except Heuristic as reason:
        return {"type": "gate", "decision": "escalate", "confidence": score,
                "guarantee": heuristic_card(record.name, str(reason), record.n)}


def gate_batch_answers(query, record: ProfileRecord, scores: Sequence[float], warnings: list[str]) -> list[dict[str, Any]]:
    """Conformal selection (Jin & Candes, 2023) over a batch: FDR <= target in finite samples.

    Conformal p-value for each test item under H0 "this item is wrong", using
    the calibration items that were wrong as the null reference, followed by
    Benjamini-Hochberg at level ``target`` across the batch.
    """
    target = query.target
    cal = np.array([r["evidence"]["score"] for r in record.records], dtype=float)
    correct = _binary_labels(record.records) if record.n else np.array([])
    nulls = cal[correct == 0]
    m = len(scores)
    if record.n == 0 or len(nulls) == 0:
        reason = "the profile has no calibration examples labelled wrong, so no null reference exists"
        return [{"type": "gate", "decision": "escalate", "confidence": float(s),
                 "guarantee": heuristic_card(record.name, reason, record.n)} for s in scores]
    n = record.n
    # Jin & Candes: p_j = (1 + #{i null : V_i >= V_j}) / (n + 1), with V = score
    # computed over *all* n calibration points' null-indicator (wrong items).
    pvals = np.array([(1.0 + np.sum(nulls >= s)) / (n + 1.0) for s in scores])
    order = np.argsort(pvals, kind="stable")
    k = 0
    for i in range(1, m + 1):
        if pvals[order[i - 1]] <= target * i / m:
            k = i
    selected = set(order[:k].tolist())
    out = []
    for j, s in enumerate(scores):
        card = _card(record, "fdr", "conformal-selection-BH", target=target, statement=(
            f"Across this batch of {m}, the expected fraction of approved items that are wrong is at "
            f"most {target:g} (conformal selection with Benjamini-Hochberg, n={n})."))
        out.append({"type": "gate", "decision": "auto_approve" if j in selected else "escalate",
                    "confidence": float(s), "p_value": float(pvals[j]), "guarantee": card})
    return out


# ---------------------------------------------------------------------------
# Interval (ordinal)
# ---------------------------------------------------------------------------


def _ordinal_path(p: np.ndarray) -> list[tuple[int, int, int, float]]:
    """Greedy contiguous growth from the mode: (lo, hi, added_level, cumulative_mass)."""
    k = len(p)
    lo = hi = int(np.argmax(p))
    mass = float(p[lo])
    path = [(lo, hi, lo, mass)]
    while lo > 0 or hi < k - 1:
        left = p[lo - 1] if lo > 0 else -1.0
        right = p[hi + 1] if hi < k - 1 else -1.0
        if right > left:
            hi += 1
            added = hi
        else:
            lo -= 1
            added = lo
        mass += float(p[added])
        path.append((lo, hi, added, mass))
    return path


def _ordinal_score(p: np.ndarray, y: int, rng: np.random.Generator) -> float:
    path = _ordinal_path(p)
    for t, (_, _, added, mass) in enumerate(path):
        if added == y:
            prev = path[t - 1][3] if t > 0 else 0.0
            return prev + rng.uniform() * float(p[y])
    raise AssertionError("unreachable: every level is eventually added")


def _ordinal_predict(p: np.ndarray, q_hat: float) -> tuple[int, int]:
    path = _ordinal_path(p)
    for lo, hi, _, mass in path:
        if mass >= q_hat:
            return lo, hi
    return path[-1][0], path[-1][1]


def interval_evidence(backend, query, context, sample_count, usage, resolve=None) -> dict[str, Any]:
    options = {level: None for level in query.levels}
    probs = scoring.option_probabilities(backend, context, query.instructions, options, sample_count, usage)
    return {"probs": [probs[level] for level in query.levels]}


def interval_answer(query, record: ProfileRecord, evidence, group, warnings) -> dict[str, Any]:
    if query.method not in (None, "ordinal-aps"):
        raise ConfigurationError("local mode supports Interval(method='ordinal-aps') for ordinal levels")
    levels = list(query.levels)
    test = np.asarray(evidence["probs"], dtype=float)
    legend = {str(i): level for i, level in enumerate(levels)}
    point = float(np.dot(test, np.arange(len(levels))))
    alpha = query.alpha if query.alpha is not None else DEFAULT_ALPHA
    try:
        _require_n(record.n, minimum_calibration_size(alpha), f"alpha={alpha}")
        cal = np.array([r["evidence"]["probs"] for r in record.records], dtype=float)
        labels = _labels_index(record.records, levels)
        rng = np.random.default_rng(CALIBRATION_SEED)
        scores = np.array([_ordinal_score(cal[i], int(labels[i]), rng) for i in range(len(labels))])
        q_hat = conformal_quantile(scores, alpha)
        lo, hi = _ordinal_predict(test, q_hat)
        card = _card(record, "coverage", "ordinal-aps", alpha=alpha,
                     coverage_ci=list(coverage_confidence_interval(record.n, alpha)))
        return {"type": "interval", "point_estimate": point, "interval": [lo, hi],
                "levels": [levels[lo], levels[hi]], "legend": legend, "guarantee": card}
    except Heuristic as reason:
        return {"type": "interval", "point_estimate": point, "interval": [0, len(levels) - 1],
                "levels": [levels[0], levels[-1]], "legend": legend,
                "guarantee": heuristic_card(record.name, str(reason), record.n)}


# ---------------------------------------------------------------------------
# Claim (conformal factuality)
# ---------------------------------------------------------------------------


def claim_evidence(backend, query, context, sample_count, usage, resolve=None, claims=None) -> dict[str, Any]:
    texts = claims if claims is not None else scoring.decompose_claims(backend, context, usage)
    return {"claims": [{"text": text, "score": scoring.claim_support(backend, context, text, sample_count, usage)}
                       for text in texts]}


def claim_answer(query, record: ProfileRecord, evidence, group, warnings) -> dict[str, Any]:
    claims = evidence["claims"]
    alpha = query.alpha if query.alpha is not None else DEFAULT_ALPHA
    try:
        _require_n(record.n, minimum_calibration_size(alpha), f"alpha={alpha}")
        # r_i: the highest support score among the example's false claims. Keeping
        # only claims scored above the conformal quantile of r keeps every false
        # claim out with probability >= 1 - alpha (Mohri & Hashimoto, ICML 2024).
        r = []
        for rec in record.records:
            false_scores = [c["score"] for c in rec["evidence"]["claims"] if not c.get("supported")]
            r.append(max(false_scores) if false_scores else -1.0)
        tau = conformal_quantile(np.array(r), alpha)
        retained = [c["text"] for c in claims if c["score"] > tau]
        dropped = [{"text": c["text"], "reason": "below calibrated support threshold", "score": c["score"]}
                   for c in claims if c["score"] <= tau]
        card = _card(record, "coverage", "conformal-factuality", alpha=alpha, statement=(
            f"With probability at least {1 - alpha:.0%}, every retained claim is supported "
            f"(conformal factuality, n={record.n})."))
        return {"type": "claim", "retained_claims": retained, "dropped_claims": dropped,
                "threshold": None if math.isinf(tau) else tau, "guarantee": card}
    except Heuristic as reason:
        return {"type": "claim", "retained_claims": [],
                "dropped_claims": [{"text": c["text"], "reason": "uncalibrated", "score": c["score"]} for c in claims],
                "guarantee": heuristic_card(record.name, str(reason), record.n)}


# ---------------------------------------------------------------------------
# Judge (trust or escalate)
# ---------------------------------------------------------------------------


def judge_stages(query) -> list[Optional[str]]:
    """Backend names per stage (``None`` = the default backend); human_queue ends the cascade."""
    if not query.cascade:
        return [None]
    stages = []
    for stage in query.cascade:
        name = stage.get("backend") if isinstance(stage, dict) else stage
        if name == "human_queue":
            break
        stages.append(name)
    if not stages:
        raise ConfigurationError("Judge cascade needs at least one model stage before 'human_queue'")
    return stages


def judge_evidence(backend, query, context, sample_count, usage, resolve: Resolver = None, upto=None) -> dict[str, Any]:
    out = []
    for i, name in enumerate(judge_stages(query)):
        if upto is not None and i > upto:
            break
        stage_backend = backend if name is None else resolve(name)
        probs = scoring.option_probabilities(stage_backend, context, query.instructions,
                                             scoring.JUDGE_OPTIONS, sample_count, usage)
        out.append([probs["response_a"], probs["response_b"]])
    return {"stages": out}


def judge_thresholds(query, record: ProfileRecord) -> tuple[list[float], float, float]:
    alpha = query.alpha if query.alpha is not None else DEFAULT_ALPHA
    delta = DEFAULT_DELTA
    stages = judge_stages(query)
    _require_n(record.n, math.ceil(math.log(len(LTT_STARTS) * len(stages) / delta) / alpha), f"alpha={alpha}")
    labels = _labels_index(record.records, list(scoring.JUDGE_OPTIONS))
    remaining = np.ones(record.n, dtype=bool)
    thresholds = []
    for s in range(len(stages)):
        probs = np.array([r["evidence"]["stages"][s] for r in record.records], dtype=float)
        conf = probs.max(axis=1)
        agree = (probs.argmax(axis=1) == labels).astype(float)
        lam = _selective_threshold(conf[remaining], agree[remaining], alpha, delta / len(stages))
        thresholds.append(lam)
        remaining &= ~(conf >= lam)
    return thresholds, alpha, delta


# ---------------------------------------------------------------------------
# Route (conformal cascade)
# ---------------------------------------------------------------------------


def route_tiers(query) -> list[dict[str, Any]]:
    tiers = []
    for stage in query.cascade:
        stage = stage if isinstance(stage, dict) else {"backend": stage}
        if stage.get("backend") == "human_queue":
            break
        tiers.append(stage)
    return tiers


def route_task(query):
    task = getattr(query, "task", None)
    from cli_sdk.queries import Set

    if not isinstance(task, Set):
        raise ConfigurationError("local Route needs task=Set(...): the decision each tier makes")
    if query.guarantee != "accuracy":
        raise ConfigurationError("local Route supports guarantee='accuracy' (error rate at most alpha)")
    return task


def route_evidence(backend, query, context, sample_count, usage, resolve: Resolver = None) -> dict[str, Any]:
    task = route_task(query)
    tiers = []
    for tier in route_tiers(query):
        probs = scoring.option_probabilities(resolve(tier["backend"]), context, task.instructions,
                                             task.options, sample_count, usage)
        tiers.append([probs[k] for k in task.options])
    return {"tiers": tiers}


def route_thresholds(query, record: ProfileRecord) -> tuple[list[float], float, str]:
    task = route_task(query)
    tiers = route_tiers(query)
    alpha = query.alpha if query.alpha is not None else DEFAULT_ALPHA
    per_tier = alpha / len(tiers)
    method = task.method or "LAC"
    _require_n(record.n, minimum_calibration_size(per_tier), f"alpha={alpha} across {len(tiers)} tiers")
    labels = _labels_index(record.records, list(task.options))
    q_hats = []
    for t in range(len(tiers)):
        cal = np.array([r["evidence"]["tiers"][t] for r in record.records], dtype=float)
        q_hats.append(conformal_quantile(_set_scores(method, cal, labels), per_tier))
    return q_hats, alpha, method


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

EVIDENCE = {
    "set": set_evidence,
    "belief": binary_evidence,
    "gate": binary_evidence,
    "interval": interval_evidence,
    "claim": claim_evidence,
    "judge": judge_evidence,
    "route": route_evidence,
}

ANSWER = {
    "set": set_answer,
    "belief": belief_answer,
    "gate": gate_answer,
    "interval": interval_answer,
    "claim": claim_answer,
}

DEFAULT_METHODS = {
    "set": "APS", "belief": "IVAP", "gate": "CRC", "interval": "ordinal-aps",
    "claim": "conformal-factuality", "judge": "trust-or-escalate", "route": "calibrated-cascade",
}

# A Gate's method follows its guarantee (see gate_answer), not a fixed default.
GATE_METHODS = {"risk": "CRC", "risk_high_probability": "RCPS", "fdr": "LTT-selective"}


def profile_method(query_type: str, query_payload: dict[str, Any], stored: Optional[str] = None) -> str:
    """The method a profile's answers are computed with, for reporting on ``CalibrationProfile``."""
    if query_type == "gate":
        return GATE_METHODS.get(query_payload.get("guarantee") or "risk", "CRC")
    return stored or DEFAULT_METHODS.get(query_type, "")
