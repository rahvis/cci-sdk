"""Offline calibration with the standalone statistics engine: no network, no mock (PRD 8.3, 11.6).

Everything below is real computation with ``cli_sdk.stats`` on NumPy arrays.
Sockets are disabled for the whole run, and the HTTP client is never
imported, to show the engine works inside an air-gapped environment.

The data is a synthetic 5-way intent classifier whose softmax is
deliberately overconfident (the common real-world failure). Because the
data is synthetic, the true labels of "production" items are known, so every
guarantee can be checked against what actually happened on held-out data.

  1. The classifier and its miscalibration
  2. APS prediction sets: empirical coverage vs the 90% target
  3. LAC vs APS: set size, and where each spends it
  4. Venn-Abers (IVAP) intervals for "is the top answer correct?"
  5. Gate-style accept thresholds: CRC and a high-probability LTT threshold
  6. Batch FDR control with conformal e-values and e-BH
  7. An anytime-valid coverage monitor: quiet in spec, fires after drift
  8. Prediction-powered inference: a label-efficient accuracy estimate

Every guarantee here is marginal over data exchangeable with the
calibration set; none is a statement about one particular item.

Run:

    python examples/offline_calibration_no_network.py
"""

from __future__ import annotations

import socket
import sys


def _disable_network() -> None:
    """Make any attempt to open a connection fail loudly for this process."""

    def blocked(*args, **kwargs):
        raise RuntimeError("offline example attempted network access")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked  # type: ignore[method-assign]
    socket.create_connection = blocked  # type: ignore[assignment]
    socket.getaddrinfo = blocked  # type: ignore[assignment]


_disable_network()

import numpy as np

from cli_sdk.calibration.audit import audit_coverage
from cli_sdk.calibration.label_efficient import judge_quality
from cli_sdk.stats.conformal import aps, coverage_confidence_interval, crc, lac, ltt
from cli_sdk.stats.evalues import CoverageMonitor, ebh, ppi
from cli_sdk.stats.evalues.anytime import normal_ppf
from cli_sdk.stats.venn_abers import interval_width, ivap, merge_to_probability

INTENTS = ["billing", "technical", "sales", "account", "legal"]
K = len(INTENTS)
ALPHA = 0.10
SEED = 7


def simulate(n: int, rng: np.random.Generator, confusion: float = 0.0):
    """Probabilities, true labels, and per-item clarity for n items.

    ``clarity`` is how strongly an item's features point at its label; the
    temperature of 3 makes the softmax overconfident. ``confusion`` is the
    drift knob: that fraction of items has its evidence pointing at a wrong
    class, as after a silent model or prompt change.
    """
    labels = rng.integers(0, K, n)
    clarity = rng.uniform(0.2, 3.0, n)
    signal_to = labels.copy()
    confused = rng.uniform(size=n) < confusion
    signal_to[confused] = (labels[confused] + rng.integers(1, K, confused.sum())) % K
    logits = rng.normal(0.0, 1.0, (n, K))
    logits[np.arange(n), signal_to] += clarity
    probs = np.exp(3.0 * logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, labels, clarity


def coverage_of(sets: list[np.ndarray], labels: np.ndarray) -> np.ndarray:
    return np.array([label in s for label, s in zip(labels, sets)])


def sizes_of(sets: list[np.ndarray]) -> np.ndarray:
    return np.array([len(s) for s in sets])


def header(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def main() -> None:
    rng = np.random.default_rng(SEED)
    cal_probs, cal_labels, _ = simulate(2000, rng)
    test_probs, test_labels, test_clarity = simulate(5000, rng)
    cal_conf, test_conf = cal_probs.max(axis=1), test_probs.max(axis=1)
    cal_wrong = (cal_probs.argmax(axis=1) != cal_labels).astype(float)
    test_wrong = (test_probs.argmax(axis=1) != test_labels).astype(float)

    # 1 ----------------------------------------------------------------------
    header("1. The classifier and its miscalibration")
    print(f"calibration set n={len(cal_labels)}, held-out test set n={len(test_labels)}, {K} intents")
    print(f"top-1 accuracy {1 - test_wrong.mean():.3f}, but mean top-1 confidence {test_conf.mean():.3f}")
    high = test_conf >= 0.9
    print(f"items the model is >= 90% sure of: {high.mean():.0%} of traffic, "
          f"actually correct {1 - test_wrong[high].mean():.1%} of the time")

    # 2 ----------------------------------------------------------------------
    header(f"2. APS prediction sets (target coverage {1 - ALPHA:.0%})")
    q_aps = aps.calibrate(cal_probs, cal_labels, alpha=ALPHA, seed=0)
    aps_sets = aps.predict(test_probs, q_aps)
    aps_covered = coverage_of(aps_sets, test_labels)
    aps_sizes = sizes_of(aps_sets)
    lo, hi = coverage_confidence_interval(len(cal_labels), ALPHA)
    print(f"q_hat = {q_aps:.4f} from {len(cal_labels)} calibration examples")
    print(f"empirical coverage on held-out data: {aps_covered.mean():.3f} (target >= {1 - ALPHA:.2f})")
    print(f"coverage interval implied by n={len(cal_labels)}: [{lo:.3f}, {hi:.3f}]; "
          "deterministic APS sits at or above it")
    counts = np.bincount(aps_sizes, minlength=K + 1)[1:]
    print("set sizes: " + ", ".join(f"{k}: {c / len(aps_sizes):.0%}" for k, c in enumerate(counts, start=1)))
    audit = audit_coverage(aps_covered[:300].tolist(), target=1 - ALPHA)
    print(f"audit on 300 fresh labels (the check 'cli calibration audit-local' runs): {audit.result}, "
          f"coverage {audit.realized_coverage:.3f}, 95% CI [{audit.ci_lower:.3f}, {audit.ci_upper:.3f}]")
    print("  (an audit fails only when the whole interval sits below the target, i.e. on real evidence)")

    order = np.argsort(-test_probs, axis=1)
    mass = np.cumsum(np.take_along_axis(test_probs, order, axis=1), axis=1)
    naive_k = (mass < 1 - ALPHA).sum(axis=1) + 1
    naive_covered = np.array([test_labels[i] in order[i, : naive_k[i]] for i in range(len(test_labels))])
    print(f"uncalibrated alternative, 'take classes until 90% of the raw probability mass': "
          f"coverage {naive_covered.mean():.3f}, mean size {naive_k.mean():.2f}")

    # 3 ----------------------------------------------------------------------
    header("3. LAC vs APS")
    q_lac = lac.calibrate(cal_probs, cal_labels, alpha=ALPHA)
    lac_sets = lac.predict(test_probs, q_lac)
    lac_covered, lac_sizes = coverage_of(lac_sets, test_labels), sizes_of(lac_sets)
    easy = test_clarity >= np.quantile(test_clarity, 2 / 3)
    hard = test_clarity <= np.quantile(test_clarity, 1 / 3)
    print(f"{'method':<6} {'coverage':>9} {'mean size':>10} {'empty':>6} {'size easy':>10} "
          f"{'size hard':>10} {'coverage hard':>14}")
    for name, covered, sizes in (("LAC", lac_covered, lac_sizes), ("APS", aps_covered, aps_sizes)):
        print(f"{name:<6} {covered.mean():>9.3f} {sizes.mean():>10.2f} {(sizes == 0).mean():>6.1%} "
              f"{sizes[easy].mean():>10.2f} {sizes[hard].mean():>10.2f} {covered[hard].mean():>14.3f}")
    print("LAC gives the smallest sets on average; APS spends more of its size on hard items,")
    print("so coverage on the hardest third holds up better (neither guarantees coverage per difficulty).")
    print(f"Coverage for one fixed calibration set scatters around the target (the [{lo:.3f}, {hi:.3f}]")
    print("interval above, plus test-sample noise); the 90% guarantee is on average over calibration draws.")

    # 4 ----------------------------------------------------------------------
    header("4. Venn-Abers intervals for 'is the top answer correct?' (a Belief)")
    va_test = slice(0, 500)
    p0, p1 = ivap.calibrate_and_predict(cal_conf, 1 - cal_wrong, test_conf[va_test])
    merged = merge_to_probability(p0, p1)
    truth = 1 - test_wrong[va_test]
    print(f"{'raw confidence':<16} {'items':>6} {'mean raw':>9} {'mean Venn-Abers':>16} {'actual accuracy':>16}")
    for low, high_ in ((0.0, 0.7), (0.7, 0.9), (0.9, 1.01)):
        in_bin = (test_conf[va_test] >= low) & (test_conf[va_test] < high_)
        label = f"[{low:.1f}, {min(high_, 1.0):.1f}{']' if high_ > 1 else ')'}"
        print(f"{label:<16} {in_bin.sum():>6} {test_conf[va_test][in_bin].mean():>9.3f} "
              f"{merged[in_bin].mean():>16.3f} {truth[in_bin].mean():>16.3f}")
    print("\nper-item intervals [p0, p1] (width is an ambiguity signal on its own):")
    for i in np.argsort(test_conf[va_test])[[25, 150, 300, 420, 490]]:
        print(f"  raw {test_conf[i]:.3f} -> [{p0[i]:.3f}, {p1[i]:.3f}], width {p1[i] - p0[i]:.3f}")
    straddle = (p0 < 0.8) & (p1 > 0.8)
    print(f"items whose interval straddles an 0.8 approval bar (escalate these): {straddle.sum()} of {len(p0)}")
    print("\nmean interval width shrinks as calibration data grows:")
    probe = test_conf[:150]
    for n in (30, 300, 2000):
        w0, w1 = ivap.calibrate_and_predict(cal_conf[:n], 1 - cal_wrong[:n], probe)
        print(f"  n={n:<5} mean width {interval_width(w0, w1).mean():.3f}")

    # 5 ----------------------------------------------------------------------
    header(f"5. Gate thresholds: approve when confidence >= threshold (alpha = {ALPHA})")
    lam_crc = crc.calibrate(cal_conf, cal_wrong, alpha=ALPHA)

    grid = np.round(np.linspace(0.50, 0.995, 45), 3)  # fixed before looking at any labels

    def selective_risk(config: dict) -> tuple[float, int]:
        accepted = cal_conf >= config["threshold"]
        n_accepted = int(accepted.sum())
        return (float(cal_wrong[accepted].mean()) if n_accepted else 0.0), n_accepted

    delta = 0.10
    valid = ltt.calibrate_grid([{"threshold": t} for t in grid[::-1]], selective_risk,
                               alpha=ALPHA, delta=delta, order="bonferroni")
    chosen = ltt.select_smallest(valid, size_fn=lambda c: c["threshold"])
    lam_ltt = chosen["threshold"] if chosen else float("inf")

    print(f"{'rule':<42} {'threshold':>9} {'approved':>9} {'approved & wrong':>17} {'wrong among approved':>21}")
    for name, lam in (('CRC  guarantee="risk"', lam_crc),
                      ('LTT  guarantee="risk_high_probability"', lam_ltt)):
        approved = test_conf >= lam
        wrong_among = test_wrong[approved].mean() if approved.any() else 0.0
        print(f"{name:<42} {lam:>9.3f} {approved.mean():>9.1%} {(test_wrong * approved).mean():>17.1%} "
              f"{wrong_among:>21.1%}")
    print(f"CRC bounds the expected share of all items that are approved AND wrong: <= {ALPHA:.0%}.")
    print(f"LTT (Hoeffding-Bentkus p-values, Bonferroni over {len(grid)} fixed thresholds) bounds the share")
    print(f"wrong among approved items at <= {ALPHA:.0%}, with probability >= {1 - delta:.0%} over the calibration draw.")
    print("Both are measured above on held-out data the thresholds never saw.")

    # 6 ----------------------------------------------------------------------
    header("6. Gate with guarantee=\"fdr\": conformal e-values and e-BH on a batch")
    q = 0.10

    def conformal_e_values(cal_scores, cal_is_null, batch_scores, level):
        """Conformal e-values for H0_j: "item j's answer is wrong".

        The threshold T is the most permissive score at which the estimated
        false-discovery proportion is <= level; e-BH on these e-values then
        approves exactly the items scoring >= T (conformal selection).
        """
        n, m = len(cal_scores), len(batch_scores)
        null_sorted = np.sort(cal_scores[cal_is_null])
        batch_sorted = np.sort(batch_scores)
        candidates = np.unique(batch_scores)
        null_above = len(null_sorted) - np.searchsorted(null_sorted, candidates, side="left")
        batch_above = m - np.searchsorted(batch_sorted, candidates, side="left")
        fdp_hat = (m / (n + 1)) * (1 + null_above) / np.maximum(batch_above, 1)
        passing = np.flatnonzero(fdp_hat <= level)
        threshold = candidates[passing[0]] if passing.size else np.inf
        return (n + 1) * (batch_scores >= threshold) / (1 + np.sum(cal_scores[cal_is_null] >= threshold))

    batch = slice(0, 200)
    e_values = conformal_e_values(cal_conf, cal_wrong == 1, test_conf[batch], q)
    selected = ebh.select(e_values, q=q)
    fdp = test_wrong[batch][selected].mean() if selected.size else 0.0
    print(f"one batch of 200: e-BH approved {selected.size}, of which {int(test_wrong[batch][selected].sum())} "
          f"wrong (false-discovery proportion {fdp:.1%})")
    if fdp > q:
        print(f"  FDR bounds the expected proportion over batches; a single batch can land above {q:.0%}, as this one does")

    fdps, approved_counts, naive_fdps = [], [], []
    rep_rng = np.random.default_rng(SEED + 1)
    for _ in range(300):
        c_probs, c_labels, _ = simulate(2000, rep_rng)
        b_probs, b_labels, _ = simulate(200, rep_rng)
        c_conf, b_conf = c_probs.max(axis=1), b_probs.max(axis=1)
        c_wrong = c_probs.argmax(axis=1) != c_labels
        b_wrong = b_probs.argmax(axis=1) != b_labels
        picked = ebh.select(conformal_e_values(c_conf, c_wrong, b_conf, q), q=q)
        fdps.append(b_wrong[picked].mean() if picked.size else 0.0)
        approved_counts.append(picked.size)
        naive = b_conf >= 0.9
        naive_fdps.append(b_wrong[naive].mean() if naive.any() else 0.0)
    print(f"over 300 fresh calibration sets and batches: mean FDP {np.mean(fdps):.1%} (target <= {q:.0%}), "
          f"mean approved {np.mean(approved_counts):.0f} of 200")
    print(f"uncalibrated rule 'approve if raw confidence >= 0.9': mean FDP {np.mean(naive_fdps):.1%}")
    print("e-BH controls FDR without assuming the decisions in a batch are independent of each other.")

    # 7 ----------------------------------------------------------------------
    header("7. Anytime-valid coverage monitor (target 0.90, false-alarm rate 0.05)")
    monitor = CoverageMonitor(target=1 - ALPHA, false_alarm_rate=0.05)
    stream_probs, stream_labels, _ = simulate(2000, rng)
    in_spec = coverage_of(aps.predict(stream_probs, q_aps), stream_labels)
    alarm = None
    for covered in in_spec:
        alarm = monitor.update(bool(covered)) or alarm
    print(f"in spec: {len(in_spec)} labelled production outcomes, coverage {in_spec.mean():.3f}, "
          f"alarm: {'yes' if alarm else 'no'} (it alarms when the e-value reaches 1/0.05 = 20)")

    drift_probs, drift_labels, _ = simulate(3000, rng, confusion=0.30)
    drifted = coverage_of(aps.predict(drift_probs, q_aps), drift_labels)
    print("injected drift: a silent model change points 30% of items at a wrong intent")
    for t, covered in enumerate(drifted, start=1):
        alert = monitor.update(bool(covered))
        if alert:
            print(f"alarm {t} outcomes after the change (observation {alert['n']}): coverage since the change "
                  f"{drifted[:t].mean():.3f}, e-value {alert['e_value']:.1f}")
            break
    else:
        print(f"no alarm within {len(drifted)} outcomes after the change")
    print("The false-alarm guarantee holds however often the monitor is checked. It tests the miss rate")
    print("accumulated since it started, so evidence banked while in spec lengthens the time to detection.")

    # 8 ----------------------------------------------------------------------
    header("8. Prediction-powered inference: production accuracy from few human labels")
    pool_probs, pool_labels, _ = simulate(10000, rng)
    pool_correct = (pool_probs.argmax(axis=1) == pool_labels).astype(float)
    # An LLM judge grades every item's top answer. It is right most of the
    # time but lenient: it passes 30% of the wrong answers.
    passes = np.where(pool_correct == 1, rng.uniform(size=pool_correct.size) < 0.92,
                      rng.uniform(size=pool_correct.size) < 0.30)
    judge = passes.astype(float)
    human_idx = rng.choice(pool_correct.size, size=200, replace=False)
    rest = np.setdiff1d(np.arange(pool_correct.size), human_idx)
    human = pool_correct[human_idx]
    z = normal_ppf(0.975)

    print(f"true top-1 accuracy on the 10,000-item pool (known because the data is synthetic): "
          f"{pool_correct.mean():.3f}")
    h_se = human.std(ddof=1) / np.sqrt(human.size)
    print(f"{'200 human labels only':<34} {human.mean():.3f}  95% CI [{human.mean() - z * h_se:.3f}, "
          f"{human.mean() + z * h_se:.3f}]  width {2 * z * h_se:.3f}")
    print(f"{'judge grades only':<34} {judge.mean():.3f}  (biased by the lenient judge: no valid interval)")
    result = ppi.estimate_mean(labelled_predictions=judge[human_idx], labelled_labels=human,
                               unlabelled_predictions=judge[rest], alpha=0.05)
    print(f"{'PPI: 200 human + 9,800 judge':<34} {result.estimate:.3f}  95% CI [{result.ci_lower:.3f}, "
          f"{result.ci_upper:.3f}]  width {result.ci_upper - result.ci_lower:.3f}")
    quality = judge_quality(human, judge[human_idx])
    print(f"judge on the human sample: agreement {quality.agreement_with_humans:.2f}, correlation "
          f"{quality.correlation:.2f}; each human label now does the work of about "
          f"{quality.effective_label_multiplier:.1f} human labels used alone")
    print("Both intervals are normal-approximation (CLT) intervals. A poor judge only widens the PPI")
    print("interval; the human labels correct its bias.")

    header("Network")
    print("sockets were disabled for this run and no connection was attempted; "
          f"HTTP client imported: {'yes' if 'httpx' in sys.modules else 'no'}")


if __name__ == "__main__":
    main()
