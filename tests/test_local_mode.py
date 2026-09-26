"""Local mode: calibration, guarantees, and fail-closed behavior of ``LocalCLIClient``.

Every test runs fully offline against ``MockEvidenceBackend`` with a custom
scorer, so the statistical checks exercise the same engine a real model
backend would, with a known data-generating process.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from cli_sdk import Belief, Claim, Gate, InsufficientCalibrationError, Interval, Judge, Route, Set
from cli_sdk.evidence import MockEvidenceBackend
from cli_sdk.exceptions import ConfigurationError
from cli_sdk.local import LocalCLIClient, engine
from cli_sdk.stats.conformal import crc, rcps
from cli_sdk.stats.venn_abers import ivap

OPTIONS = {"billing": None, "technical": None, "account": None}


def _noisy_scorer(seed: int, strength: float = 2.0):
    """A fallible 'model': favors the true label (stored in the context) with noise."""
    rng = random.Random(seed)

    def scorer(context, instructions, options):
        weights = {}
        for key in options:
            bonus = strength if key == context["truth"] else 0.0
            weights[key] = float(np.exp(bonus + rng.gauss(0.0, 1.0)))
        return weights

    return scorer


def _set_examples(rng: random.Random, n: int, group_choices=("en",)):
    out = []
    for i in range(n):
        truth = rng.choice(list(OPTIONS))
        out.append({"context": {"truth": truth, "id": i, "language": rng.choice(group_choices)}, "label": truth})
    return out


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["LAC", "APS", "RAPS"])
def test_set_coverage_meets_target(tmp_path, method):
    alpha = 0.1
    rng = random.Random(0)
    covered = 0
    total = 0
    for trial in range(4):
        client = LocalCLIClient(MockEvidenceBackend(scorer=_noisy_scorer(trial)), store=tmp_path / f"{method}{trial}")
        query = Set(instructions="Route the ticket", options=OPTIONS, calibration_profile="routing",
                    alpha=alpha, method=method)
        client.calibrate(query, _set_examples(rng, 200))
        for example in _set_examples(rng, 150):
            answer = client.evaluate(example["context"], {"q": query}).answers["q"]
            covered += example["label"] in answer.set
            total += 1
    rate = covered / total
    # Marginal coverage >= 1 - alpha; allow three standard errors of slack.
    assert rate >= 1 - alpha - 3 * np.sqrt(alpha * (1 - alpha) / total)


def test_set_answer_carries_guarantee_and_venn_abers(tmp_path):
    client = LocalCLIClient(MockEvidenceBackend(scorer=_noisy_scorer(1)), store=tmp_path)
    query = Set(instructions="Route", options=OPTIONS, calibration_profile="routing", alpha=0.1)
    profile = client.calibrate(query, _set_examples(random.Random(1), 120))
    assert profile.n == 120 and profile.status == "serving"
    answer = client.evaluate({"truth": "billing", "id": -1, "language": "en"}, {"q": query}).answers["q"]
    assert not answer.is_heuristic
    assert answer.guarantee.type == "coverage" and answer.guarantee.alpha == 0.1
    assert answer.guarantee.calibration_n == 120
    assert set(answer.venn_abers) == set(OPTIONS)
    for p0, p1 in answer.venn_abers.values():
        assert 0.0 <= p0 <= p1 <= 1.0


def test_mondrian_reports_underpowered_groups(tmp_path):
    client = LocalCLIClient(MockEvidenceBackend(scorer=_noisy_scorer(2)), store=tmp_path)
    query = Set(instructions="Route", options=OPTIONS, calibration_profile="routing-by-lang", alpha=0.1,
                group_by="language")
    rng = random.Random(2)
    examples = _set_examples(rng, 150, ("en",)) + _set_examples(rng, 4, ("de",))
    profile = client.calibrate(query, examples)
    assert profile.groups["en"].status == "calibrated"
    assert profile.groups["de"].status == "underpowered"
    response = client.evaluate({"truth": "billing", "id": -1, "language": "de"}, {"q": query})
    assert any("marginal threshold" in w for w in response.warnings)
    calibrated = client.evaluate({"truth": "billing", "id": -2, "language": "en"}, {"q": query})
    assert calibrated.answers["q"].guarantee.calibration_n == 150


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def test_missing_profile_fails_closed(tmp_path):
    client = LocalCLIClient(MockEvidenceBackend(scorer=_noisy_scorer(3)), store=tmp_path)
    s = Set(instructions="Route", options=OPTIONS, calibration_profile="nope")
    g = Gate(instructions="Approve?", calibration_profile="nope", guarantee="risk", target=0.05)
    b = Belief(instructions="True?", calibration_profile="nope")
    ctx = {"truth": "billing"}
    answers = client.evaluate(ctx, {"s": s, "g": g, "b": b}).answers
    assert answers["s"].is_heuristic and answers["s"].set == list(OPTIONS)
    assert answers["g"].is_heuristic and answers["g"].decision == "escalate" and not answers["g"].approved
    assert answers["b"].is_heuristic and answers["b"].venn_abers == (0.0, 1.0)


def test_too_few_examples_fails_closed(tmp_path):
    client = LocalCLIClient(MockEvidenceBackend(scorer=_noisy_scorer(4)), store=tmp_path)
    query = Set(instructions="Route", options=OPTIONS, calibration_profile="tiny", alpha=0.05)
    client.calibrate(query, _set_examples(random.Random(4), 10))  # needs 19 for alpha=0.05
    answer = client.evaluate({"truth": "billing"}, {"q": query}).answers["q"]
    assert answer.is_heuristic and answer.set == list(OPTIONS)
    assert "19" in answer.guarantee.statement


def test_backend_change_invalidates_profile(tmp_path):
    query = Set(instructions="Route", options=OPTIONS, calibration_profile="routing", alpha=0.1)
    LocalCLIClient(MockEvidenceBackend(scorer=_noisy_scorer(5), seed=1), store=tmp_path).calibrate(
        query, _set_examples(random.Random(5), 60))
    other = LocalCLIClient(MockEvidenceBackend(scorer=_noisy_scorer(5), seed=2), store=tmp_path)
    response = other.evaluate({"truth": "billing"}, {"q": query})
    assert response.answers["q"].is_heuristic
    assert any("different backend" in w for w in response.warnings)


def test_prompt_change_invalidates_profile(tmp_path):
    client = LocalCLIClient(MockEvidenceBackend(scorer=_noisy_scorer(6)), store=tmp_path)
    original = Set(instructions="Route", options=OPTIONS, calibration_profile="routing", alpha=0.1)
    client.calibrate(original, _set_examples(random.Random(6), 60))
    edited = Set(instructions="Route this support ticket", options=OPTIONS, calibration_profile="routing", alpha=0.1)
    answer = client.evaluate({"truth": "billing"}, {"q": edited}).answers["q"]
    assert answer.is_heuristic
    # Changing alpha is not a scoring change: the stored evidence still applies.
    relaxed = Set(instructions="Route", options=OPTIONS, calibration_profile="routing", alpha=0.2)
    assert not client.evaluate({"truth": "billing"}, {"q": relaxed}).answers["q"].is_heuristic


def test_strict_mode_raises(tmp_path):
    client = LocalCLIClient(MockEvidenceBackend(scorer=_noisy_scorer(7)), store=tmp_path, strict_guarantees=True)
    gate = Gate(instructions="Approve?", calibration_profile="absent", guarantee="risk", target=0.1)
    with pytest.raises(InsufficientCalibrationError):
        client.evaluate({}, {"g": gate})


def test_profile_file_is_reviewable_json_and_append_checks_fingerprints(tmp_path):
    backend = MockEvidenceBackend(scorer=_noisy_scorer(8))
    client = LocalCLIClient(backend, store=tmp_path)
    query = Set(instructions="Route", options=OPTIONS, calibration_profile="routing", alpha=0.1)
    client.calibrate(query, _set_examples(random.Random(8), 30))
    data = json.loads((tmp_path / "routing.json").read_text())
    assert data["n"] == 30 and data["query"]["instructions"] == "Route"
    assert data["backend_fingerprint"]["provider"] == "mock"
    assert client.add_examples(query, _set_examples(random.Random(9), 10)).n == 40
    changed = LocalCLIClient(MockEvidenceBackend(scorer=_noisy_scorer(8), seed=5), store=tmp_path)
    with pytest.raises(ConfigurationError):
        changed.add_examples(query, _set_examples(random.Random(10), 5))
    assert [p.name for p in client.list_profiles()] == ["routing"]


def test_invalid_profile_name_is_rejected(tmp_path):
    client = LocalCLIClient(MockEvidenceBackend(), store=tmp_path)
    query = Set(instructions="Route", options=OPTIONS, calibration_profile="../escape", alpha=0.1)
    with pytest.raises(ValueError):
        client.calibrate(query, _set_examples(random.Random(0), 5))


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _gate_world(seed: int):
    """Scores are noisy estimates of the true probability that approving is correct."""
    rng = random.Random(seed)

    def draw():
        p = rng.betavariate(2.0, 0.7)
        return {"p": p, "noise": rng.gauss(0, 0.08)}, rng.random() < p

    def scorer(context, instructions, options):
        s = min(max(context["p"] + context["noise"], 0.001), 0.999)
        return {"true": s, "false": 1 - s}

    return draw, scorer


def test_gate_crc_controls_joint_risk(tmp_path):
    target = 0.05
    losses = []
    for trial in range(4):
        draw, scorer = _gate_world(trial)
        client = LocalCLIClient(MockEvidenceBackend(scorer=scorer), store=tmp_path / str(trial))
        gate = Gate(instructions="Approve?", calibration_profile="g", guarantee="risk", target=target)
        client.calibrate(gate, [dict(zip(("context", "label"), draw())) for _ in range(300)])
        for _ in range(200):
            context, correct = draw()
            answer = client.evaluate(context, {"g": gate}).answers["g"]
            losses.append(answer.approved and not correct)
    rate = float(np.mean(losses))
    assert rate <= target + 3 * np.sqrt(target * (1 - target) / len(losses))


def test_gate_modes_produce_their_guarantee_cards(tmp_path):
    draw, scorer = _gate_world(11)
    client = LocalCLIClient(MockEvidenceBackend(scorer=scorer), store=tmp_path)
    examples = [dict(zip(("context", "label"), draw())) for _ in range(600)]
    for guarantee, extra, method in [("risk", {}, "CRC"), ("risk_high_probability", {"delta": 0.1}, "RCPS"),
                                     ("fdr", {"delta": 0.1}, "LTT-selective")]:
        gate = Gate(instructions="Approve?", calibration_profile=f"g-{guarantee}", guarantee=guarantee,
                    target=0.1, **extra)
        client.calibrate(gate, examples)
        answer = client.evaluate({"p": 0.995, "noise": 0.0}, {"g": gate}).answers["g"]
        assert answer.guarantee.type == guarantee and answer.guarantee.method == method
        assert answer.approved, guarantee
        assert answer.guarantee.describe()  # a real sentence, not the heuristic fallback
        assert "Heuristic" not in answer.guarantee.describe()


def test_gate_batch_controls_fdr(tmp_path):
    target = 0.1
    fdps = []
    for trial in range(6):
        draw, scorer = _gate_world(100 + trial)
        client = LocalCLIClient(MockEvidenceBackend(scorer=scorer), store=tmp_path / str(trial))
        gate = Gate(instructions="Duplicate?", calibration_profile="b", guarantee="fdr", target=target)
        client.calibrate(gate, [dict(zip(("context", "label"), draw())) for _ in range(300)])
        batch = [draw() for _ in range(60)]
        answers = client.gate_batch([c for c, _ in batch], gate)
        approved = [ok for (c, ok), a in zip(batch, answers) if a.approved]
        fdps.append(0.0 if not approved else sum(not ok for ok in approved) / len(approved))
        assert all(a.guarantee.method == "conformal-selection-BH" for a in answers)
    assert np.mean(fdps) <= target + 0.08


# ---------------------------------------------------------------------------
# Belief, Interval, Claim
# ---------------------------------------------------------------------------


def test_belief_venn_abers_interval_brackets_rate(tmp_path):
    rng = random.Random(12)

    def scorer(context, instructions, options):
        return {"true": context["score"], "false": 1 - context["score"]}

    client = LocalCLIClient(MockEvidenceBackend(scorer=scorer), store=tmp_path)
    belief = Belief(instructions="Readmitted within 30 days?", calibration_profile="readmit")
    examples = []
    for _ in range(400):
        score = rng.choice([0.2, 0.5, 0.8])
        examples.append({"context": {"score": score}, "label": rng.random() < score ** 2})
    client.calibrate(belief, examples)
    answer = client.evaluate({"score": 0.8}, {"b": belief}).answers["b"]
    p0, p1 = answer.venn_abers
    assert p0 <= 0.64 + 0.1 and p1 >= 0.64 - 0.1
    assert p1 - p0 < 0.1
    assert answer.guarantee.method == "IVAP"


def test_interval_is_contiguous_and_covers(tmp_path):
    levels = ["none", "mild", "moderate", "severe", "critical"]
    rng = random.Random(13)

    def scorer(context, instructions, options):
        return {lvl: float(np.exp(-abs(i - context["severity"] - context["noise"]) * 1.5))
                for i, lvl in enumerate(options)}

    client = LocalCLIClient(MockEvidenceBackend(scorer=scorer), store=tmp_path)
    query = Interval(instructions="Acuity", levels=levels, calibration_profile="acuity", alpha=0.1)

    def example():
        s = rng.randrange(5)
        return {"context": {"severity": s, "noise": rng.gauss(0, 0.8)}, "label": levels[s]}

    client.calibrate(query, [example() for _ in range(250)])
    hits = []
    for _ in range(250):
        ex = example()
        answer = client.evaluate(ex["context"], {"i": query}).answers["i"]
        lo, hi = answer.interval
        assert lo <= hi
        hits.append(lo <= levels.index(ex["label"]) <= hi)
    assert np.mean(hits) >= 0.9 - 3 * np.sqrt(0.09 / 250)


def test_claim_filter_keeps_only_supported_claims_with_high_probability(tmp_path):
    rng = random.Random(14)

    def scorer(context, instructions, options):
        claim = instructions.split("Claim:", 1)[1].strip()
        supported = claim in context.get("chart", "")
        p = min(max((0.75 if supported else 0.3) + rng.gauss(0, 0.12), 0.01), 0.99)
        return {"true": p, "false": 1 - p}

    facts = [f"fact number {i} holds" for i in range(12)]
    fakes = [f"invented detail {i}" for i in range(12)]

    def example():
        true_claims = rng.sample(facts, 3)
        false_claims = rng.sample(fakes, 1)
        context = {"chart": ". ".join(true_claims), "answer": ". ".join(true_claims + false_claims)}
        label = [{"text": c, "supported": True} for c in true_claims] + \
                [{"text": c, "supported": False} for c in false_claims]
        return {"context": context, "label": label}

    client = LocalCLIClient(MockEvidenceBackend(scorer=scorer), store=tmp_path)
    query = Claim(instructions="Verify the summary against the chart", calibration_profile="claims", alpha=0.1)
    client.calibrate(query, [example() for _ in range(150)])
    audit = client.audit(query, [example() for _ in range(150)])
    assert audit.result == "pass"
    assert audit.realized_coverage >= 0.9 - 3 * np.sqrt(0.09 / 150)


# ---------------------------------------------------------------------------
# Judge and Route
# ---------------------------------------------------------------------------


def _judge_scorer(noise: float, seed: int):
    rng = random.Random(seed)

    def scorer(context, instructions, options):
        p = min(max(context["margin"] + rng.gauss(0, noise), 0.01), 0.99)
        return {"response_a": p, "response_b": 1 - p}

    return scorer


def test_judge_cascade_escalates_and_keeps_agreement(tmp_path):
    rng = random.Random(15)
    small = MockEvidenceBackend(scorer=_judge_scorer(0.25, 1), name="small")
    large = MockEvidenceBackend(scorer=_judge_scorer(0.08, 2), name="large")
    client = LocalCLIClient(small, backends={"small": small, "large": large}, store=tmp_path)
    judge = Judge(instructions="Which answer is more accurate?", calibration_profile="judge", alpha=0.2,
                  cascade=[{"backend": "small"}, {"backend": "large"}, {"backend": "human_queue"}])

    def example():
        m = rng.choice([0.03, 0.1, 0.2, 0.45, 0.55, 0.8, 0.9, 0.97])
        return {"context": {"margin": m}, "label": "response_a" if rng.random() < m else "response_b"}

    client.calibrate(judge, [example() for _ in range(800)])
    decided, agree, humans = 0, 0, 0
    for _ in range(300):
        ex = example()
        answer = client.evaluate(ex["context"], {"j": judge}).answers["j"]
        if answer.needs_human:
            humans += 1
        else:
            decided += 1
            agree += answer.winner == ex["label"]
    assert decided > 0 and humans > 0
    assert agree / decided >= 0.8 - 0.07


def test_route_serves_cheap_tier_when_confident(tmp_path):
    rng = random.Random(16)
    options = {"approve": None, "deny": None, "review": None}

    def tier_scorer(noise):
        local = random.Random(noise)

        def scorer(context, instructions, opts):
            return {k: (4.0 if k == context["truth"] else 1.0) * local.uniform(1 - noise, 1 + noise) for k in opts}

        return scorer

    cheap = MockEvidenceBackend(scorer=tier_scorer(0.9), name="cheap")
    strong = MockEvidenceBackend(scorer=tier_scorer(0.2), name="strong")
    client = LocalCLIClient(cheap, backends={"gemma": cheap, "frontier": strong}, store=tmp_path)
    task = Set(instructions="Decide the claim", options=options, calibration_profile="task", method="LAC")
    route = Route(cascade=[{"backend": "gemma", "cost_cents": 0.01}, {"backend": "frontier", "cost_cents": 0.4}],
                  calibration_profile="route", guarantee="accuracy", alpha=0.1, task=task)
    examples = [{"context": {"truth": (t := rng.choice(list(options)))}, "label": t} for _ in range(300)]
    client.calibrate(route, examples)
    served = [client.evaluate({"truth": "deny"}, {"r": route}).answers["r"] for _ in range(30)]
    tiers = {a.served_by["backend"] for a in served}
    assert "gemma" in tiers
    wrong = [a for a in served if a.output not in (None, "deny")]
    assert len(wrong) <= 3
    assert served[0].guarantee.method == "conformal-cascade"


def test_route_requires_task_and_accuracy(tmp_path):
    client = LocalCLIClient(MockEvidenceBackend(), backends={"a": MockEvidenceBackend()}, store=tmp_path)
    route = Route(cascade=[{"backend": "a"}], calibration_profile="r", guarantee="accuracy", alpha=0.1)
    with pytest.raises(ConfigurationError):
        client.calibrate(route, [{"context": {}, "label": "x"}])


# ---------------------------------------------------------------------------
# Statistics regressions fixed alongside local mode
# ---------------------------------------------------------------------------


def test_crc_uses_joint_risk():
    # 10 accepted points at the top, 1 of them wrong, plus 90 rejected points:
    # the joint risk at lambda=0.9 is 1/100, well inside a 0.05 target, even
    # though the error rate among accepted points is 10%.
    scores = np.array([0.95] * 10 + [0.1] * 90)
    losses = np.array([1.0] + [0.0] * 9 + [1.0] * 90)
    assert crc.calibrate(scores, losses, alpha=0.05) == pytest.approx(0.95)


def test_rcps_does_not_stop_at_the_empty_acceptance_set():
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=2000)
    losses = (rng.uniform(size=2000) > scores).astype(float)  # wrong less often at high scores
    lam = rcps.calibrate(scores, losses, alpha=0.05, delta=0.1)
    assert np.isfinite(lam)
    assert np.mean(losses * (scores >= lam)) <= 0.05


def test_ivap_pools_tied_scores():
    # Two distinct scores with many ties: the fitted probability at a score
    # must be (close to) the label frequency among its ties, whatever the
    # order of the tied points.
    cal_scores = np.array([0.1] * 100 + [0.9] * 100)
    cal_labels = np.array([1.0] * 5 + [0.0] * 95 + [1.0] * 90 + [0.0] * 10)
    p0, p1 = ivap.calibrate_and_predict(cal_scores, cal_labels, np.array([0.1, 0.9]))
    assert 0.03 <= p0[0] <= p1[0] <= 0.07
    assert 0.88 <= p0[1] <= p1[1] <= 0.92


def test_selective_threshold_is_finite_when_top_scores_are_clean():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.5, 1.0, 1000)
    correct = (rng.uniform(size=1000) < conf).astype(float)
    lam = engine._selective_threshold(conf, correct, alpha=0.1, delta=0.1)
    assert np.isfinite(lam)
    accepted = conf >= lam
    assert 1 - correct[accepted].mean() <= 0.1
