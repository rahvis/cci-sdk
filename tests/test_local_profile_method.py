"""Regression: a local Gate profile reports the method its answers actually use.

``LocalCLIClient`` used to record every Gate profile as ``"CRC"``, whatever
the guarantee, so ``CalibrationProfile.method`` (printed by the examples and
used in audit write-ups) said conformal risk control for RCPS and
Learn-then-Test profiles too. The method now follows the guarantee, and it
is derived from the query when the profile is read back, so it stays right
even for profiles stored before the fix.
"""

from __future__ import annotations

import random

import pytest

from cli_sdk import Belief, Gate, Set
from cli_sdk.evidence import MockEvidenceBackend
from cli_sdk.local import LocalCLIClient


def _scorer(context, instructions, options):
    if set(options) == {"true", "false"}:
        p = 0.92 if context["x"] > 0.5 else 0.1
        return {"true": p, "false": 1.0 - p}
    return {k: (3.0 if k == "a" else 1.0) for k in options}


def _examples(n: int = 200):
    rng = random.Random(7)
    out = []
    for _ in range(n):
        x = rng.random()
        out.append({"context": {"x": x}, "label": (x > 0.5) == (rng.random() < 0.97)})
    return out


@pytest.mark.parametrize("guarantee,delta,method", [("risk", None, "CRC"),
                                                    ("risk_high_probability", 0.1, "RCPS"),
                                                    ("fdr", 0.1, "LTT-selective")])
def test_gate_profile_method_follows_the_guarantee(tmp_path, guarantee, delta, method):
    client = LocalCLIClient(MockEvidenceBackend(scorer=_scorer), store=tmp_path)
    gate = Gate(instructions="Approve?", calibration_profile="g", guarantee=guarantee, target=0.1, delta=delta)

    assert client.calibrate(gate, _examples()).method == method
    assert client.get_profile("g").method == method
    answer = client.evaluate({"x": 0.9}, {"g": gate}).answers["g"]
    assert answer.guarantee.method == method, "the profile and the answer name the same method"


def test_profiles_stored_with_the_old_label_are_reported_correctly(tmp_path):
    client = LocalCLIClient(MockEvidenceBackend(scorer=_scorer), store=tmp_path)
    gate = Gate(instructions="Approve?", calibration_profile="old", guarantee="fdr", target=0.1, delta=0.1)
    client.calibrate(gate, _examples())
    record = client.store.load("old")
    record.method = "CRC"  # what earlier versions wrote for every Gate
    client.store.save(record)

    assert client.get_profile("old").method == "LTT-selective"


def test_other_primitives_keep_their_methods(tmp_path):
    client = LocalCLIClient(MockEvidenceBackend(scorer=_scorer), store=tmp_path)
    belief = Belief(instructions="True?", calibration_profile="b")
    labels = Set(instructions="Which?", calibration_profile="s", options={"a": None, "b": None},
                 alpha=0.1, method="LAC")
    assert client.calibrate(belief, _examples()).method == "IVAP"
    set_examples = [{"context": e["context"], "label": "a" if e["label"] else "b"} for e in _examples()]
    assert client.calibrate(labels, set_examples).method == "LAC"
