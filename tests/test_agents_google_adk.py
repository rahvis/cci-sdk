"""The Google ADK examples, end to end in mock mode (keyless and offline).

Skipped when ``google-adk`` is not installed, so the core suite runs without it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("google.adk")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "agents"
DATA = EXAMPLES / "data"


def _load(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def aml():
    return _load(EXAMPLES / "google_adk" / "aml_account_hold_agent.py", "adk_example_aml")


@pytest.fixture(scope="module")
def trial():
    return _load(EXAMPLES / "google_adk" / "clinical_trial_screening_agent.py", "adk_example_trial")


@pytest.fixture(scope="module")
def aml_data():
    return _load(DATA / "generate_aml_alerts.py", "adk_data_aml")


@pytest.fixture(scope="module")
def trial_data():
    return _load(DATA / "generate_trial_screening.py", "adk_data_trial")


@pytest.fixture(scope="module")
def shared_store(tmp_path_factory):
    """One calibrated store for the tests that do not check calibration itself."""
    return tmp_path_factory.mktemp("adk_profiles")


@pytest.fixture
def hermetic_env(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "AZURE_OPENAI_API_KEY",
                 "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
                 "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL", "VLLM_MODEL", "VLLM_BASE_URL",
                 "VLLM_API_KEY", "SGLANG_MODEL", "SGLANG_BASE_URL", "SGLANG_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _summary(outcome: dict[str, Any]) -> tuple:
    return outcome["action"], outcome["reviewed"], outcome["executed"], tuple(outcome["audit"])


# ---------------------------------------------------------------------------
# AML account holds: Gate(fdr) through CLIGuardPlugin on the Runner
# ---------------------------------------------------------------------------


def test_aml_scenarios_in_mock_mode(aml, tmp_path, capsys):
    result = aml.run(["--store", str(tmp_path)])
    s = result["scenarios"]

    assert _summary(s["A-5001"]) == ("allow", None, True, ("allow",))
    assert _summary(s["A-5002"]) == ("escalate", False, False, ("escalate", "human_rejected"))
    assert _summary(s["A-5003"]) == ("escalate", True, True, ("escalate", "human_approved"))
    assert [h["hold_id"] for h in result["holds"]] == ["HOLD-A-5001", "HOLD-A-5003"]
    assert "declined" in s["A-5002"]["final_text"]

    assert result["profile"] == {"name": "aml-holds-v1", "method": "LTT-selective", "n": 360,
                                 "status": "serving", "calibrated_now": True}
    out = capsys.readouterr().out
    assert "Synthetic data for demonstration only" in out
    assert "With 90% confidence, at most 0.1 of auto-approved" in out
    assert "adk_request_confirmation issued" in out


def test_aml_reviewer_decisions_control_execution(aml, shared_store):
    flipped = aml.Reviewer(script={"A-5002": True, "A-5003": False})
    s = aml.run(["--store", str(shared_store)], reviewer=flipped)["scenarios"]

    assert _summary(s["A-5001"]) == ("allow", None, True, ("allow",))
    assert _summary(s["A-5002"]) == ("escalate", True, True, ("escalate", "human_approved"))
    assert _summary(s["A-5003"]) == ("escalate", False, False, ("escalate", "human_rejected"))


def test_aml_second_run_reuses_the_profile_and_repeats_decisions(aml, tmp_path, capsys):
    first = aml.run(["--store", str(tmp_path)])
    profile_file = next(tmp_path.glob("*.json"))
    stamp = profile_file.stat().st_mtime_ns
    first_out = capsys.readouterr().out

    second = aml.run(["--store", str(tmp_path)])
    second_out = capsys.readouterr().out

    assert first["profile"]["calibrated_now"] and not second["profile"]["calibrated_now"]
    assert profile_file.stat().st_mtime_ns == stamp, "a cached profile is not rewritten"
    assert {k: _summary(v) for k, v in first["scenarios"].items()} == \
           {k: _summary(v) for k, v in second["scenarios"].items()}
    assert first_out.replace("calibrated now", "cached") == second_out


def test_aml_batch_mode_uses_conformal_selection(aml, shared_store, capsys):
    batch = aml.run(["--store", str(shared_store), "--batch"])["batch"]

    assert batch["size"] == 40
    assert 0 < len(batch["auto_held"]) < 40
    assert sorted(batch["auto_held"] + batch["escalated"]) == [f"A-{7001 + i}" for i in range(40)]
    assert "Benjamini-Hochberg" in batch["statement"] and "batch of 40" in batch["statement"]
    out = capsys.readouterr().out
    assert "auto-held:" in out and "sent to analysts:" in out


def test_aml_hold_on_the_wrong_account_escalates(aml, shared_store):
    from cli_sdk.integrations import GuardRule, ToolGuard
    from cli_sdk.local import LocalCLIClient

    aml.run(["--store", str(shared_store), "--batch"])  # ensures the profile exists
    client = LocalCLIClient(aml.providers.evidence_backend("mock", mock_scorer=aml.mock_hold_evidence),
                            store=shared_store)
    cases = aml.CaseSystem(aml.SCENARIO_ALERTS)
    guard = ToolGuard(client, [GuardRule(tool="place_account_hold", query=aml.HOLD_GATE, context=cases.hold_context)])

    good = guard.check("place_account_hold", {"account_id": "ACC-40317", "alert_id": "A-5001", "reason": "x"})
    wrong = guard.check("place_account_hold", {"account_id": "ACC-00000", "alert_id": "A-5001", "reason": "x"})
    unknown = guard.check("place_account_hold", {"account_id": "ACC-40317", "alert_id": "A-9999", "reason": "x"})

    assert good.action == "allow"
    assert wrong.action == "escalate" and "could not build the evaluation context" in wrong.reason
    assert unknown.action == "escalate"


# ---------------------------------------------------------------------------
# Trial pre-screening: Belief (Venn-Abers) through an agent before_tool_callback
# ---------------------------------------------------------------------------


def test_trial_scenarios_in_mock_mode(trial, tmp_path, capsys):
    result = trial.run(["--store", str(tmp_path)])
    s = result["scenarios"]

    assert _summary(s["P-6001"]) == ("allow", None, True, ("allow",))
    assert _summary(s["P-6002"]) == ("escalate", True, True, ("escalate", "human_approved"))
    assert _summary(s["P-6003"]) == ("block", None, False, ("block",))
    assert "cannot mark P-6003" in s["P-6003"]["final_text"]
    assert result["marked"] == [{"patient_id": "P-6001", "trial_id": "SYN-CKD-201"},
                                {"patient_id": "P-6002", "trial_id": "SYN-CKD-201"}]

    p0, _ = s["P-6001"]["venn_abers"]
    lo, hi = s["P-6002"]["venn_abers"]
    _, p1 = s["P-6003"]["venn_abers"]
    assert p0 >= 0.85 and p1 < 0.15 and lo < 0.85 and hi >= 0.15

    assert result["profile"]["method"] == "IVAP" and result["profile"]["status"] == "serving"
    out = capsys.readouterr().out
    assert "Not medical advice" in out and "Venn-Abers pair" in out


def test_trial_coordinator_rejection_leaves_patient_unmarked(trial, shared_store):
    s = trial.run(["--store", str(shared_store)], reviewer=trial.Reviewer(script={"P-6002": False}))["scenarios"]

    assert _summary(s["P-6002"]) == ("escalate", False, False, ("escalate", "human_rejected"))
    assert "declined" in s["P-6002"]["final_text"]


def test_trial_second_run_is_identical(trial, tmp_path, capsys):
    trial.run(["--store", str(tmp_path)])
    first_out = capsys.readouterr().out
    second = trial.run(["--store", str(tmp_path)])
    assert not second["profile"]["calibrated_now"]
    assert first_out.replace("calibrated now", "cached") == capsys.readouterr().out


def test_trial_mark_for_an_uncalibrated_trial_escalates(trial, shared_store):
    from cli_sdk.integrations import GuardRule, ToolGuard
    from cli_sdk.local import LocalCLIClient

    trial.run(["--store", str(shared_store)])
    client = LocalCLIClient(trial.providers.evidence_backend("mock", mock_scorer=trial.mock_eligibility_evidence),
                            store=shared_store)
    system = trial.ScreeningSystem(trial.SCENARIO_RECORDS)
    guard = ToolGuard(client, [GuardRule(tool="mark_eligible", query=trial.ELIGIBILITY,
                                         context=system.eligibility_context, allow_above=0.85, block_below=0.15)])

    other = guard.check("mark_eligible", {"patient_id": "P-6001", "trial_id": "SYN-OTHER-9"})
    assert other.action == "escalate" and "no calibration profile" in other.reason


# ---------------------------------------------------------------------------
# keys, providers and data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("example", ["aml", "trial"])
@pytest.mark.parametrize("provider,env", [("openai", "OPENAI_API_KEY"), ("azure", "AZURE_OPENAI_API_KEY"),
                                          ("anthropic", "ANTHROPIC_API_KEY"), ("gemini", "GOOGLE_API_KEY")])
def test_placeholder_keys_stop_before_any_call(example, provider, env, request, hermetic_env, tmp_path):
    module = request.getfixturevalue(example)
    with pytest.raises(SystemExit) as exc:
        module.main(["--provider", provider, "--store", str(tmp_path)])
    assert env in str(exc.value)
    assert not list(tmp_path.iterdir()), "nothing is calibrated or written before the key check"


def test_placeholder_evidence_key_stops_batch_mode(aml, hermetic_env, tmp_path):
    with pytest.raises(SystemExit) as exc:
        aml.main(["--provider", "mock", "--evidence-provider", "openai", "--batch", "--store", str(tmp_path)])
    assert "OPENAI_API_KEY" in str(exc.value)


def test_adk_model_factory_builds_each_provider_without_network(aml, hermetic_env):
    pytest.importorskip("litellm")
    adk_models = aml.adk_models
    hermetic_env.setenv("OPENAI_API_KEY", "test-key-not-real")
    hermetic_env.setenv("AZURE_OPENAI_API_KEY", "test-key-not-real")
    hermetic_env.setenv("GOOGLE_API_KEY", "test-key-not-real")

    assert adk_models.agent_model("openai").model == "openai/gpt-4.1-mini"
    assert adk_models.agent_model("azure").model == "azure/YOUR_DEPLOYMENT_NAME"
    assert adk_models.agent_model("vllm").model == "hosted_vllm/google/gemma-4-12B-it"
    assert adk_models.agent_model("sglang").model == "openai/google/gemma-4-12B-it"
    assert type(adk_models.agent_model("gemini")).__name__ == "Gemini"
    with pytest.raises(ValueError):
        adk_models.agent_model("mock")  # the mock needs a scripted policy
    if importlib.util.find_spec("anthropic") is not None:
        hermetic_env.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
        model = adk_models.agent_model("anthropic")
        assert type(model).__name__ == "AnthropicLlm" and model.model == "claude-sonnet-5"


def test_committed_datasets_match_their_generators(aml_data, trial_data):
    for module, name in ((aml_data, "aml_alerts.jsonl"), (trial_data, "trial_screening.jsonl")):
        committed = [json.loads(line) for line in (DATA / name).read_text().splitlines() if line.strip()]
        assert committed == module.generate(), f"{name} is stale: rerun its generator"
        assert 250 <= len(committed) <= 400
        share = sum(r["label"] for r in committed) / len(committed)
        assert 0.35 < share < 0.65, f"{name} label balance {share:.2f}"


def _shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _shape(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return ["list"]
    return type(value).__name__


def test_serving_contexts_match_calibration_contexts(aml, trial, aml_data, trial_data):
    """Exchangeability: the guard evaluates contexts built exactly like the calibration set."""
    rows = [json.loads(line) for line in (DATA / "aml_alerts.jsonl").read_text().splitlines()]
    for row in rows:
        assert "label" not in json.dumps(row["context"])
        assert aml_data.alert_context(row["context"]["alert"]) == row["context"]
    cases = aml.CaseSystem(aml.SCENARIO_ALERTS)
    for alert in aml.SCENARIO_ALERTS:
        ctx = cases.hold_context({"alert_id": alert["alert_id"], "account_id": alert["account_id"]})
        assert _shape(ctx) == _shape(rows[0]["context"])
        assert ctx["guideline"] == rows[0]["context"]["guideline"]

    rows = [json.loads(line) for line in (DATA / "trial_screening.jsonl").read_text().splitlines()]
    for row in rows:
        assert "label" not in json.dumps(row["context"])
        assert trial_data.screening_context(row["context"]["patient"]) == row["context"]
    system = trial.ScreeningSystem(trial.SCENARIO_RECORDS)
    for record in trial.SCENARIO_RECORDS:
        ctx = system.eligibility_context({"patient_id": record["patient_id"], "trial_id": "SYN-CKD-201"})
        assert _shape(ctx) == _shape(rows[0]["context"])
        assert ctx["protocol"] == rows[0]["context"]["protocol"]


def test_mock_evidence_models_are_fallible_not_label_lookups(aml, trial):
    for scorer, name in ((aml.mock_hold_evidence, "aml_alerts.jsonl"),
                         (trial.mock_eligibility_evidence, "trial_screening.jsonl")):
        rows = [json.loads(line) for line in (DATA / name).read_text().splitlines()]
        scores = [scorer(r["context"], None, None)["true"] for r in rows]
        assert scores == [scorer(r["context"], None, None)["true"] for r in rows], "deterministic"
        accuracy = sum((s >= 0.5) == bool(r["label"]) for s, r in zip(scores, rows)) / len(rows)
        assert 0.8 < accuracy < 0.99, f"{name}: accuracy {accuracy:.3f}"
