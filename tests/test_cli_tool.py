"""The ``cli calibration ...`` command-line tool, driven through ``main([...])``.

``audit-local`` runs fully offline; ``show`` and ``audit`` talk to the API
and are pointed at ``MockAPI`` by swapping the ``CLIClient`` they construct.
Exit codes: 0 pass, 1 audit failure (so CI can gate a deploy), 2 usage or
input error.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import cli_sdk.client.sync_client as sync_client
from cli_sdk.cli_tool import main
from support import API_KEY, PROFILE, MockAPI


def write_outcomes(tmp_path: Path, lines: list[str], name: str = "covered.txt") -> str:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class TestAuditLocal:
    def test_pass_exits_zero(self, tmp_path, capsys):
        path = write_outcomes(tmp_path, ["1"] * 95 + ["0"] * 5)
        assert main(["calibration", "audit-local", "--covered", path, "--target", "0.90"]) == 0
        out = capsys.readouterr().out
        assert "result             pass" in out
        assert "0.9500 on 100 examples" in out

    def test_fail_exits_one(self, tmp_path, capsys):
        path = write_outcomes(tmp_path, ["1"] * 700 + ["0"] * 300)
        assert main(["calibration", "audit-local", "--covered", path, "--target", "0.90"]) == 1
        assert "result             fail" in capsys.readouterr().out

    def test_fail_below_overrides_a_statistical_pass(self, tmp_path, capsys):
        path = write_outcomes(tmp_path, ["1"] * 88 + ["0"] * 12)
        argv = ["calibration", "audit-local", "--covered", path, "--target", "0.90"]
        assert main(argv) == 0  # 0.88 on 100 is not evidence of under-coverage
        assert main(argv + ["--fail-below", "0.89"]) == 1
        assert "below --fail-below 0.89" in capsys.readouterr().out

    def test_confidence_flag_changes_the_interval(self, tmp_path, capsys):
        path = write_outcomes(tmp_path, ["1"] * 90 + ["0"] * 10)
        main(["calibration", "audit-local", "--covered", path, "--target", "0.9", "--confidence", "0.5"])
        narrow = capsys.readouterr().out
        main(["calibration", "audit-local", "--covered", path, "--target", "0.9", "--confidence", "0.99"])
        wide = capsys.readouterr().out
        assert narrow != wide

    def test_accepted_tokens_and_blank_lines(self, tmp_path, capsys):
        path = write_outcomes(tmp_path, ["1", "true", "True", "TRUE", "", "0", "false", "False", "  1  "])
        assert main(["calibration", "audit-local", "--covered", path, "--target", "0.5"]) == 0
        assert "0.6250 on 8 examples" in capsys.readouterr().out

    def test_unrecognized_outcome_is_an_input_error(self, tmp_path, capsys):
        path = write_outcomes(tmp_path, ["1", "1", "maybe", "0"])
        assert main(["calibration", "audit-local", "--covered", path, "--target", "0.5"]) == 2
        err = capsys.readouterr().err
        assert "line 3" in err and "maybe" in err

    def test_empty_file_is_an_input_error(self, tmp_path, capsys):
        path = write_outcomes(tmp_path, [""])
        assert main(["calibration", "audit-local", "--covered", path, "--target", "0.9"]) == 2
        assert "no outcomes" in capsys.readouterr().err

    def test_missing_file_is_an_input_error(self, tmp_path, capsys):
        missing = str(tmp_path / "nope.txt")
        assert main(["calibration", "audit-local", "--covered", missing, "--target", "0.9"]) == 2
        assert "nope.txt" in capsys.readouterr().err

    def test_target_is_required(self, tmp_path):
        path = write_outcomes(tmp_path, ["1"])
        with pytest.raises(SystemExit) as excinfo:
            main(["calibration", "audit-local", "--covered", path])
        assert excinfo.value.code == 2

    @pytest.mark.parametrize("argv", [[], ["calibration"], ["bogus"]])
    def test_usage_errors(self, argv):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2


@pytest.fixture
def mocked_cli_client(monkeypatch: pytest.MonkeyPatch, api: MockAPI) -> MockAPI:
    """Make the tool's ``CLIClient()`` talk to ``api``."""
    real = sync_client.CLIClient

    def factory(**kwargs):
        return real(api_key=API_KEY, http_client=httpx.Client(transport=api.transport()), **kwargs)

    monkeypatch.setattr(sync_client, "CLIClient", factory)
    return api


class TestHostedCommands:
    def test_show(self, mocked_cli_client: MockAPI, capsys):
        mocked_cli_client.reply(
            200,
            {
                "name": PROFILE,
                "method": "APS",
                "alpha": 0.1,
                "n": 1204,
                "version": 3,
                "realized_coverage_ci": [0.884, 0.915],
                "last_audit": {"result": "pass"},
            },
        )
        assert main(["calibration", "show", PROFILE]) == 0
        out = capsys.readouterr().out
        assert mocked_cli_client.last.url.path == f"/v1/calibration-profiles/{PROFILE}"
        assert f"{PROFILE} (v3)" in out
        assert "APS / 0.1" in out
        assert "1204 (minimum 9, recommended 1000)" in out
        assert "[0.884, 0.915]" in out
        assert "serving guarantees yes" in out

    def test_show_underpowered_profile(self, mocked_cli_client: MockAPI, capsys):
        mocked_cli_client.reply(200, {"name": "tiny", "alpha": 0.05, "n": 4})
        assert main(["calibration", "show", "tiny"]) == 0
        assert "no (below minimum n)" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "server_result, fail_below, expected_code",
        [("pass", None, 0), ("fail", None, 1), ("pass", 0.95, 1), ("pass", 0.85, 0)],
    )
    def test_audit_exit_codes(
        self, mocked_cli_client: MockAPI, tmp_path, capsys, server_result, fail_below, expected_code
    ):
        examples = tmp_path / "fresh.jsonl"
        examples.write_text(
            json.dumps({"context": {"ticket": "a"}, "label": "billing"})
            + "\n\n"
            + json.dumps({"context": {"ticket": "b"}, "label": "technical"})
            + "\n",
            encoding="utf-8",
        )
        mocked_cli_client.reply(
            200,
            {
                "result": server_result,
                "realized_coverage": 0.91,
                "ci": [0.88, 0.94],
                "sample": 2,
                "target": 0.9,
            },
        )
        argv = ["calibration", "audit", PROFILE, "--examples", str(examples)]
        if fail_below is not None:
            argv += ["--fail-below", str(fail_below)]
        assert main(argv) == expected_code
        request = mocked_cli_client.last
        assert request.method == "POST" and request.url.path == f"/v1/calibration-profiles/{PROFILE}/audit"
        assert json.loads(request.content) == {
            "examples": [
                {"context": {"ticket": "a"}, "label": "billing"},
                {"context": {"ticket": "b"}, "label": "technical"},
            ]
        }
        assert f"result             {server_result}" in capsys.readouterr().out

    def test_audit_requires_examples(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["calibration", "audit", PROFILE])
        assert excinfo.value.code == 2

    # Exit 1 means "the audit failed"; anything that stops the audit from
    # running at all must be distinguishable from that, so it exits 2.

    def test_api_error_exits_two_not_one(self, mocked_cli_client: MockAPI, capsys):
        mocked_cli_client.reply(401, {"message": "invalid API key"})
        assert main(["calibration", "show", PROFILE]) == 2
        err = capsys.readouterr().err
        assert "AuthenticationError" in err and "invalid API key" in err

    def test_audit_api_error_exits_two(self, mocked_cli_client: MockAPI, tmp_path, capsys):
        examples = tmp_path / "fresh.jsonl"
        examples.write_text(json.dumps({"context": "c", "label": "billing"}) + "\n", encoding="utf-8")
        mocked_cli_client.reply(404, {"message": "no such profile"})
        assert main(["calibration", "audit", "missing", "--examples", str(examples)]) == 2
        assert "no such profile" in capsys.readouterr().err

    def test_missing_api_key_exits_two(self, capsys):
        assert main(["calibration", "show", PROFILE]) == 2
        assert "CLI_API_KEY" in capsys.readouterr().err

    def test_unreadable_examples_exit_two(self, mocked_cli_client: MockAPI, tmp_path, capsys):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"context": "c", "label": \n', encoding="utf-8")
        assert main(["calibration", "audit", PROFILE, "--examples", str(bad)]) == 2
        assert main(["calibration", "audit", PROFILE, "--examples", str(tmp_path / "absent.jsonl")]) == 2
        assert mocked_cli_client.requests == []
        err = capsys.readouterr().err
        assert "bad.jsonl" in err and "absent.jsonl" in err
