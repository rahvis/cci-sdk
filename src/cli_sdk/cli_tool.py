"""The ``cli`` command-line tool for calibration profiles.

Examples::

    cli calibration show support-routing-v3
    cli calibration audit support-routing-v3 --examples fresh.jsonl --fail-below 0.88
    cli calibration audit-local --covered covered.txt --target 0.90

``audit`` exits with status 1 when the audit fails, so it can gate a
deploy in CI. ``audit-local`` runs the same check with no network access
on a file of 0/1 coverage outcomes. Status 2 means the audit could not run
(bad arguments or input file, missing API key, or an API error).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _cmd_show(args: argparse.Namespace) -> int:
    from cli_sdk.client.sync_client import CLIClient

    with CLIClient() as client:
        profile = client.calibration_profiles.get(args.name)
    ci = profile.realized_coverage_ci
    print(f"profile            {profile.name} (v{profile.version})")
    print(f"method / alpha     {profile.method} / {profile.alpha}")
    print(f"n                  {profile.n} (minimum {profile.minimum_n}, recommended {profile.recommended_n})")
    if ci:
        print(f"coverage CI        [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"serving guarantees {'yes' if profile.can_serve_guarantees else 'no (below minimum n)'}")
    if profile.last_audit:
        print(f"last audit         {profile.last_audit}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from cli_sdk.client.sync_client import CLIClient

    try:
        examples = _read_jsonl(args.examples)
    except (OSError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
        print(f"error: {args.examples}: {exc}", file=sys.stderr)
        return 2
    with CLIClient() as client:
        result = client.calibration_profiles.audit(args.name, fresh_examples=examples)
    return _report(result, args.fail_below)


_COVERED = {"1", "true", "yes"}
_MISSED = {"0", "false", "no"}


def _read_outcomes(path: str) -> list[bool]:
    """Parse one 0/1 (or true/false) outcome per line; blank lines are skipped.

    Anything else is an input error rather than a silent miss, so a typo can
    never quietly move the audited coverage.
    """
    outcomes = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            token = line.strip()
            if not token:
                continue
            if token.lower() in _COVERED:
                outcomes.append(True)
            elif token.lower() in _MISSED:
                outcomes.append(False)
            else:
                raise ValueError(f"{path}, line {lineno}: expected 0/1 or true/false, got {token!r}")
    if not outcomes:
        raise ValueError(f"{path}: no outcomes to audit")
    return outcomes


def _cmd_audit_local(args: argparse.Namespace) -> int:
    from cli_sdk.calibration.audit import audit_coverage

    try:
        covered = _read_outcomes(args.covered)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = audit_coverage(covered, target=args.target, confidence=args.confidence)
    return _report(result, args.fail_below)


def _report(result, fail_below: Optional[float]) -> int:
    print(f"result             {result.result}")
    print(f"realized coverage  {result.realized_coverage:.4f} on {result.sample} examples")
    print(f"confidence interval [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    failed = not result.passed
    if fail_below is not None and result.realized_coverage < fail_below:
        print(f"below --fail-below {fail_below}")
        failed = True
    return 1 if failed else 0


def _probability(text: str) -> float:
    value = float(text)
    if not (0.0 < value < 1.0):
        raise argparse.ArgumentTypeError(f"must be strictly between 0 and 1, got {text}")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="cli", description="Conformal Logit Inference tools")
    groups = parser.add_subparsers(dest="group", required=True)
    calibration = groups.add_parser("calibration", help="calibration profile commands")
    commands = calibration.add_subparsers(dest="command", required=True)

    show = commands.add_parser("show", help="print a profile's size, coverage CI, and audit status")
    show.add_argument("name")
    show.set_defaults(func=_cmd_show)

    audit = commands.add_parser("audit", help="audit a profile against fresh labelled examples")
    audit.add_argument("name")
    audit.add_argument("--examples", required=True, help="JSONL file of {context, label} examples")
    audit.add_argument("--fail-below", type=_probability, default=None)
    audit.set_defaults(func=_cmd_audit)

    local = commands.add_parser("audit-local", help="audit 0/1 coverage outcomes offline")
    local.add_argument("--covered", required=True, help="file with one 0/1 outcome per line")
    local.add_argument("--target", type=_probability, required=True)
    local.add_argument("--confidence", type=_probability, default=0.95)
    local.add_argument("--fail-below", type=_probability, default=None)
    local.set_defaults(func=_cmd_audit_local)

    args = parser.parse_args(argv)
    from cli_sdk.exceptions import CLIError

    try:
        return args.func(args)
    except CLIError as exc:
        # Exit 2, not 1: status 1 means "the audit failed", and CI must be
        # able to tell that apart from a missing key or an API error.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
