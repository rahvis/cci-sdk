"""Load labelled examples and calibrate a query once per evidence model.

Profiles are cached as JSON files in the store directory. A profile is
rebuilt automatically when the query's prompt or options change, or when a
different evidence model or configuration is used, because the calibration
only describes the scoring function it was built with.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from cli_sdk.calibration.profile import CalibrationProfile
from cli_sdk.evidence import EvidenceError
from cli_sdk.local import LocalCLIClient
from cli_sdk.queries import Query

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_jsonl(name: str | Path) -> list[dict[str, Any]]:
    """Read ``data/<name>`` (or a path) as a list of JSON objects."""
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = DATA_DIR / path
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _progress(label: str):
    def report(done: int, total: int) -> None:
        if done == total or done % max(1, total // 10) == 0:
            print(f"    {label}: scored {done}/{total}", file=sys.stderr, flush=True)

    return report


def ensure_calibrated(
    client: LocalCLIClient,
    queries: Iterable[Query],
    examples: list[dict[str, Any]],
    *,
    recalibrate: bool = False,
    quiet: bool = False,
) -> list[CalibrationProfile]:
    """Calibrate each query on ``examples`` unless a matching profile is already stored."""
    profiles = []
    for query in queries:
        ok, reason = client.calibration_status(query)
        if ok and not recalibrate:
            profiles.append(client.get_profile(query.calibration_profile))
            continue
        if not quiet:
            why = "rebuilding on request" if ok else reason
            print(f"  calibrating '{query.calibration_profile}' on {len(examples)} labelled examples "
                  f"({why}); this is one-time and cached in {client.store.root}", file=sys.stderr)
        try:
            profiles.append(client.calibrate(query, examples,
                                             progress=None if quiet else _progress(query.calibration_profile)))
        except EvidenceError as exc:
            name = getattr(client.backend, "name", "evidence")
            raise SystemExit(
                f"could not score the calibration examples for '{query.calibration_profile}' with the {name} "
                f"evidence model: {exc}\nCheck that the model server or API is reachable and the settings in "
                "examples/agents/.env.example are right, or run with --provider mock."
            ) from exc
    return profiles


def describe_profile(profile: CalibrationProfile) -> str:
    lo, hi = profile.realized_coverage_ci or (None, None)
    ci = f", realized-coverage CI [{lo:.3f}, {hi:.3f}]" if lo is not None else ""
    return (f"{profile.name}: {profile.method}, n={profile.n} (minimum {profile.minimum_n}, "
            f"recommended {profile.recommended_n}){ci}, status={profile.status}")


def default_store(example_file: str, override: Optional[str]) -> Path:
    return Path(override) if override else Path(example_file).resolve().parent / ".cli_profiles"
