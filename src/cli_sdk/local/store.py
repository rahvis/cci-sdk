"""On-disk storage for local calibration profiles.

A local profile is one JSON file holding everything a guarantee depends on:
the query it calibrates (and a fingerprint of the parts that define the
scoring function), the backend fingerprint, and the per-example evidence
(scores) with their labels. Thresholds are recomputed from the stored
evidence at evaluation time, so changing ``alpha`` or ``target`` on a query
never needs a new model call.

The file is plain JSON on purpose: it is the audit record. Commit it next
to your code, review it in pull requests, and ship it with your service.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Query fields that define the scoring function. Changing any of them means
# the stored scores were produced by a different function.
_SCORING_FIELDS = ("type", "instructions", "options", "levels", "criteria", "support_source", "task")

FORMAT_VERSION = 1


def query_fingerprint(query_payload: dict[str, Any]) -> str:
    material = {k: query_payload.get(k) for k in _SCORING_FIELDS if query_payload.get(k) is not None}
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ProfileRecord:
    """Everything stored for one local calibration profile."""

    name: str
    query_type: str
    query: dict[str, Any]
    query_fingerprint: str
    backend_fingerprint: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)
    method: Optional[str] = None
    group_by: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    version: int = 1
    format_version: int = FORMAT_VERSION

    @property
    def n(self) -> int:
        return len(self.records)

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "name": self.name,
            "version": self.version,
            "query_type": self.query_type,
            "method": self.method,
            "group_by": self.group_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "query": self.query,
            "query_fingerprint": self.query_fingerprint,
            "backend_fingerprint": self.backend_fingerprint,
            "n": self.n,
            "records": self.records,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ProfileRecord":
        return cls(
            name=data["name"],
            query_type=data["query_type"],
            query=data.get("query") or {},
            query_fingerprint=data.get("query_fingerprint", ""),
            backend_fingerprint=data.get("backend_fingerprint") or {},
            records=list(data.get("records") or []),
            method=data.get("method"),
            group_by=data.get("group_by"),
            created_at=data.get("created_at", _now()),
            updated_at=data.get("updated_at", _now()),
            version=int(data.get("version", 1)),
            format_version=int(data.get("format_version", FORMAT_VERSION)),
        )


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class LocalProfileStore:
    """A directory of ``<profile>.json`` files."""

    def __init__(self, root: str | os.PathLike[str] = ".cli_profiles") -> None:
        self.root = Path(root)

    def _path(self, name: str) -> Path:
        if not _SAFE_NAME.match(name):
            raise ValueError(
                f"invalid calibration profile name {name!r}: use letters, digits, '.', '_' or '-'"
            )
        return self.root / f"{name}.json"

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def load(self, name: str) -> Optional[ProfileRecord]:
        path = self._path(name)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as fh:
            return ProfileRecord.from_json(json.load(fh))

    def save(self, record: ProfileRecord) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record.updated_at = _now()
        path = self._path(record.name)
        # Write atomically so a crash never leaves a half-written audit record.
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f".{record.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record.to_json(), fh, indent=2, ensure_ascii=False, default=str)
                fh.write("\n")
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def delete(self, name: str) -> bool:
        path = self._path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))
