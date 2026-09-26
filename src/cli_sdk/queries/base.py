"""Shared base for every query primitive (Belief, Set, Interval, Gate, Claim, Judge, Route)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional, Union

from cli_sdk.exceptions import ConfigurationError

Instructions = Union[str, dict, list]
"""The shape every primitive's `instructions` field accepts: a plain
string, or a structured object/array holding the question in one field
and reference data in others (mirroring how `context` itself can be
structured)."""


@dataclass
class Query:
    """Base class for all seven query primitives.

    Subclasses declare their own fields and set the class attribute
    ``type`` to the primitive name the API expects (e.g. ``"set"``).
    ``to_payload`` serializes a query to the JSON shape documented in
    apps/docs/pages/api-reference.mdx, omitting any top-level field left as
    ``None`` so the server sees only what the caller actually set. Values
    *inside* a field are sent as given, so ``Set(options={"billing": None})``
    keeps its option (with a null description).
    """

    type: str = ""

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        for f in fields(self):
            if f.name == "type":
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            payload[f.name] = _serialize(value)
        return payload


def _serialize(value: Any) -> Any:
    if hasattr(value, "to_payload"):
        return value.to_payload()
    if isinstance(value, dict):
        # Keep None values: inside a field they are meaningful data (an
        # option with no description), not "unset".
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def check_probability(name: str, value: Optional[float]) -> None:
    """Reject a level (alpha, delta, target) outside the open interval (0, 1)."""
    if value is not None and not (0 < value < 1):
        raise ConfigurationError(f"`{name}` must be in (0, 1), got {value!r}")
