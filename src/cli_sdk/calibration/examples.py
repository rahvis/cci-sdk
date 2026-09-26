"""Labelled calibration examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from cli_sdk.exceptions import ConfigurationError

SOURCES = ("human", "judge")


@dataclass
class CalibrationExample:
    """One labelled example for a calibration profile.

    ``source`` records who produced the label. Label-efficient calibration
    combines a small ``human`` sample with many ``judge`` labels validly;
    see ``label_efficient`` and apps/docs/pages/calibration.mdx.
    ``group`` sets the Mondrian group explicitly when the profile's
    ``group_by`` field is not present in ``context``.
    """

    context: Any
    label: Any
    source: str = "human"
    group: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ConfigurationError(f"source must be one of {SOURCES}")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"context": self.context, "label": self.label, "source": self.source}
        if self.group is not None:
            payload["group"] = self.group
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def normalize_examples(examples: list[Any]) -> list[dict[str, Any]]:
    """Accept CalibrationExample objects or plain dicts with context/label keys."""
    normalized = []
    for i, example in enumerate(examples):
        if isinstance(example, CalibrationExample):
            normalized.append(example.to_payload())
        elif isinstance(example, dict) and "context" in example and "label" in example:
            normalized.append(dict(example))
        elif isinstance(example, dict) and "context" in example and "tier_outputs" in example:
            normalized.append(dict(example))  # Route profiles: per-tier outputs plus a gold label
        else:
            raise ConfigurationError(f"example {i} must have 'context' and 'label' keys")
    return normalized
