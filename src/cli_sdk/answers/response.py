"""The response returned by ``client.evaluate`` and answer-type dispatch."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Any, Optional

from cli_sdk.answers.base import Answer
from cli_sdk.answers.belief_answer import BeliefAnswer
from cli_sdk.answers.claim_answer import ClaimAnswer
from cli_sdk.answers.gate_answer import GateAnswer
from cli_sdk.answers.interval_answer import IntervalAnswer
from cli_sdk.answers.judge_answer import JudgeAnswer
from cli_sdk.answers.route_answer import RouteAnswer
from cli_sdk.answers.set_answer import SetAnswer

ANSWER_TYPES: dict[str, type[Answer]] = {
    cls.type: cls
    for cls in (
        BeliefAnswer,
        SetAnswer,
        IntervalAnswer,
        GateAnswer,
        ClaimAnswer,
        JudgeAnswer,
        RouteAnswer,
    )
}


def parse_answer(data: dict[str, Any]) -> Answer:
    """Turn one raw answer payload into its typed Answer class."""
    answer_type = data.get("type")
    cls = ANSWER_TYPES.get(answer_type)  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(f"unknown answer type: {answer_type!r}")
    return cls.from_payload(data)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Usage:
    backend_calls: int = 0
    backend_tokens: int = 0

    @classmethod
    def from_payload(cls, data: Optional[dict[str, Any]]) -> "Usage":
        data = data or {}
        # A null count (e.g. no backend tokens for client-side evidence) is 0;
        # usage metadata must never stop the answers from being parsed.
        return cls(
            backend_calls=int(data.get("backend_calls") or 0),
            backend_tokens=int(data.get("backend_tokens") or 0),
        )


class EvaluateResponse:
    """The result of one ``evaluate`` call.

    ``answers`` maps each query id to its typed answer. Subclass this and
    annotate query ids with their answer types to get attribute access,
    checked when the response is parsed::

        class RoutingResponse(EvaluateResponse):
            department: SetAnswer

        result = client.evaluate(..., response_model=RoutingResponse)
        result.department  # SetAnswer, same object as result.answers["department"]
    """

    _BASE_FIELDS = frozenset({"backend", "answers", "usage", "request_id", "warnings"})

    def __init__(
        self,
        answers: dict[str, Answer],
        backend: Optional[dict[str, Any]] = None,
        usage: Optional[Usage] = None,
        request_id: Optional[str] = None,
        warnings: Optional[list[str]] = None,
    ) -> None:
        self.answers = answers
        self.backend = backend or {}
        self.usage = usage or Usage()
        self.request_id = request_id
        self.warnings = warnings or []

    @property
    def heuristic_answers(self) -> list[str]:
        """Query ids whose answers carry no formal guarantee."""
        return [qid for qid, answer in self.answers.items() if answer.is_heuristic]

    @classmethod
    def from_payload(
        cls, data: dict[str, Any], request_id: Optional[str] = None
    ) -> "EvaluateResponse":
        answers = {qid: parse_answer(raw) for qid, raw in (data.get("answers") or {}).items()}
        response = cls(
            answers=answers,
            backend=data.get("backend"),
            usage=Usage.from_payload(data.get("usage")),
            request_id=request_id or data.get("request_id"),
            warnings=list(data.get("warnings") or []),
        )
        response._bind_typed_fields()
        return response

    def _bind_typed_fields(self) -> None:
        """Expose annotated query ids as attributes, checking their types."""
        hints = {}
        for klass in reversed(type(self).__mro__):
            if klass is object or not issubclass(klass, EvaluateResponse):
                continue
            hints.update(typing.get_type_hints(klass))
        for name, expected in hints.items():
            if name.startswith("_") or name in self._BASE_FIELDS:
                continue
            if expected is typing.ClassVar or typing.get_origin(expected) is typing.ClassVar:
                continue  # a class-level constant, not an answer
            if name not in self.answers:
                raise ValueError(f"response_model expects answer {name!r}, which is missing")
            answer = self.answers[name]
            if isinstance(expected, type) and not isinstance(answer, expected):
                raise TypeError(
                    f"answer {name!r} is {type(answer).__name__}, "
                    f"but response_model declares {expected.__name__}"
                )
            object.__setattr__(self, name, answer)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(answers={list(self.answers)}, request_id={self.request_id!r})"
