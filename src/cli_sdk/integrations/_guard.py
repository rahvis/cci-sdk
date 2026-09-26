"""Framework-agnostic tool guard: calibrated allow / escalate / block for agent actions.

An agent framework proposes a tool call; ``ToolGuard.check`` evaluates the
rule for that tool with a CLI client (hosted ``CLIClient`` or in-process
``LocalCLIClient``) and maps the calibrated answer to one of three actions:

=========  ==============================================================
allow      run the tool
escalate   pause for a human (each framework's native human-in-the-loop)
block      do not run the tool; tell the agent why
=========  ==============================================================

Mapping rules, all fail-closed:

- ``Gate``: ``auto_approve`` allows; ``abstain`` blocks; anything else escalates.
- ``Belief``: allow when the Venn-Abers lower bound ``p0 >= allow_above``;
  block when ``block_below`` is set and the upper bound ``p1 < block_below``;
  otherwise escalate. The interval straddling a threshold is exactly the case
  the calibration data cannot decide, so a human decides it.
- ``Set``: allow when the set is a single label in ``allow_labels`` (and,
  with ``match_argument``, equal to the label the agent proposed in that
  tool argument); block when every label in the (non-empty) set is in
  ``block_labels``; otherwise escalate. A set larger than one option means
  the guarantee needs more than one answer to be safe.
- ``Interval``: allow when the whole interval lies inside ``allow_levels``;
  block when it lies inside ``block_levels``; otherwise escalate.
- Any ``heuristic`` answer (missing or stale calibration) escalates, or
  blocks with ``on_heuristic="block"``. It never allows.
- An exception while evaluating escalates, or raises with
  ``on_error="raise"``. It never allows.

A tool with no rule is allowed: guard only the actions that carry risk.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Iterable, Mapping, Optional, Sequence

from cli_sdk.answers import Answer, BeliefAnswer, GateAnswer, IntervalAnswer, SetAnswer
from cli_sdk.queries import Belief, Gate, Interval, Query, Set

ALLOW, ESCALATE, BLOCK = "allow", "escalate", "block"
SEVERITY = {ALLOW: 0, ESCALATE: 1, BLOCK: 2}

ContextBuilder = Callable[..., Any]


def most_severe(decisions: Iterable["GuardDecision"]) -> str:
    """The most conservative action among several calls (block > escalate > allow)."""
    worst = ALLOW
    for decision in decisions:
        if SEVERITY[decision.action] > SEVERITY[worst]:
            worst = decision.action
    return worst


@dataclass(frozen=True)
class GuardDecision:
    """What the guard decided about one proposed tool call, and why."""

    action: str
    tool: str
    arguments: Mapping[str, Any]
    reason: str
    answer: Optional[Answer] = None
    rule: Optional[str] = None

    @property
    def allowed(self) -> bool:
        return self.action == ALLOW

    @property
    def needs_review(self) -> bool:
        return self.action == ESCALATE

    @property
    def blocked(self) -> bool:
        return self.action == BLOCK

    @property
    def guarantee(self) -> Optional[str]:
        return self.answer.guarantee.describe() if self.answer is not None else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary for reviewer queues, interrupts, audit logs and tool results."""
        out: dict[str, Any] = {
            "action": self.action,
            "tool": self.tool,
            "arguments": _jsonable(dict(self.arguments)),
            "reason": self.reason,
        }
        if self.answer is not None:
            g = self.answer.guarantee
            out["guarantee"] = {
                "type": g.type,
                "method": g.method,
                "statement": g.describe(),
                "calibration_profile": g.calibration_profile,
                "calibration_n": g.calibration_n,
            }
            if g.extra.get("group") is not None:
                # Mondrian answers: calibration_n counts this group only, so name the group.
                out["guarantee"]["group"] = g.extra["group"]
            out["evidence"] = _evidence(self.answer)
        return out

    def message(self) -> str:
        """A short sentence for the agent when a call does not run."""
        if self.blocked:
            return (f"The {self.tool} action was blocked by a calibrated safety check: {self.reason} "
                    "Do not retry it; explain to the user that it cannot be done automatically.")
        if self.needs_review:
            return (f"The {self.tool} action needs human review before it can run: {self.reason} "
                    "Do not retry it; tell the user it has been sent for review.")
        return f"The {self.tool} action is allowed: {self.reason}"


@dataclass
class GuardRule:
    """How to judge one tool.

    ``context`` builds the evaluation context from the tool arguments, and
    optionally from framework state: ``lambda args: {...}`` or
    ``lambda args, state: {...}``. Without it, the arguments themselves are
    the context. Whatever it returns must match the context your calibration
    examples were built with.

    ``match_argument`` (Set rules) names the tool argument holding the label
    the agent proposes, e.g. ``"level"`` for ``send_triage_advice(level=...)``.
    The call is then allowed only when the calibrated set is exactly that
    label; a confident set for a *different* label escalates.
    """

    tool: str
    query: Query
    context: Optional[ContextBuilder] = None
    allow_above: float = 0.5
    block_below: Optional[float] = None
    allow_labels: Optional[Collection[str]] = None
    block_labels: Optional[Collection[str]] = None
    allow_levels: Optional[Collection[str]] = None
    block_levels: Optional[Collection[str]] = None
    match_argument: Optional[str] = None
    on_heuristic: str = ESCALATE
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, (Gate, Belief, Set, Interval)):
            raise TypeError("GuardRule.query must be a Gate, Belief, Set or Interval")
        if self.on_heuristic not in (ESCALATE, BLOCK):
            raise ValueError("on_heuristic must be 'escalate' or 'block'")
        if isinstance(self.query, Set) and not self.allow_labels:
            raise ValueError("a Set rule needs allow_labels: the labels that let the tool run")
        if isinstance(self.query, Interval) and not self.allow_levels:
            raise ValueError("an Interval rule needs allow_levels: the levels that let the tool run")
        if not (0.0 <= self.allow_above <= 1.0):
            raise ValueError("allow_above must be in [0, 1]")
        if self.block_below is not None and not (0.0 <= self.block_below <= self.allow_above):
            raise ValueError("block_below must be in [0, allow_above]")

    def build_context(self, arguments: Mapping[str, Any], state: Any = None) -> Any:
        if self.context is None:
            return dict(arguments)
        try:
            n_params = len(inspect.signature(self.context).parameters)
        except (TypeError, ValueError):
            n_params = 1
        return self.context(dict(arguments), state) if n_params >= 2 else self.context(dict(arguments))


class ToolGuard:
    """Decide proposed tool calls with calibrated guarantees.

    Parameters
    ----------
    client:
        A ``LocalCLIClient`` or ``CLIClient`` (anything with ``evaluate(context, queries, ...)``).
    rules:
        One ``GuardRule`` per guarded tool.
    backend:
        Optional backend passed to ``client.evaluate``.
    on_error:
        ``"escalate"`` (default) or ``"raise"`` when evaluation fails.
    cache_size:
        Decisions are cached per (tool, arguments, context) so frameworks that
        re-run a node on resume get the same decision without a second model call.
    """

    def __init__(self, client: Any, rules: Iterable[GuardRule], *, backend: Any = None,
                 on_error: str = ESCALATE, cache_size: int = 1024) -> None:
        self.client = client
        self.rules = {rule.tool: rule for rule in rules}
        if not self.rules:
            raise ValueError("ToolGuard needs at least one GuardRule")
        if on_error not in (ESCALATE, "raise"):
            raise ValueError("on_error must be 'escalate' or 'raise'")
        self.backend = backend
        self.on_error = on_error
        self._cache: "OrderedDict[str, GuardDecision]" = OrderedDict()
        self._cache_size = cache_size
        self.log: list[GuardDecision] = []

    @property
    def guarded_tools(self) -> frozenset[str]:
        return frozenset(self.rules)

    def guards(self, tool: str) -> bool:
        return tool in self.rules

    def check(self, tool: str, arguments: Mapping[str, Any], state: Any = None) -> GuardDecision:
        """Decide one proposed call. Tools without a rule are allowed."""
        rule = self.rules.get(tool)
        if rule is None:
            return GuardDecision(ALLOW, tool, dict(arguments), "no guard rule for this tool")
        try:
            context = rule.build_context(arguments, state)
        except Exception as exc:
            return self._failure(tool, arguments, rule, f"could not build the evaluation context: {exc}", exc)
        key = _cache_key(tool, arguments, context)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        try:
            kwargs = {"backend": self.backend} if self.backend is not None else {}
            response = self.client.evaluate(context, {"guard": rule.query}, **kwargs)
            answer = response.answers["guard"]
            decision = decide(rule, answer, tool, arguments)
        except Exception as exc:
            return self._failure(tool, arguments, rule, f"calibrated check failed: {exc}", exc)
        self._remember(key, decision)
        return decision

    async def acheck(self, tool: str, arguments: Mapping[str, Any], state: Any = None) -> GuardDecision:
        """``check`` without blocking the event loop (runs in a worker thread)."""
        return await asyncio.to_thread(self.check, tool, arguments, state)

    def _failure(self, tool, arguments, rule, reason: str, exc: Exception) -> GuardDecision:
        if self.on_error == "raise":
            raise exc
        decision = GuardDecision(ESCALATE, tool, dict(arguments), reason, rule=rule.name or rule.tool)
        self.log.append(decision)
        return decision

    def _remember(self, key: str, decision: GuardDecision) -> None:
        self.log.append(decision)
        self._cache[key] = decision
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def decide(rule: GuardRule, answer: Answer, tool: str, arguments: Mapping[str, Any]) -> GuardDecision:
    """Map one calibrated answer to allow / escalate / block under ``rule``."""
    args = dict(arguments)
    name = rule.name or rule.tool
    if answer.is_heuristic:
        statement = answer.guarantee.statement or "no calibrated guarantee is available"
        return GuardDecision(rule.on_heuristic, tool, args,
                             f"no guarantee available ({statement})", answer, name)

    if isinstance(answer, GateAnswer):
        confidence = answer.raw.get("confidence")
        threshold = answer.raw.get("threshold")
        detail = (f" (confidence {confidence:.3f}, calibrated threshold {threshold:.3f})"
                  if isinstance(confidence, (int, float)) and isinstance(threshold, (int, float)) else "")
        if answer.approved:
            return GuardDecision(ALLOW, tool, args, f"auto-approved under the calibrated risk bound{detail}.", answer, name)
        if answer.decision == "abstain":
            return GuardDecision(BLOCK, tool, args, f"the gate abstained{detail}.", answer, name)
        if "threshold" in answer.raw and threshold is None and isinstance(confidence, (int, float)):
            # Local mode reports threshold=None when no threshold meets the target on
            # this calibration set: say so, rather than implying one was missed.
            return GuardDecision(ESCALATE, tool, args,
                                 f"no auto-approval threshold meets the target on this calibration set "
                                 f"(confidence {confidence:.3f}), so every request escalates until the profile "
                                 "has more or better-separated examples.", answer, name)
        return GuardDecision(ESCALATE, tool, args, f"below the calibrated auto-approval threshold{detail}.", answer, name)

    if isinstance(answer, BeliefAnswer):
        if answer.venn_abers is None:
            return GuardDecision(ESCALATE, tool, args, "no Venn-Abers interval was returned.", answer, name)
        p0, p1 = answer.venn_abers
        interval = f"calibrated probability in [{p0:.2f}, {p1:.2f}]"
        if p0 >= rule.allow_above:
            return GuardDecision(ALLOW, tool, args, f"{interval}, entirely at or above {rule.allow_above:g}.", answer, name)
        if rule.block_below is not None and p1 < rule.block_below:
            return GuardDecision(BLOCK, tool, args, f"{interval}, entirely below {rule.block_below:g}.", answer, name)
        return GuardDecision(ESCALATE, tool, args, f"{interval}: the calibration data cannot settle it.", answer, name)

    if isinstance(answer, SetAnswer):
        labels = list(answer.set)
        allow = set(rule.allow_labels or ())
        block = set(rule.block_labels or ())
        proposed = args.get(rule.match_argument) if rule.match_argument else None
        if rule.match_argument and len(labels) == 1 and labels[0] != proposed:
            return GuardDecision(ESCALATE, tool, args,
                                 f"the calibrated prediction set is {{{labels[0]}}}, but the agent proposed "
                                 f"{proposed!r}.", answer, name)
        if len(labels) == 1 and labels[0] in allow:
            return GuardDecision(ALLOW, tool, args, f"the calibrated prediction set is exactly {{{labels[0]}}}.", answer, name)
        if labels and block and set(labels) <= block:
            return GuardDecision(BLOCK, tool, args, f"every label in the calibrated set {sorted(labels)} is blocked.", answer, name)
        shown = "empty" if not labels else "{" + ", ".join(labels) + "}"
        return GuardDecision(ESCALATE, tool, args, f"the calibrated prediction set is {shown}, not a single allowed label.", answer, name)

    if isinstance(answer, IntervalAnswer):
        levels = answer.raw.get("levels") or []
        legend = answer.legend or {}
        lo, hi = int(answer.interval[0]), int(answer.interval[1])
        covered = [legend.get(str(i), str(i)) for i in range(lo, hi + 1)]
        allow = set(rule.allow_levels or ())
        block = set(rule.block_levels or ())
        span = f"[{levels[0]}, {levels[1]}]" if len(levels) == 2 else f"[{lo}, {hi}]"
        if covered and set(covered) <= allow:
            return GuardDecision(ALLOW, tool, args, f"the calibrated interval {span} lies inside the allowed levels.", answer, name)
        if covered and block and set(covered) <= block:
            return GuardDecision(BLOCK, tool, args, f"the calibrated interval {span} lies inside the blocked levels.", answer, name)
        return GuardDecision(ESCALATE, tool, args, f"the calibrated interval {span} crosses a decision boundary.", answer, name)

    return GuardDecision(ESCALATE, tool, args, f"unsupported answer type {type(answer).__name__}.", answer, name)


def _evidence(answer: Answer) -> dict[str, Any]:
    if isinstance(answer, GateAnswer):
        keys = ("decision", "confidence", "threshold", "venn_abers")
    elif isinstance(answer, BeliefAnswer):
        keys = ("probability", "venn_abers")
    elif isinstance(answer, SetAnswer):
        keys = ("set", "probabilities")
    elif isinstance(answer, IntervalAnswer):
        keys = ("interval", "levels", "point_estimate")
    else:
        keys = ()
    return {k: _jsonable(answer.raw[k]) for k in keys if k in answer.raw}


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def _cache_key(tool: str, arguments: Mapping[str, Any], context: Any) -> str:
    blob = json.dumps([tool, arguments, context], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


__all__ = ["ALLOW", "ESCALATE", "BLOCK", "GuardRule", "GuardDecision", "ToolGuard", "decide", "most_severe"]
