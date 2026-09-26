"""A stand-in for your review queue: scripted by default, or interactive with ``--interactive``.

In production, escalations go to a queue your reviewers work through, and
the agent resumes when a decision arrives, possibly hours later and in
another process (use a durable checkpointer or session store for that).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional


class Reviewer:
    def __init__(self, *, interactive: bool = False, script: Optional[Mapping[str, bool]] = None,
                 default: bool = False, role: str = "reviewer") -> None:
        self.interactive = interactive
        self.script = dict(script or {})
        self.default = default
        self.role = role

    def decide(self, case_id: str, request: Mapping[str, Any]) -> bool:
        """Show the review request and return True to approve, False to reject."""
        print(f"    review request for {case_id}:")
        for line in format_request(request).splitlines():
            print(f"      {line}")
        if self.interactive:
            answer = input(f"    {self.role}, approve this action? [y/N] ").strip().lower()
            approved = answer in ("y", "yes")
        else:
            approved = self.script.get(case_id, self.default)
            print(f"    [simulated {self.role}] {'approved' if approved else 'rejected'}")
        return approved


def format_request(request: Mapping[str, Any]) -> str:
    """Readable text for a review request: a guard decision dict or a framework payload."""
    cli = request.get("cli") if isinstance(request.get("cli"), Mapping) else request
    lines = []
    if cli.get("tool"):
        lines.append(f"action:    {cli['tool']}({json.dumps(cli.get('arguments', {}), sort_keys=True)})")
    if cli.get("reason"):
        lines.append(f"why:       {cli['reason']}")
    guarantee = cli.get("guarantee")
    if isinstance(guarantee, Mapping) and guarantee.get("statement"):
        lines.append(f"guarantee: {guarantee['statement']}")
    if not lines:
        lines.append(json.dumps(request, default=str)[:400])
    return "\n".join(lines)
