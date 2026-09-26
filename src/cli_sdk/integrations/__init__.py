"""Calibrated guardrails for agent frameworks.

``ToolGuard`` (framework-agnostic) decides each proposed tool call —
allow, escalate to a human, or block — from a CLI answer with a
finite-sample guarantee. Thin adapters map those decisions onto each
framework's native hooks and human-in-the-loop mechanism:

======================================  ========================================
Module                                  Framework
======================================  ========================================
``cli_sdk.integrations.langchain``      LangChain 1.x ``create_agent`` middleware
``cli_sdk.integrations.langgraph``      LangGraph guard node + ``interrupt()``
``cli_sdk.integrations.google_adk``     Google ADK ``before_tool_callback`` / plugin
``cli_sdk.integrations.agent_framework`` Microsoft Agent Framework function middleware
======================================  ========================================

Framework packages are optional; each adapter imports its framework only
when used. Install with ``pip install "cli-sdk[langchain]"`` (or
``[langgraph]``, ``[google-adk]``, ``[agent-framework]``).
"""

from cli_sdk.integrations._guard import (
    ALLOW,
    BLOCK,
    ESCALATE,
    GuardDecision,
    GuardRule,
    ToolGuard,
    decide,
    most_severe,
)

__all__ = ["ALLOW", "BLOCK", "ESCALATE", "GuardDecision", "GuardRule", "ToolGuard", "decide", "most_severe"]
