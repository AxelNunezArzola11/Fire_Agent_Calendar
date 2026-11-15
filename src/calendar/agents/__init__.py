"""FireAgent Calendar multi-agent analysis (deterministic-first, provider-backed).

Agents consume verified evidence payloads and return structured,
validated explanations. Every LLM answer passes the evidence
validator; any failure falls back to deterministic templates.
"""

from .contracts import (KINDS, STATUSES, AgentRequest, AgentResponse,
                        CoordinatorResult, Finding, TraceStep)

__all__ = ["KINDS", "STATUSES", "AgentRequest", "AgentResponse",
           "CoordinatorResult", "Finding", "TraceStep"]
