"""Shared attempt loop for LLM-backed agents (plumbing, not an agent).

Iterates providers in order with validity-level failover: transport
failures AND invalid JSON both advance to the next provider. Only
when every provider fails (or none is configured) does the
deterministic fallback run. Agents supply findings + fallback text;
this module never invents either. Latency reported per response is
the sum of its attempts; ok responses carry the winning attempt only.
"""

from __future__ import annotations

import json

from . import evidence_validator as validator
from .contracts import AgentRequest, AgentResponse

PROMPT_BUDGET_CHARS = 100000
REASON_CAP_CHARS = 2000


def _coerce_parsed(parsed) -> dict | None:
    """Validated parsed shape or None (never raises, never explodes)."""
    try:
        if not isinstance(parsed, dict):
            return None
        if not isinstance(parsed.get("summary"), str) \
                or not parsed["summary"].strip():
            return None
        for key in ("evidence_used", "uncertainty", "limitations"):
            items = parsed.get(key)
            if not isinstance(items, list) or not all(
                    isinstance(v, str) for v in items):
                return None
        return {"summary": parsed["summary"],
                "evidence_used": tuple(parsed["evidence_used"]),
                "uncertainty": tuple(parsed["uncertainty"]),
                "limitations": tuple(parsed["limitations"])}
    except (AttributeError, TypeError):
        return None


def _safe_limitations(payload) -> tuple:
    if not isinstance(payload, dict):
        return ()
    items = payload.get("limitations", [])
    if not isinstance(items, list) or not all(
            isinstance(v, str) for v in items):
        return ()
    return tuple(items)


def run_llm_backed(*, agent_name: str, request: AgentRequest,
                   chain, system: str, findings,
                   fallback_text: str) -> AgentResponse:
    """Execute one agent: deterministic findings + optional LLM prose."""
    if not isinstance(fallback_text, str) or not fallback_text.strip():
        return AgentResponse(
            agent_name, "error", "Agent misconfigured: fallback template "
            "missing or empty.", findings=(),
            fallback_reason="fallback template missing",
            validation="fallback template missing")
    structure = validator.check_findings(findings)
    if not structure["valid"]:
        return AgentResponse(
            agent_name, "error",
            f"Finding assembly failed: {structure['reason']}.",
            findings=(), fallback_reason=structure["reason"],
            validation=structure["reason"])
    providers = list(getattr(chain, "providers", None) or [])
    citable = validator.evidence_ids(request.payload)
    prompt = ("EVIDENCE (JSON, authoritative):\n"
              + json.dumps(request.payload, sort_keys=True, default=str)
              + "\n\nCitable evidence values (copy exactly into "
                "evidence_used; never invent, never use key names):\n"
              + (", ".join(citable[:50]) if citable else "(none)"))
    if len(prompt) > PROMPT_BUDGET_CHARS:
        return AgentResponse(
            agent_name, "error",
            "Evidence payload exceeds the prompt budget; refusing to "
            "truncate evidence silently.", findings=tuple(findings),
            fallback_reason="prompt exceeds size budget",
            validation="prompt exceeds size budget")
    failures: list[str] = []
    latency_total = 0.0
    last_provider, last_model = "", ""
    for provider in providers:
        name = getattr(provider, "name", "?")
        try:
            attempt = provider.complete(prompt, system)
        except Exception as exc:  # noqa: BLE001 - faulty provider
            failures.append(f"{name}: faulty provider "
                            f"({type(exc).__name__})")
            continue
        latency_total += max(0.0, getattr(attempt, "latency_ms", 0.0) or 0.0)
        last_provider = getattr(attempt, "provider", name) or name
        last_model = getattr(attempt, "model", "") or ""
        if not getattr(attempt, "ok", False):
            failures.append(f"{last_provider}: "
                            f"{getattr(attempt, 'error', 'failed')}")
            continue
        verdict = validator.check_llm_json(
            getattr(attempt, "text", ""), request.payload)
        if not verdict["valid"]:
            failures.append(f"{last_provider} response invalid: "
                            f"{verdict['reason']}")
            continue
        parsed = _coerce_parsed(verdict["parsed"])
        if parsed is None:
            failures.append(f"{last_provider} response malformed "
                            f"after validation")
            continue
        return AgentResponse(
            agent_name, "ok", parsed["summary"],
            evidence_used=parsed["evidence_used"],
            uncertainty=parsed["uncertainty"],
            limitations=parsed["limitations"],
            findings=tuple(findings), provider=last_provider,
            model=last_model,
            latency_ms=max(
                0.0, getattr(attempt, "latency_ms", 0.0) or 0.0),
            validation="all checks passed",
            retries=int(getattr(attempt, "retries", 0) or 0),
            response_chars=int(
                getattr(attempt, "response_chars", 0) or 0),
            input_tokens=getattr(attempt, "input_tokens", None),
            generated_tokens=getattr(attempt, "generated_tokens", None))
    reason = ("no providers configured" if not providers
              else "no provider usable (" + "; ".join(failures) + ")")
    return AgentResponse(
        agent_name, "fallback", fallback_text,
        evidence_used=validator.evidence_ids(request.payload),
        uncertainty=("no model inference ran",),
        limitations=_safe_limitations(request.payload),
        findings=tuple(findings), provider=last_provider, model=last_model,
        latency_ms=latency_total, fallback_reason=reason[:REASON_CAP_CHARS],
        validation=(reason[:REASON_CAP_CHARS]))
