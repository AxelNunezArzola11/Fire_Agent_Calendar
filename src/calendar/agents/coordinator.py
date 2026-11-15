"""Multi-agent coordinator: route, execute, validate, aggregate, audit.

Routes one request to its specialist agent (detections/event/
science), validates the outcome through the evidence agent's gates
(applied inside each agent run), aggregates a traceable final
explanation, and records per-agent audit rows in qwen_log when a
connection is supplied. Pure orchestration: computes no science.

Never raises on worker faults or provider faults (those become
error/fallback results). Raises ValueError on bad kind, target id,
or payload (caller contract, enforced at the boundary).
"""

from __future__ import annotations

from . import detection_intel, event_analysis, evidence_validator
from . import science_observer
from .contracts import AgentResponse, CoordinatorResult, TraceStep
from .providers import ProviderChain, default_chain, redact

_ROUTES = {
    "detections": detection_intel,
    "event": event_analysis,
    "science": science_observer,
}

assert set(_ROUTES) == {"detections", "event", "science"}


def _safe_limitations(payload) -> tuple:
    if not isinstance(payload, dict):
        return ()
    items = payload.get("limitations", [])
    if not isinstance(items, list) or not all(
            isinstance(v, str) for v in items):
        return ()
    return tuple(items)


def _error_response(agent_name: str, request, reason: str) -> AgentResponse:
    safe = redact(reason)[:500]
    return AgentResponse(agent_name, "error",
                         f"Agent {agent_name} failed; no analysis is "
                         f"available for {request.target_id}.",
                         findings=(), fallback_reason=safe,
                         validation=safe)


def analyze(kind: str, target_id: str, payload: dict, *,
            chain: ProviderChain | None = None,
            conn=None) -> CoordinatorResult:
    """Run the agent workflow for one target."""
    from .contracts import AgentRequest

    request = AgentRequest(kind, target_id, payload)  # boundary: may raise
    try:
        worker = _ROUTES[request.kind]
    except KeyError:
        return CoordinatorResult(
            request.kind, request.target_id,
            "No agent is registered for this analysis kind.",
            steps=(TraceStep(agent="coordinator", status="error",
                             provider="", model="", latency_ms=0.0,
                             validation="unrouted kind",
                             fallback_reason="unrouted kind"),))
    if chain is None:
        try:
            chain = default_chain()
        except Exception as exc:  # noqa: BLE001 - broken default chain
            return CoordinatorResult(
                request.kind, request.target_id,
                "Agent providers failed to initialize; no analysis "
                "is available.",
                steps=(TraceStep(
                    agent="coordinator", status="error", provider="",
                    model="", latency_ms=0.0,
                    validation=redact(
                        f"{type(exc).__name__}")[:500],
                    fallback_reason="provider chain unavailable"),))
    try:
        response = worker.analyze(request, chain)
    except Exception as exc:  # noqa: BLE001 - agent fault -> error result
        response = _error_response(
            worker.NAME, request, f"{type(exc).__name__}: {exc}")
    if not isinstance(response, AgentResponse):
        response = _error_response(
            worker.NAME, request,
            f"worker returned {type(response).__name__}, "
            f"not AgentResponse")
    steps = [TraceStep(agent=response.agent, status=response.status,
                       provider=response.provider, model=response.model,
                       latency_ms=response.latency_ms,
                       validation=response.validation,
                       fallback_reason=response.fallback_reason,
                       retries=response.retries,
                       response_chars=response.response_chars,
                       input_tokens=response.input_tokens,
                       generated_tokens=response.generated_tokens)]
    audit_refs: list[str] = []
    if conn is not None:
        audit_refs = _audit(conn, request, response)
    evidence = response.evidence_used or \
        evidence_validator.evidence_ids(request.payload)
    limitations = response.limitations or _safe_limitations(
        request.payload)
    return CoordinatorResult(
        request.kind, request.target_id, response.summary,
        evidence_ids=tuple(evidence), steps=tuple(steps),
        limitations=tuple(limitations), audit_refs=tuple(audit_refs))


def _audit(conn, request, response) -> list[str]:
    from ..events import store as store_mod

    refs = []
    try:
        row_id = store_mod.append_qwen_log(
            conn, target_type=f"agent:{response.agent}",
            target_id=f"{request.kind}:{request.target_id}",
            status=response.status, latency_ms=response.latency_ms,
            model=response.model,
            error=redact(response.fallback_reason
                         or response.validation)[:500])
        refs.append(f"qwen_log:{row_id}")
    except Exception:  # noqa: BLE001 - audit never breaks analysis
        pass
    return refs
