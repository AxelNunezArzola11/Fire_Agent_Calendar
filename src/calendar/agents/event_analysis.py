"""Agent 2 — Fire Event Analysis Agent.

Analyzes event lifecycle payloads (state, alerts, anomalies): assembles
status timelines and alert summaries, then narrates them. Reuses the
tested qwen payload builders and deterministic fallback; never calls
an event a confirmed fire perimeter. Payload contract: kind
("event"|"alert") plus the qwen builder fields, all read defensively.
"""

from __future__ import annotations

from ..events import qwen as qwen_mod
from . import runner
from .contracts import AgentRequest, AgentResponse, Finding

NAME = "event-analysis"

# Prompt rules vs enforcement: JSON schema, banned claims, evidence
# grounding, and number grounding are enforced by evidence_validator;
# the brevity ask below is guidance only and is NOT enforced.
SYSTEM = """\
You interpret verified fire-event evidence for analysts.
Rules (violations invalidate your answer):
- Use ONLY the supplied EVIDENCE payload. Never invent observations,
  metrics, dates, identifiers, or baselines.
- Events group observations over time; they are not confirmed fires,
  perimeters, or incidents. Resolved never means contained.
- evidence_used must list values copied from EVIDENCE (ids, periods, dates, versions as they appear); never key names.
- Never write any of these words anywhere in your answer, even in disclaimers: perimeter, perimeters, containment, contained, extinguish, extinguished. Say "observation extent" and "no activity observed" instead. Any occurrence invalidates the answer.
- Reply with STRICT JSON only (no markdown fences, no extra text):
  {"summary": "<brief>",
   "evidence_used": ["<ids/periods from EVIDENCE>"],
   "uncertainty": ["<what is uncertain>"],
   "limitations": ["<known limitations from EVIDENCE>"]}.
"""


def _field(payload: dict, key: str) -> str:
    value = payload.get(key)
    return str(value) if value is not None else "missing"


def _as_list(value):
    return value if isinstance(value, list) else []


def analyze_findings(payload: dict) -> list[Finding]:
    """Deterministic fact selection from an event/alert payload."""
    kind = payload.get("kind", "event")
    if kind == "alert":
        return [Finding(
            "alert-status",
            f"Alert {_field(payload, 'alert_id')} "
            f"({_field(payload, 'alert_type')}, "
            f"{_field(payload, 'severity')}, status "
            f"{_field(payload, 'status')}), period "
            f"{_field(payload, 'period')}.")]
    alerts = [a for a in _as_list(payload.get("alerts"))
              if isinstance(a, dict)]
    by_sev: dict[str, int] = {}
    for alert in alerts:
        sev = str(alert.get("severity", "unknown"))
        by_sev[sev] = by_sev.get(sev, 0) + 1
    sev_text = ", ".join(f"{n} {sev}" for sev, n in sorted(by_sev.items()))
    first = str(payload.get("first_observed_utc") or "missing")[:10]
    last = str(payload.get("last_observed_utc") or "missing")[:10]
    return [
        Finding(
            "event-status",
            f"Event {_field(payload, 'event_id')} is "
            f"{_field(payload, 'lifecycle_state')} with "
            f"{_field(payload, 'severity')} activity signal: "
            f"{_field(payload, 'detection_count')} observations from "
            f"{first} to {last}."),
        Finding("linked-alerts",
                f"{len(alerts)} linked alert(s)"
                + (f" ({sev_text})." if sev_text else ".")),
        Finding(
            "anomaly-refs",
            f"{len(_as_list(payload.get('anomaly_refs')))} anomaly "
            f"reference(s); "
            f"{len(_as_list(payload.get('anomalies')))} anomaly "
            f"detail(s) attached."),
    ]


def _fallback(payload: dict) -> str:
    kind = payload.get("kind", "event")
    if kind not in ("event", "alert"):
        raise ValueError(f"event agent cannot narrate kind {kind!r}")
    try:
        return qwen_mod.deterministic_fallback(kind, payload)
    except Exception:  # noqa: BLE001 - fallback must not raise
        return ("Event analysis unavailable: evidence payload "
                "unusable.")


def analyze(request: AgentRequest, chain) -> AgentResponse:
    """Run Agent 2 (kind must be 'event')."""
    if request.kind != "event":
        raise ValueError(f"event agent cannot handle {request.kind!r}")
    findings = analyze_findings(request.payload)
    return runner.run_llm_backed(
        agent_name=NAME, request=request, chain=chain, system=SYSTEM,
        findings=findings,
        fallback_text=_fallback(request.payload))
