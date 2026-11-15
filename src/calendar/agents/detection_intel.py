"""Agent 1 — Fire Detection Intelligence Agent.

Analyzes verified observation payloads (cluster timelines): selects
salient facts (peak day, span, sensor split, FRP availability) and
narrates them. Selects and formats supplied aggregates only; never
recomputes measurements and never calls hotspots confirmed fires.
Payload contract: cluster_id (str), n_detections (number); timeline
rows {day, n, n_modis, n_viirs, frp_missing} read defensively.
"""

from __future__ import annotations

from . import runner
from .contracts import AgentRequest, AgentResponse, Finding

NAME = "detection-intelligence"

# Prompt rules vs enforcement: JSON schema, banned claims, evidence
# grounding, and number grounding are enforced by evidence_validator;
# the brevity ask below is guidance only and is NOT enforced.
SYSTEM = """\
You interpret verified satellite fire-observation evidence for analysts.
Rules (violations invalidate your answer):
- Use ONLY the supplied EVIDENCE payload. Never invent observations,
  metrics, dates, identifiers, or baselines.
- Hotspots are satellite observations, never independent confirmed fires.
- Keep MODIS and VIIRS identities separate; never pool their FRP.
- evidence_used must list values copied from EVIDENCE (ids, periods, dates, versions as they appear); never key names.
- Never write any of these words anywhere in your answer, even in disclaimers: perimeter, perimeters, containment, contained, extinguish, extinguished. Say "observation extent" and "no activity observed" instead. Any occurrence invalidates the answer.
- Reply with STRICT JSON only (no markdown fences, no extra text):
  {"summary": "<brief>",
   "evidence_used": ["<ids/periods from EVIDENCE>"],
   "uncertainty": ["<what is uncertain>"],
   "limitations": ["<known limitations from EVIDENCE>"]}.
"""


def _num(value, default=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return value


def _str_list(value):
    if not isinstance(value, list) or not all(
            isinstance(v, str) for v in value):
        return []
    return value


def analyze_findings(payload: dict) -> list[Finding]:
    """Deterministic fact selection from a detection payload."""
    findings = [Finding(
        "observation-count",
        f"{_num(payload.get('n_detections'))} satellite observations "
        f"({_num(payload.get('n_modis'))} MODIS, "
        f"{_num(payload.get('n_viirs'))} VIIRS).")]
    raw_timeline = payload.get("timeline")
    rows = [r for r in raw_timeline
            if isinstance(r, dict) and isinstance(r.get("day"), str)]
    if rows and isinstance(raw_timeline, list):
        peak = max(rows, key=lambda r: _num(r.get("n")))
        days = sorted(r["day"] for r in rows)
        findings.append(Finding(
            "temporal-span",
            f"Observed on {len(days)} day(s), {days[0]} to {days[-1]}; "
            f"peak {peak['day']} with {_num(peak.get('n'))} "
            f"observations."))
        missing = sum(_num(r.get("frp_missing")) for r in rows)
        findings.append(Finding(
            "frp-availability",
            f"{missing} member(s) missing FRP (excluded, never zero)."))
    else:
        findings.append(Finding("temporal-span",
                                "No usable observation days supplied."))
    sats = _str_list(payload.get("satellites"))
    if sats:
        findings.append(Finding("sensor-identity",
                                "Contributing satellites: "
                                + ", ".join(sats) + "."))
    return findings


def fallback_text(payload: dict, findings) -> str:
    """Deterministic template (infallible; static on assembly faults)."""
    try:
        body = " ".join(f.finding for f in findings)
        limits = _str_list(payload.get("limitations"))
        text = f"Cluster {payload.get('cluster_id')}: {body}"
        if limits:
            text += "\nLimitations: " + " ".join(limits)
        return text
    except Exception:  # noqa: BLE001 - fallback must not raise
        return ("Detection analysis unavailable: evidence payload "
                "unusable.")


def analyze(request: AgentRequest, chain) -> AgentResponse:
    """Run Agent 1 (kind must be 'detections')."""
    if request.kind != "detections":
        raise ValueError(f"detection agent cannot handle {request.kind!r}")
    findings = analyze_findings(request.payload)
    return runner.run_llm_backed(
        agent_name=NAME, request=request, chain=chain, system=SYSTEM,
        findings=findings,
        fallback_text=fallback_text(request.payload, findings))
