"""Agent 3 — Scientific Observation Agent.

Inspects Observation Operator outputs: restates supplied aggregates,
flags unadjusted cross-sensor FRP displays, and lists unavailable
inputs. Applies fixed rules to supplied numbers only; never fits,
refits, or re-derives operator mathematics. Payload contract:
observed_sum/expected_sum (finite numbers), by_sensor (dict),
inputs_missing (list), method_version (short str).
"""

from __future__ import annotations

import math

from . import runner
from .contracts import AgentRequest, AgentResponse, Finding

NAME = "science-observer"

# Prompt rules vs enforcement: JSON schema, banned claims, evidence
# grounding, and number grounding are enforced by evidence_validator;
# the brevity ask below is guidance only and is NOT enforced.
SYSTEM = """\
You interpret Fire Radiative Power observation-operator evidence.
Rules (violations invalidate your answer):
- Use ONLY the supplied EVIDENCE payload. Never invent values.
- Expected retrieved FRP is a detection-discounted diagnostic, never
  an observed or corrected per-fire measurement. Never compare raw
  MODIS and VIIRS FRP as if commensurate. Never apply a universal
  sensor multiplier. Never infer counts, perimeters, or occurrence.
- evidence_used must list values copied from EVIDENCE (ids, periods, dates, versions as they appear); never key names.
- Never write any of these words anywhere in your answer, even in disclaimers: perimeter, perimeters, containment, contained, extinguish, extinguished. Say "observation extent" and "no activity observed" instead. Any occurrence invalidates the answer.
- Reply with STRICT JSON only (no markdown fences, no extra text):
  {"summary": "<brief>",
   "evidence_used": ["<ids/periods from EVIDENCE>"],
   "uncertainty": ["<what is uncertain>"],
   "limitations": ["<known limitations from EVIDENCE>"]}.
"""

HARD_LIMITATIONS = (
    "Expected retrieved FRP is E[retrieved | P] = p(P) x P, a "
    "detection-discounted diagnostic conditional on detection.",
    "No universal MODIS<->VIIRS multiplier exists; no x1.2 factor applied.",
    "Operator outputs never alter counts, clusters, events, or perimeters.",
)

MAX_VERSION_CHARS = 200


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def analyze_findings(payload: dict) -> list[Finding]:
    """Deterministic rule checks over an operator payload."""
    findings = []
    observed = _finite(payload.get("observed_sum"))
    expected = _finite(payload.get("expected_sum"))
    if observed is None or expected is None:
        findings.append(Finding("expected-vs-observed",
                                "Observed/expected aggregates missing or "
                                "non-numeric; no comparison stated."))
    elif observed == 0.0:
        findings.append(Finding(
            "expected-vs-observed",
            f"Observed sum is zero with expected {expected:,.1f} MW; "
            f"no percentage stated."))
    else:
        pct = (expected - observed) / observed * 100.0
        findings.append(Finding(
            "expected-vs-observed",
            f"Expected retrieved FRP {expected:,.1f} MW vs observed "
            f"{observed:,.1f} MW ({pct:+.2f}%); the gap is the "
            f"detection discount over {payload.get('n', 0)} supplied "
            f"observations."))
    by_sensor = payload.get("by_sensor")
    if isinstance(by_sensor, dict):
        sensors = sorted(str(k) for k in by_sensor)
        if len(sensors) >= 2 and not payload.get("mapping_applied"):
            findings.append(Finding(
                "cross-sensor-display",
                f"Observed FRP shown side by side for "
                f"{', '.join(sensors)} without operator mapping: not "
                f"comparable.", severity="caution"))
    else:
        findings.append(Finding("cross-sensor-display",
                                "Per-sensor breakdown missing; no "
                                "cross-sensor statement."))
    missing = payload.get("inputs_missing")
    if isinstance(missing, list) and missing:
        findings.append(Finding(
            "unavailable-inputs",
            "Unavailable inputs: "
            + ", ".join(str(v) for v in missing) + ".",
            severity="caution"))
    version = payload.get("method_version")
    if isinstance(version, str) and version.strip():
        findings.append(Finding(
            "method-version", f"Method {version[:MAX_VERSION_CHARS]}."))
    else:
        findings.append(Finding("method-version",
                                "Method version unstated.",
                                severity="caution"))
    return findings


def fallback_text(payload: dict, findings) -> str:
    """Deterministic template (infallible; static on assembly faults)."""
    try:
        body = " ".join(f.finding for f in findings)
        return ("Observation-operator review: " + body
                + "\nLimitations: " + " ".join(HARD_LIMITATIONS))
    except Exception:  # noqa: BLE001 - fallback must not raise
        return ("Science review unavailable: evidence payload "
                "unusable.")


def analyze(request: AgentRequest, chain) -> AgentResponse:
    """Run Agent 3 (kind must be 'science')."""
    if request.kind != "science":
        raise ValueError(f"science agent cannot handle {request.kind!r}")
    findings = analyze_findings(request.payload)
    return runner.run_llm_backed(
        agent_name=NAME, request=request, chain=chain, system=SYSTEM,
        findings=findings,
        fallback_text=fallback_text(request.payload, findings))
