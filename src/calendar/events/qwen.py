"""Stage 5E: optional Qwen interpretation layer (qwen_v1).

Qwen is an interpretation layer ONLY. It never computes metrics, FRP,
matching, clustering, event IDs, lifecycle states, alert severity, data
quality, perimeters, containment/extinction, spread/danger predictions,
or any modification of analytical outputs. Deterministic Python remains
authoritative for all of those; Qwen only rephrases supplied verified
evidence into concise natural language.

Architecture (no Qwen infrastructure existed in the repo — see the 5A
report — so this module is a dependency-free, optional client):
- Transport: OpenAI-style ``POST {base_url}/chat/completions`` via the
  already-pinned ``requests`` package. No new dependency, no new
  provider SDK.
- Switch: enabled only when CAL_QWEN_ENABLED is true AND base URL, API
  key, and model are all configured. Anything else -> status
  "unavailable", deterministic fallback, application unaffected.
- Evidence grounding: the model receives a constrained structured
  payload (verified fields only) and must return strict JSON with
  {summary, evidence_used, uncertainty, limitations}.
- Validation (deterministic, before display): JSON shape, required
  keys, every number traceable to the payload, every evidence_used entry
  traceable to a supplied identifier, banned-claim scan (containment,
  extinction, perimeters, predictions, danger ratings, evacuations).
  Any failure -> deterministic fallback. No silent display of invalid
  output.
- Every call is recorded in qwen_log (status, latency, model; errors
  sanitized) so failures are auditable.

The UI must label Qwen text "AI-generated interpretation" with evidence
references and limitations, and must distinguish "AI unavailable" from
"scientific data unavailable".
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from .. import settings
from .model import QWEN_VERSION

from . import store as store_mod

logger = logging.getLogger(__name__)

AI_LABEL = "AI-generated interpretation (Qwen)"
GRANITE_AI_LABEL = "AI-generated interpretation (IBM Granite)"
FALLBACK_LABEL = "Deterministic explanation (Qwen unavailable)"
GRANITE_FALLBACK_LABEL = "Deterministic explanation (IBM Granite unavailable)"
DETERMINISTIC_LABEL = "Deterministic explanation (rule-based)"

REQUIRED_KEYS = ("summary", "evidence_used", "uncertainty", "limitations")

# Claims Qwen must never make (case-insensitive substring scan). Any hit
# invalidates the response and triggers the deterministic fallback.
BANNED_PHRASES = (
    "contained", "containment", "extinguish", "perimeter",
    "evacuat", "danger rating", "fire danger", "spread prediction",
    "will spread", "predicted", "prediction", "forecast",
    "official nasa", "nasa assessment", "confirmed wildfire",
    "confirmed fire",
)

_NUM_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")

MAX_SUMMARY_CHARS = 2000


@dataclass(frozen=True)
class QwenConfig:
    """Effective Qwen configuration (resolved from settings)."""

    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_s: float = 20.0
    version: str = QWEN_VERSION

    @classmethod
    def from_settings(cls) -> "QwenConfig":
        enabled_raw = settings.get_setting("CAL_QWEN_ENABLED", "false")
        base_url = settings.get_setting("CAL_QWEN_BASE_URL", "").rstrip("/")
        api_key = settings.get_setting("CAL_QWEN_API_KEY", "")
        model = settings.get_setting("CAL_QWEN_MODEL", "")
        return cls(
            enabled=enabled_raw.lower() == "true",
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=settings.CAL_QWEN_TIMEOUT_S,
        )

    def status(self) -> dict:
        if not self.enabled:
            return {"available": False, "reason": "CAL_QWEN_ENABLED is not true"}
        missing = [name for name, value in
                   (("CAL_QWEN_BASE_URL", self.base_url),
                    ("CAL_QWEN_API_KEY", self.api_key),
                    ("CAL_QWEN_MODEL", self.model)) if not value]
        if missing:
            return {"available": False,
                    "reason": f"missing configuration: {', '.join(missing)}"}
        return {"available": True, "reason": f"model {self.model}",
                "model": self.model}


SYSTEM_PROMPT = """\
You interpret verified wildfire-observation evidence for analysts.
Rules (violations invalidate your answer):
- Use ONLY the supplied EVIDENCE payload. Never invent observations,
  metrics, dates, identifiers, or baselines.
- Distinguish: (a) verified observations, (b) statistical
  interpretation, (c) uncertainty, (d) missing or unavailable evidence.
- NEVER claim containment, extinction, fire perimeters, spread or danger
  predictions, evacuation guidance, or NASA authorship.
- NEVER recompute, correct, or override the supplied values.
- Reply with STRICT JSON only (no markdown fences, no extra text):
  {"summary": "<=150 words>",
   "evidence_used": ["<ids/periods from EVIDENCE>"],
   "uncertainty": ["<what is uncertain>"],
   "limitations": ["<known limitations from EVIDENCE>"]}.
"""


def build_event_payload(event: dict, alerts_list: list[dict] | None = None,
                        anomalies: list[dict] | None = None) -> dict:
    """Constrained evidence payload for one event (verified fields only)."""
    frp = event.get("frp") or {}
    return {
        "kind": "event",
        "event_id": event["event_id"],
        "lifecycle_state": event["lifecycle_state"],
        "severity": event.get("severity_name"),
        "first_observed_utc": event.get("first_observed_utc"),
        "last_observed_utc": event.get("last_observed_utc"),
        "detection_count": event.get("detection_count"),
        "sensor_modis_n": event.get("sensor_modis_n"),
        "sensor_viirs_n": event.get("sensor_viirs_n"),
        "satellites": event.get("satellites"),
        "cluster_ids": event.get("cluster_ids"),
        "frp": {"frp_sum": frp.get("frp_sum"),
                "frp_mean": frp.get("frp_mean"),
                "frp_max": frp.get("frp_max"),
                "frp_n": frp.get("frp_n"),
                "frp_null": frp.get("frp_null"),
                "aggregation": frp.get("aggregation")},
        "centroid": event.get("centroid"),
        "ambiguity_flags": (event.get("ambiguity") or {}).get("flags", []),
        "completeness": event.get("completeness"),
        "alerts": [{"alert_id": a["alert_id"], "type": a["alert_type"],
                    "severity": a["severity_name"],
                    "status": a["status"]} for a in (alerts_list or [])],
        "anomaly_refs": event.get("anomaly_refs", []),
        "anomalies": [{"anomaly_id": a.get("anomaly_id"),
                       "classification": a.get("classification"),
                       "observed_value": a.get("observed_value"),
                       "baseline_value": a.get("baseline_value")}
                      for a in (anomalies or [])],
        "limitations": [
            "Satellite observations are not confirmed fires.",
            "Cluster extent is an observation bounding box, not a perimeter.",
            "Resolved never means contained or extinguished.",
        ],
        "provenance": {"assoc": (event.get("provenance") or {}).get("assoc"),
                       "qwen_version": QWEN_VERSION},
    }


def build_alert_payload(alert: dict, event: dict | None = None) -> dict:
    """Constrained evidence payload for one alert (verified fields only)."""
    payload = {
        "kind": "alert",
        "alert_id": alert["alert_id"],
        "alert_type": alert["alert_type"],
        "severity": alert["severity_name"],
        "status": alert.get("status"),
        "scope": alert.get("scope"),
        "metric": alert.get("metric"),
        "period": alert.get("period"),
        "trigger": alert.get("trigger"),
        "baseline_ref": alert.get("baseline_ref"),
        "data_quality": alert.get("data_quality"),
        "rule_version": alert.get("rule_version"),
        "limitations": [
            "Activity signal severity is not a fire danger rating.",
            "Satellite observations are not confirmed ground truth.",
        ],
        "provenance": {"qwen_version": QWEN_VERSION},
    }
    if event is not None:
        payload["event"] = {
            "event_id": event["event_id"],
            "lifecycle_state": event["lifecycle_state"],
            "detection_count": event.get("detection_count"),
            "sensor_modis_n": event.get("sensor_modis_n"),
            "sensor_viirs_n": event.get("sensor_viirs_n"),
            "first_observed_utc": event.get("first_observed_utc"),
            "last_observed_utc": event.get("last_observed_utc"),
        }
    else:
        payload["event"] = None
    return payload


def _payload_numbers(payload: dict) -> list[float]:
    numbers: list[float] = []

    def _walk(node):
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _walk(value)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            numbers.append(float(node))

    _walk(payload)
    return numbers


def _payload_strings(payload: dict) -> list[str]:
    strings: list[str] = []

    def _walk(node):
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _walk(value)
        elif isinstance(node, str):
            strings.append(node)

    _walk(payload)
    return strings


def payload_ids(payload: dict) -> set[str]:
    """Traceable identifier set for one payload (public accessor)."""
    return _payload_ids(payload)


def _payload_ids(payload: dict) -> set[str]:
    ids: set[str] = set()

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("event_id", "alert_id", "anomaly_id",
                           "cluster_ids", "period", "eval_period",
                           "first_observed_utc", "last_observed_utc",
                           "method_version"):
                    if isinstance(value, list):
                        ids.update(str(v) for v in value)
                    elif value is not None:
                        ids.add(str(value))
                _walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                _walk(value)

    _walk(payload)
    return ids


def _check_numbers(text: str, payload: dict) -> list[float]:
    """Numbers in the text not traceable to the payload (tolerance-aware)."""
    claimed = []
    for match in _NUM_RE.finditer(text):
        try:
            claimed.append(float(match.group().replace(",", "")))
        except ValueError:
            continue
    if not claimed:
        return []
    available = _payload_numbers(payload)
    strings = " ".join(_payload_strings(payload))
    suspicious = []
    for value in claimed:
        raw_variants = {f"{value:g}", f"{value:,.0f}", f"{value:,.2f}"}
        if any(variant in strings for variant in raw_variants):
            continue  # dates, ids, and quoted values are citable verbatim
        tolerance = max(abs(value) * 0.01, 0.5)
        if not any(abs(value - known) <= tolerance for known in available):
            suspicious.append(value)
    return suspicious


def describe_checks() -> list[dict]:
    """Static guardrail catalog for the validation receipt (pure).

    Mirrors validate_response below: every listed check runs on every
    Qwen answer, and any failure falls back instead of displaying.
    """
    return [
        {"check": "JSON contract",
         "rule": "summary/evidence_used/uncertainty/limitations keys; "
                 "non-empty summary within the length limit"},
        {"check": "Banned-claim scan",
         "rule": f"{len(BANNED_PHRASES)} forbidden phrases: containment, "
                 "perimeters, spread/danger predictions, evacuation, "
                 "NASA authorship, confirmed wildfire"},
        {"check": "Evidence grounding",
         "rule": "every evidence_used id must come from the supplied "
                 "payload"},
        {"check": "Number grounding",
         "rule": "every number must match a payload value (1% tolerance) "
                 "or be quoted verbatim"},
    ]


def validate_response(parsed: dict, payload: dict) -> dict:
    """Deterministic validation of a Qwen JSON response.

    Returns {valid, reason}. Every failure mode falls back; nothing
    invalid is ever displayed as AI interpretation.
    """
    if not isinstance(parsed, dict):
        return {"valid": False, "reason": "response is not a JSON object"}
    missing = [key for key in REQUIRED_KEYS if key not in parsed]
    if missing:
        return {"valid": False,
                "reason": f"missing keys: {', '.join(missing)}"}
    summary = parsed["summary"]
    if not isinstance(summary, str) or not summary.strip():
        return {"valid": False, "reason": "empty summary"}
    if len(summary) > MAX_SUMMARY_CHARS:
        return {"valid": False, "reason": "summary exceeds length limit"}
    lowered = summary.lower()
    for banned in BANNED_PHRASES:
        if banned in lowered:
            return {"valid": False,
                    "reason": f"banned claim detected: {banned!r}"}
    for key in ("evidence_used", "uncertainty", "limitations"):
        if not isinstance(parsed[key], list) or not all(
                isinstance(item, str) for item in parsed[key]):
            return {"valid": False,
                    "reason": f"{key} must be a list of strings"}
    known_ids = _payload_ids(payload)
    for entry in parsed["evidence_used"]:
        if entry not in known_ids:
            return {"valid": False,
                    "reason": f"unsupported evidence reference: {entry!r}"}
    suspicious = _check_numbers(summary, payload)
    if suspicious:
        return {"valid": False,
                "reason": f"untraceable numbers: {suspicious}"}
    return {"valid": True, "reason": "all checks passed"}


def deterministic_fallback(kind: str, payload: dict) -> str:
    """Deterministic rule-based explanation used when LLMs are absent or fail."""
    if kind == "event":
        n_anom = len(payload.get("anomalies") or [])
        n_refs = len(payload.get("anomaly_refs") or [])
        lines = [
            f"Event {payload.get('event_id')} "
            f"({payload.get('lifecycle_state')}, "
            f"{payload.get('severity')} activity signal).",
            f"{payload.get('detection_count')} satellite observations "
            f"({payload.get('sensor_modis_n')} MODIS, "
            f"{payload.get('sensor_viirs_n')} VIIRS) from "
            f"{str(payload.get('first_observed_utc'))[:10]} to "
            f"{str(payload.get('last_observed_utc'))[:10]}.",
        ]
        frp = payload.get("frp") or {}
        if frp.get("frp_max") is not None:
            lines.append(f"Maximum observed FRP {frp['frp_max']} MW "
                         f"({frp.get('frp_n')} contributing observations, "
                         f"{frp.get('frp_null')} missing).")
        else:
            lines.append("FRP unavailable for this event (missing, not zero).")
        flags = payload.get("ambiguity_flags") or []
        if flags:
            lines.append("Evidence notes: " + ", ".join(flags) + ".")
        alerts_list = payload.get("alerts") or []
        if alerts_list:
            lines.append("Alerts: " + ", ".join(
                f"{a['alert_id']} ({a['severity']})" for a in alerts_list)
                + ".")
        else:
            lines.append("No alerts are attached to this event.")
        lines.append(
            "Rules applied: "
            f"RULE_LIFECYCLE_STATE={payload.get('lifecycle_state')}; "
            f"RULE_SEVERITY_SIGNAL={payload.get('severity')}; "
            "RULE_FRP_SEPARATION=MODIS/VIIRS kept separate (missing FRP never zeroed); "
            f"RULE_ANOMALY_LINKAGE={max(n_anom, n_refs)} anomaly ref(s)."
        )
    else:
        lines = [
            f"Alert {payload.get('alert_id')} "
            f"({payload.get('alert_type')}, {payload.get('severity')} "
            f"activity signal, status {payload.get('status')}).",
            f"Scope {payload.get('scope')}, metric {payload.get('metric')}, "
            f"period {payload.get('period')}.",
        ]
        trigger = payload.get("trigger") or {}
        lines.append("Trigger values: "
                     + json.dumps(trigger, sort_keys=True, default=str) + ".")
        event = payload.get("event")
        if event:
            lines.append(f"Associated event {event['event_id']} "
                         f"({event['lifecycle_state']}).")
        else:
            lines.append("Pilot-wide alert: no single event is implicated.")
        lines.append(
            "Rules applied: "
            f"RULE_ALERT_TYPE={payload.get('alert_type')}; "
            f"RULE_SEVERITY_SIGNAL={payload.get('severity')}; "
            f"RULE_STATUS={payload.get('status')}; "
            f"RULE_VERSION={payload.get('rule_version') or 'default'}."
        )
    lines.append("Limitations: " + " ".join(payload.get("limitations", [])))
    return "\n".join(lines)


def http_transport(prompt: str, config: QwenConfig) -> str:
    """Default transport: OpenAI-style chat completions (requests)."""
    import requests

    response = requests.post(
        f"{config.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key}",
                 "Content-Type": "application/json"},
        json={"model": config.model,
              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": prompt}],
              "temperature": 0.0,
              "max_tokens": 600},
        timeout=config.timeout_s)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _parse_and_validate_ai_json(raw: str, payload: dict) -> tuple[dict | None, str]:
    try:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        parsed = json.loads(clean)
    except (ValueError, AttributeError):
        return None, f"unparseable response: {str(raw)[:200]}"
    verdict = validate_response(parsed, payload)
    if not verdict["valid"]:
        return None, verdict["reason"]
    return parsed, verdict["reason"]


def _format_ai_sections(parsed: dict) -> str:
    return "\n".join([
        parsed["summary"],
        "",
        "Evidence used: " + ", ".join(parsed["evidence_used"]) + ".",
        "Uncertainty: " + " ".join(parsed["uncertainty"]),
        "Limitations: " + " ".join(parsed["limitations"]),
    ])


def explain(kind: str, target_id: str, payload: dict, *,
            config: QwenConfig | None = None,
            transport=http_transport,
            conn=None,
            ai_provider: str | None = None,
            granite_config=None,
            granite_post=None) -> dict:
    """Explain one event/alert with Qwen or IBM Granite, or fall back deterministically.

    Never raises for transport/validation failures: those return a
    fallback result (status != "ok") and are recorded in qwen_log when a
    connection is supplied. Returns {status, label, text, model,
    provider, latency_ms, validation, evidence_ids}.
    """
    if kind not in ("event", "alert"):
        raise ValueError(f"unknown explanation kind {kind!r}")
    config_supplied = config is not None
    config = config or QwenConfig.from_settings()
    status = config.status()
    evidence_ids = sorted(_payload_ids(payload))
    fallback_text = deterministic_fallback(kind, payload)

    mode = (ai_provider or "").strip().lower()
    if not mode:
        if config_supplied:
            mode = "qwen"
        else:
            env_mode = settings.get_setting("CAL_AI_PROVIDER", "auto").strip().lower()
            mode = env_mode if env_mode in ("auto", "granite", "qwen", "deterministic") else "auto"

    def _result(state: str, label: str, text: str, validation: str,
                latency_ms: float | None = None,
                *, model_name: str = "", provider_name: str = "deterministic") -> dict:
        result = {"status": state, "label": label, "text": text,
                  "model": model_name if state == "ok" else "",
                  "provider": provider_name if state == "ok" else "deterministic",
                  "requested_provider": mode,
                  "latency_ms": latency_ms, "validation": validation,
                  "evidence_ids": evidence_ids}
        if conn is not None:
            try:
                store_mod.append_qwen_log(
                    conn, target_type=kind, target_id=target_id,
                    status=state, latency_ms=latency_ms,
                    model=model_name if state == "ok" else "",
                    error="" if state == "ok" else validation[:500])
            except Exception as exc:  # noqa: BLE001 - logging never breaks
                logger.error("ai log failure: %s", str(exc)[:200])
        return result

    if mode == "deterministic":
        return _result("deterministic", DETERMINISTIC_LABEL, fallback_text,
                       "deterministic rules mode enabled",
                       provider_name="deterministic")

    prompt = ("EVIDENCE (JSON, authoritative):\n"
              + json.dumps(payload, sort_keys=True, default=str))

    def _run_qwen():
        if not status["available"]:
            return _result("disabled", FALLBACK_LABEL, fallback_text,
                           status["reason"], provider_name="qwen")
        started = time.monotonic()
        try:
            raw = transport(prompt, config)
        except Exception as exc:  # noqa: BLE001 - any transport failure
            message = str(exc)[:200].lower()
            state = "timeout" if "timeout" in message or "timed out" in message \
                else "error"
            logger.warning("qwen %s for %s: %s", state, target_id, str(exc)[:200])
            return _result(state, FALLBACK_LABEL, fallback_text,
                           f"{state}: {str(exc)[:200]}",
                           (time.monotonic() - started) * 1000.0,
                           provider_name="qwen")
        latency_ms = (time.monotonic() - started) * 1000.0
        parsed, reason = _parse_and_validate_ai_json(raw, payload)
        if parsed is None:
            if "banned claim" in reason or "unsupported" in reason or "untraceable" in reason:
                logger.warning("qwen invalid for %s: %s", target_id, reason)
            return _result("invalid", FALLBACK_LABEL, fallback_text,
                           reason, latency_ms, provider_name="qwen")
        return _result("ok", AI_LABEL, _format_ai_sections(parsed),
                       reason, latency_ms,
                       model_name=config.model, provider_name="qwen")

    def _run_granite():
        from ..agents import granite as granite_mod

        gp = granite_mod.GraniteProvider(config=granite_config, post=granite_post)
        usable, reason = gp.available()
        if not usable:
            return _result("disabled", GRANITE_FALLBACK_LABEL, fallback_text,
                           reason, provider_name="granite")
        attempt = gp.complete(prompt, SYSTEM_PROMPT)
        if not attempt.ok:
            state = "timeout" if "timeout" in (attempt.error or "").lower() else "error"
            return _result(state, GRANITE_FALLBACK_LABEL, fallback_text,
                           attempt.error or "granite request failed",
                           attempt.latency_ms, provider_name="granite")
        parsed, val_reason = _parse_and_validate_ai_json(attempt.text, payload)
        if parsed is None:
            logger.warning("granite invalid for %s: %s", target_id, val_reason)
            return _result("invalid", GRANITE_FALLBACK_LABEL, fallback_text,
                           val_reason, attempt.latency_ms, provider_name="granite")
        return _result("ok", GRANITE_AI_LABEL, _format_ai_sections(parsed),
                       val_reason, attempt.latency_ms,
                       model_name=gp.model, provider_name="granite")

    if mode == "granite":
        return _run_granite()
    if mode == "qwen":
        return _run_qwen()

    # mode == "auto": try Qwen when configured; otherwise fail over to IBM Granite,
    # then deterministic rules. Preserve Qwen failure logging only if Qwen was tried
    # and Granite is also unavailable.
    if status["available"]:
        qwen_res = _result_q = None
        started = time.monotonic()
        try:
            raw = transport(prompt, config)
            latency_ms = (time.monotonic() - started) * 1000.0
            parsed, reason = _parse_and_validate_ai_json(raw, payload)
            if parsed is not None:
                return _result("ok", AI_LABEL, _format_ai_sections(parsed),
                               reason, latency_ms,
                               model_name=config.model, provider_name="qwen")
            qwen_err = reason
        except Exception as exc:  # noqa: BLE001
            qwen_err = f"{type(exc).__name__}: {str(exc)[:160]}"
        granite_res = _run_granite()
        if granite_res["status"] == "ok":
            return granite_res
        return _result("error", FALLBACK_LABEL, fallback_text,
                       f"qwen failed ({qwen_err}); granite: {granite_res['validation']}",
                       provider_name="deterministic")

    granite_res = _run_granite()
    if granite_res["status"] == "ok":
        return granite_res
    return _result("disabled", FALLBACK_LABEL, fallback_text,
                   f"qwen: {status['reason']}; granite: {granite_res['validation']}",
                   provider_name="deterministic")
