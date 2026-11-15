"""Stage 5C: deterministic alert engine (alert_rules_v1).

Transparent alerts from verified analytical conditions and event
evidence. No new anomaly methodology, no severity weights, no silent
threshold changes:

- Historical Anomaly / Cluster Anomaly alerts reuse Phase 4C
  classifications verbatim (|z| >= 2 AND |pct| >= 25% stays untouched).
  Sensor-specific and harmonized scopes stay distinguishable; global
  (pilot-wide) alerts carry no event ID rather than a fabricated link.
- Activity Spike / High FRP alerts reuse event evidence with
  distribution-derived bands (nearest-rank percentiles over the pinned
  2C cluster population, recorded in the registry). No magic MW or
  count constants.
- Data Quality alerts reuse catalog coverage verdicts; they are never
  represented as wildfire activity.

Stable alert IDs, idempotent upserts, configurable cooldowns, re-alert
only on material evidence change, acknowledgment/resolution with full
history. Missing FRP stays missing (no High FRP alert is possible
without an observed maximum).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .. import settings
from ..analytics import runs
from ..ui import query
from .model import (ALERT_RULES_VERSION, SEVERITY_LEVELS, check_alert_type,
                    check_severity)

from . import store as store_mod

logger = logging.getLogger(__name__)

# Nearest-rank percentile bands over the cluster population. Documented
# activity bands, not danger ratings: p99.9 ≈ the top ~138 clusters,
# p99 ≈ the top ~1,376, p95 ≈ the top ~6,882.
P_CRITICAL = 99.9
P_HIGH = 99.0
P_MODERATE = 95.0

# Re-alert suppression window after an alert resolves (days). A resolved
# alert reopens within the window only when its evidence fingerprint
# changes (a new severity band or new trigger values).
COOLDOWN_D = 7

# 4C classifications that justify an alert, mapped to the lowest
# defensible severity (spec: unjustified severity steps down, never up).
ANOMALY_SEVERITY = {
    "notable-high": "High",
    "provisional-notable-high": "Moderate",
    "notable-low": "Low",
    "provisional-notable-low": "Informational",
}

NON_ACTIONABLE = ("typical", "provisional-typical", "insufficient-evidence",
                  "unstable", "missing-data", "indeterminate")

ACTIVITY_CAVEAT = ("Satellite observations are not confirmed fires; repeated "
                   "observations of the same fire are included.")


@dataclass(frozen=True)
class AlertConfig:
    """Versioned alert policy (alert_rules_v1)."""

    version: str = ALERT_RULES_VERSION
    p_critical: float = P_CRITICAL
    p_high: float = P_HIGH
    p_moderate: float = P_MODERATE
    cooldown_d: int = COOLDOWN_D


def nearest_rank_threshold(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (deterministic; documented method).

    rank = ceil(p/100 * N); threshold = sorted[N-rank…]: the smallest
    value with at least (100-p)% of the population at or above it.
    """
    if not values:
        raise ValueError("cannot derive a band from an empty population")
    if not 0.0 < pct < 100.0:
        raise ValueError(f"percentile {pct} outside (0, 100)")
    ordered = sorted(values)
    import math

    rank = int(math.ceil(pct / 100.0 * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def band_for(value: float, bands: dict[str, float]) -> str | None:
    """Severity band for a value (None below the Moderate band)."""
    if value >= bands["Critical"]:
        return "Critical"
    if value >= bands["High"]:
        return "High"
    if value >= bands["Moderate"]:
        return "Moderate"
    return None


def fingerprint_alert(alert_type: str, severity: str,
                      trigger: dict) -> str:
    """Stable evidence fingerprint (drives re-alert vs suppression)."""
    payload = json.dumps({"t": alert_type, "s": severity, "v": trigger},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def alert_id_for(alert_type: str, key: str) -> str:
    """Stable deterministic alert ID (rule generation suffix r1)."""
    check_alert_type(alert_type)
    prefix = {"Activity Spike": "asp", "Cluster Anomaly": "can",
              "High FRP": "frp", "Historical Anomaly": "han",
              "Data Quality": "dqv"}[alert_type]
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-"
                   for c in key)[:120]
    return f"alt-{prefix}-{safe}-r1"


def explain_event_alert(alert_type: str, severity: str, event: dict,
                        trigger: dict, bands: dict[str, float],
                        population: int) -> str:
    """Deterministic human-readable alert explanation (English)."""
    lines = [
        f"{alert_type} alert ({severity} activity signal) for "
        f"{event['event_id']}.",
        f"Source cluster {event['cluster_ids'][0]}: "
        f"{event['detection_count']} satellite observations from "
        f"{event['first_observed_utc'][:10]} to "
        f"{event['last_observed_utc'][:10]}.",
        f"Sensor participation: {event['sensor_modis_n']} MODIS, "
        f"{event['sensor_viirs_n']} VIIRS.",
    ]
    if alert_type == "Activity Spike":
        lines.append(
            f"Trigger: {trigger['n_detections']} observations "
            f"(band thresholds over {population:,} clusters: Moderate >= "
            f"{bands['Moderate']:g}, High >= {bands['High']:g}, Critical >= "
            f"{bands['Critical']:g}).")
    else:
        lines.append(
            f"Trigger: maximum observed FRP {trigger['frp_max']} MW "
            f"(band thresholds over {population:,} clusters with observed "
            f"FRP: Moderate >= {bands['Moderate']:g} MW, High >= "
            f"{bands['High']:g} MW, Critical >= {bands['Critical']:g} MW).")
        by_sensor = trigger.get("by_sensor_max", {})
        if by_sensor:
            lines.append("Per-sensor maxima: "
                         + ", ".join(f"{k} {v} MW"
                                     for k, v in sorted(by_sensor.items()))
                         + " (MODIS and VIIRS FRP are not directly "
                         "comparable).")
    lines.append(f"Caveats: {ACTIVITY_CAVEAT}")
    lines.append("Cluster extent is an observation bounding box, not a fire "
                 "perimeter.")
    lines.append("Retrospective-only classification: bands derive from the "
                 "full available cluster population and are not validated "
                 "as a real-time alert signal.")
    return "\n".join(lines)


def explain_anomaly_alert(record: dict, severity: str) -> str:
    """Deterministic explanation for a 4C-derived alert (English)."""
    from ..analytics import confidence as conf_mod

    scope = record.get("scope", "")
    header = (f"{'Cluster' if record.get('metric') == 'cluster' else 'Historical'} "
              f"Anomaly alert ({severity} activity signal): "
              f"{record.get('metric')} / {scope}, period "
              f"{record.get('eval_period')}.")
    body = [
        f"Observed {record.get('observed_value')} vs baseline "
        f"{record.get('baseline_value')} "
        f"({record.get('baseline_method')}, "
        f"{record.get('reference_strategy')}, n={record.get('reference_n')}).",
        f"Classification {record.get('classification')} "
        f"(Phase 4 rule: |z| >= 2 and |pct| >= 25%).",
        f"Quality: {record.get('eligibility')}; z "
        f"{record.get('z_score')} ({record.get('z_status')}), pct "
        f"{record.get('pct_deviation')}% ({record.get('pct_status')}).",
        "Pilot-wide series result: no single event is implicated, so no "
        "event ID is attached.",
        conf_mod.DISCLAIMER,
    ]
    return "\n".join([header, *body])


def explain_data_quality_alert(kind: str, detail: dict) -> str:
    """Deterministic explanation for a data-quality alert (English)."""
    return "\n".join([
        f"Data Quality alert ({detail.get('severity', 'Low')}): {kind}.",
        detail.get("description", ""),
        "This is a data-availability notice, not a wildfire activity alert. "
        "Missing observations are never interpreted as zero activity.",
    ])


def _event_bands(conn, config: AlertConfig) -> dict:
    """Derive size/FRP bands from the event population (recorded)."""
    sizes: list[float] = []
    frp_maxima: list[float] = []
    offset = 0
    while True:
        batch = store_mod.list_events(conn, limit=5000, offset=offset,
                                          order="id")
        if not batch:
            break
        for event in batch:
            sizes.append(float(event["detection_count"]))
            maximum = (event.get("frp") or {}).get("frp_max")
            if maximum is not None:
                frp_maxima.append(float(maximum))
        offset += len(batch)
        if len(batch) < 5000:
            break
    size_bands = {
        "Critical": nearest_rank_threshold(sizes, config.p_critical),
        "High": nearest_rank_threshold(sizes, config.p_high),
        "Moderate": nearest_rank_threshold(sizes, config.p_moderate),
    }
    frp_bands = {
        "Critical": nearest_rank_threshold(frp_maxima, config.p_critical),
        "High": nearest_rank_threshold(frp_maxima, config.p_high),
        "Moderate": nearest_rank_threshold(frp_maxima, config.p_moderate),
    }
    return {"size": size_bands, "frp": frp_bands, "n_events": len(sizes),
            "n_frp": len(frp_maxima)}


def _by_sensor_max(event: dict) -> dict:
    out = {}
    for sensor, payload in ((event.get("frp") or {}).get("by_sensor",
                                                         {}) or {}).items():
        maximum = (payload or {}).get("frp_max")
        if maximum is not None:
            out[sensor] = maximum
    return out


def generate_event_alerts(conn, bands: dict, *, exec_id: str,
                          created_at: str) -> list[dict]:
    """Activity Spike + High FRP alerts (Moderate band and above only)."""
    config_bands = bands
    records = []
    offset = 0
    while True:
        batch = store_mod.list_events(conn, limit=5000, offset=offset,
                                          order="id")
        if not batch:
            break
        for event in batch:
            event_id = event["event_id"]
            size_band = band_for(float(event["detection_count"]),
                                 config_bands["size"])
            if size_band is not None:
                trigger = {"n_detections": event["detection_count"],
                           "bands": config_bands["size"],
                           "population": config_bands["n_events"]}
                records.append({
                    "alert_id": alert_id_for("Activity Spike", event_id),
                    "alert_type": "Activity Spike",
                    "severity_name": size_band, "event_id": event_id,
                    "scope": "event", "metric": "detection",
                    "grain": "event", "period": event["last_observed_utc"][:10],
                    "trigger": trigger, "baseline_ref": {
                        "method": "population-bands",
                        "bands": config_bands["size"],
                        "population": config_bands["n_events"]},
                    "data_quality": {
                        "ambiguity": event["ambiguity"]["flags"],
                        "coverage": event["completeness"].get(
                            "coverage_status", "unknown")},
                    "explanation": explain_event_alert(
                        "Activity Spike", size_band, event, trigger,
                        config_bands["size"], config_bands["n_events"]),
                })
            maximum = (event.get("frp") or {}).get("frp_max")
            if maximum is None:
                continue  # missing FRP stays missing: no alert possible
            frp_band = band_for(float(maximum), config_bands["frp"])
            if frp_band is not None:
                trigger = {"frp_max": maximum,
                           "frp_n": event["frp"].get("frp_n"),
                           "frp_null": event["frp"].get("frp_null"),
                           "by_sensor_max": _by_sensor_max(event),
                           "bands": config_bands["frp"],
                           "population": config_bands["n_frp"]}
                records.append({
                    "alert_id": alert_id_for("High FRP", event_id),
                    "alert_type": "High FRP",
                    "severity_name": frp_band, "event_id": event_id,
                    "scope": "event", "metric": "frp",
                    "grain": "event", "period": event["last_observed_utc"][:10],
                    "trigger": trigger, "baseline_ref": {
                        "method": "population-bands",
                        "bands": config_bands["frp"],
                        "population": config_bands["n_frp"]},
                    "data_quality": {
                        "frp_missing": event["frp"].get("frp_null"),
                        "coverage": event["completeness"].get(
                            "coverage_status", "unknown")},
                    "explanation": explain_event_alert(
                        "High FRP", frp_band, event, trigger,
                        config_bands["frp"], config_bands["n_frp"]),
                })
        offset += len(batch)
        if len(batch) < 5000:
            break
    for record in records:
        record["severity_level"] = SEVERITY_LEVELS[record["severity_name"]]
        record["created_at"] = created_at
        record["updated_at"] = created_at
        record["rule_version"] = ALERT_RULES_VERSION
        record["status"] = "open"
        record["fingerprint"] = fingerprint_alert(
            record["alert_type"], record["severity_name"], record["trigger"])
    return records


def generate_anomaly_alerts(anomalies: list[dict], *, exec_id: str,
                            created_at: str) -> list[dict]:
    """Historical/Cluster Anomaly alerts from 4C classifications (verbatim)."""
    _ = exec_id
    records = []
    for row in anomalies:
        classification = row.get("classification", "")
        if classification in NON_ACTIONABLE or \
                classification not in ANOMALY_SEVERITY:
            continue
        severity = ANOMALY_SEVERITY[classification]
        alert_type = ("Cluster Anomaly" if row.get("metric") == "cluster"
                      else "Historical Anomaly")
        trigger = {"observed_value": row.get("observed_value"),
                   "baseline_value": row.get("baseline_value"),
                   "pct_deviation": row.get("pct_deviation"),
                   "z_score": row.get("z_score"),
                   "classification": classification}
        records.append({
            "alert_id": alert_id_for(alert_type, row["anomaly_id"]),
            "alert_type": alert_type, "severity_name": severity,
            "severity_level": SEVERITY_LEVELS[severity], "event_id": None,
            "scope": row.get("scope", ""), "metric": row.get("metric", ""),
            "grain": row.get("grain", "") or "",
            "period": row.get("eval_period", ""),
            "trigger": trigger,
            "baseline_ref": {
                "anomaly_id": row["anomaly_id"],
                "baseline_method": row.get("baseline_method"),
                "reference_strategy": row.get("reference_strategy"),
                "reference_start": row.get("reference_start"),
                "reference_end": row.get("reference_end"),
                "reference_n": row.get("reference_n"),
                "eligibility": row.get("eligibility"),
                "anomaly_exec": row.get("exec_id", "")},
            "data_quality": {
                "eligibility": row.get("eligibility"),
                "pct_status": row.get("pct_status"),
                "z_status": row.get("z_status"),
                "limitations": row.get("limitations", [])},
            "explanation": explain_anomaly_alert(row, severity),
            "created_at": created_at, "updated_at": created_at,
            "rule_version": ALERT_RULES_VERSION, "status": "open",
            "fingerprint": fingerprint_alert(alert_type, severity, trigger),
        })
    records.sort(key=lambda r: r["alert_id"])
    return records


def generate_data_quality_alerts(*, exec_id: str,
                                 created_at: str) -> list[dict]:
    """Coverage/sensor-gap alerts from catalog verdicts (honest zero ok)."""
    _ = exec_id
    catalog = json.loads(Path(settings.CATALOG_PATH).read_text(
        encoding="utf-8"))
    spans = query.catalog_spans(catalog)
    records = []
    # Sensor-gap days: dates inside one dataset's span but outside another's.
    if len(spans) >= 2:
        items = sorted(spans.items())
        for day in _daterange(items):
            inside = [ds for ds, span in items
                      if span["min"] <= day <= span["max"]]
            if 0 < len(inside) < len(items):
                missing = sorted(set(spans) - set(inside))
                detail = {"severity": "Low",
                          "description":
                          f"{day}: only {', '.join(inside)} have verified "
                          f"coverage; missing {', '.join(missing)}."}
                trigger = {"day": day, "covered": inside, "missing": missing}
                records.append({
                    "alert_id": alert_id_for("Data Quality", f"gap-{day}"),
                    "alert_type": "Data Quality", "severity_name": "Low",
                    "severity_level": SEVERITY_LEVELS["Low"], "event_id": None,
                    "scope": "coverage", "metric": "availability",
                    "grain": "daily", "period": day, "trigger": trigger,
                    "baseline_ref": {"spans": spans},
                    "data_quality": {"covered": inside, "missing": missing},
                    "explanation": explain_data_quality_alert(
                        "Partial sensor coverage", detail),
                    "created_at": created_at, "updated_at": created_at,
                    "rule_version": ALERT_RULES_VERSION, "status": "open",
                    "fingerprint": fingerprint_alert("Data Quality", "Low",
                                                     trigger),
                })
    records.sort(key=lambda r: r["alert_id"])
    return records


def _daterange(items: list[tuple[str, dict]]):
    from datetime import date as _date, timedelta as _td

    lo = min(span["min"] for _, span in items)
    hi = max(span["max"] for _, span in items)
    day = _date.fromisoformat(lo)
    stop = _date.fromisoformat(hi)
    while day <= stop:
        yield day.isoformat()
        day += _td(days=1)


def persist_alerts(conn, records: list[dict], *, exec_id: str,
                   created_at: str) -> dict:
    """Idempotent persist with dedup + re-alert-on-material-change.

    - Unknown alert_id -> insert + "created" history.
    - Known id, same fingerprint -> no-op (duplicate suppressed, no new
      history row: re-running generation is silent).
    - Known id, changed fingerprint -> update trigger/severity/evidence +
      "re-alert" history (open) or reopen (resolved, cooldown-aware).
    - Alert history is never erased.
    """
    from datetime import date as _date

    stats = {"created": 0, "suppressed": 0, "re_alerted": 0, "reopened": 0}
    today = _date.fromisoformat(created_at[:10])
    # Chunked commits: one transaction per 1000 alerts (fast, resumable).
    for chunk_start in range(0, max(len(records), 1), 1000):
        chunk = records[chunk_start:chunk_start + 1000]
        with store_mod.batch(conn):
            _persist_chunk(conn, chunk, exec_id, created_at, today, stats)
    return stats


def _persist_chunk(conn, chunk, exec_id, created_at, today, stats) -> None:
    """Persist one chunk (caller owns the transaction)."""
    from datetime import date as _date

    for record in chunk:
        check_severity(record["severity_name"])
        current = store_mod.get_alert(conn, record["alert_id"])
        if current is None:
            store_mod.upsert_alert(conn, record)
            store_mod.append_alert_history(
                conn, record["alert_id"], "created",
                {"severity": record["severity_name"]}, exec_id, created_at)
            stats["created"] += 1
            continue
        if current["fingerprint"] == record["fingerprint"]:
            stats["suppressed"] += 1
            continue
        if current["status"] == "resolved":
            resolved_day = _date.fromisoformat(
                (current["resolved_at"] or created_at)[:10])
            if (today - resolved_day).days < COOLDOWN_D:
                store_mod.append_alert_history(
                    conn, record["alert_id"], "suppressed-cooldown",
                    {"severity": record["severity_name"]}, exec_id,
                    created_at)
                stats["suppressed"] += 1
                continue
            record = {**record, "status": "open", "resolved_at": None,
                      "resolve_note": ""}
            stats["reopened"] += 1
        else:
            record = {**record, "status": current["status"],
                      "ack_at": current["ack_at"],
                      "ack_note": current["ack_note"],
                      "resolved_at": current["resolved_at"],
                      "resolve_note": current["resolve_note"]}
            stats["re_alerted"] += 1
        record["created_at"] = current["created_at"]
        store_mod.upsert_alert(conn, record)
        store_mod.append_alert_history(
            conn, record["alert_id"], "re-alert",
            {"severity": record["severity_name"],
             "previous_severity": current["severity_name"]},
            exec_id, created_at)


def sync_event_severity(conn, *, exec_id: str,
                        created_at: str = "") -> dict:
    """Set each alerted event's severity to its max open alert severity.

    Events without alerts keep Informational. Severity provenance names
    the winning alert. Safety interaction with 5B: an event that already
    reached Resolved while still Informational, but now classifies as
    High/Critical without an approved review, is escalated back to
    Pending Closure (audited transition) so the 5B review gate cannot be
    bypassed by stage ordering. Returns {updated, escalated, max_level}.
    """
    from .lifecycle import REVIEW_SEVERITIES

    rows = conn.execute(
        "SELECT event_id, MAX(severity_level) AS m FROM alerts "
        "WHERE event_id IS NOT NULL AND status <> 'resolved' "
        "GROUP BY event_id").fetchall()
    updated = 0
    escalated = 0
    pending: list = []

    def _flush() -> None:
        nonlocal updated, escalated
        with store_mod.batch(conn):
            for item, level, name, winner_id in pending:
                store_mod.update_event_fields(
                    conn, item["event_id"],
                    {"severity_name": name,
                     "severity_provenance":
                     "max open alert severity via "
                     f"{winner_id or 'unknown'} "
                     f"({ALERT_RULES_VERSION}, exec {exec_id[:8]})"})
                updated += 1
                if (item["lifecycle_state"] == "Resolved" and name in
                        REVIEW_SEVERITIES and item.get("review_status")
                        != "approved"):
                    store_mod.record_transition(
                        conn, event_id=item["event_id"],
                        prev_state="Resolved",
                        new_state="Pending Closure",
                        reason="severity-escalation-review",
                        evidence={"severity": name,
                                  "via_alert": winner_id,
                                  "rule_version": ALERT_RULES_VERSION},
                        completeness=item.get("completeness", {}),
                        algo_version=ALERT_RULES_VERSION,
                        config_version=ALERT_RULES_VERSION,
                        review_required=True, exec_id=exec_id,
                        at_utc=created_at)
                    escalated += 1
        pending.clear()

    for row in rows:
        event = store_mod.get_event(conn, row["event_id"])
        if event is None:
            continue
        level = int(row["m"])
        name = {v: k for k, v in SEVERITY_LEVELS.items()}[level]
        winner = conn.execute(
            "SELECT alert_id FROM alerts WHERE event_id = ? AND "
            "severity_level = ? AND status <> 'resolved' "
            "ORDER BY alert_id LIMIT 1",
            (row["event_id"], level)).fetchone()
        pending.append((event, level, name,
                        winner["alert_id"] if winner else None))
        if len(pending) >= 1000:
            _flush()
    _flush()
    top = conn.execute("SELECT MAX(severity_level) AS m FROM events"
                       ).fetchone()
    return {"updated": updated, "escalated": escalated,
            "max_level": top["m"]}


def run_alerts(db_path: str, runs_path: str) -> dict:
    """Generate all alert categories (idempotent)."""
    from datetime import datetime, timezone

    t0 = time.monotonic()
    config = AlertConfig()
    created_at = datetime.now(timezone.utc).isoformat()
    conn = store_mod.connect(db_path)
    store_mod.init_schema(conn)
    exec_id = runs.new_execution(
        runs_path, stage="5C",
        inputs={"db": db_path, "anomalies": str(
            Path(settings.CAL_ANALYTICS_ROOT) / settings.CAL_EXEC_4C
            / "anomalies_4c.json"),
                "catalog": settings.CATALOG_PATH},
        params={"percentiles": [config.p_moderate, config.p_high,
                                config.p_critical],
                "cooldown_d": config.cooldown_d,
                "anomaly_map": ANOMALY_SEVERITY,
                "algo": config.version})
    bands = _event_bands(conn, config)
    store_mod.meta_set(conn, "alert_bands", bands)
    event_records = generate_event_alerts(conn, bands, exec_id=exec_id,
                                          created_at=created_at)
    anomalies = query.load_anomalies()
    anomaly_records = generate_anomaly_alerts(anomalies, exec_id=exec_id,
                                              created_at=created_at)
    dq_records = generate_data_quality_alerts(exec_id=exec_id,
                                              created_at=created_at)
    stats = persist_alerts(conn, event_records + anomaly_records
                           + dq_records, exec_id=exec_id, created_at=created_at)
    severity = sync_event_severity(conn, exec_id=exec_id,
                                  created_at=created_at)
    rows = conn.execute("SELECT alert_type, severity_name, COUNT(*) AS n "
                        "FROM alerts GROUP BY alert_type, severity_name "
                        "ORDER BY alert_type, severity_name").fetchall()
    by_type = [dict(row) for row in rows]
    store_mod.meta_set(conn, "exec_5c", exec_id)
    outputs = {"bands": bands,
               "generated": {"event": len(event_records),
                             "anomaly": len(anomaly_records),
                             "data_quality": len(dq_records)},
               "persist": stats, "severity_sync": severity,
               "by_type_severity": by_type,
               "elapsed_s": round(time.monotonic() - t0, 1)}
    runs.complete_execution(runs_path, exec_id, outputs)
    logger.info("5C complete: %s in %ss", stats, outputs["elapsed_s"])
    return {"exec_id": exec_id, **outputs}
