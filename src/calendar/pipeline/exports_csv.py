"""Pipeline CSV exports (deterministic schemas, versioned).

Four deliverables: observation rows (FIRMS + operator results),
event rows, agent execution rows, and metrics rows. Nulls stay null
(empty fields, never zero); column order fixed; UTF-8 with BOM for
Excel parity with the existing export view.
"""

from __future__ import annotations

import csv
import io

SCHEMA_VERSION = "fireagent-pipeline-csv-v1"

OBSERVATION_COLUMNS = (
    "observation_id", "acquisition_utc", "sensor", "satellite",
    "product", "latitude", "longitude", "daynight_flag",
    "observed_frp_MW", "observed_frp_units", "scan_km", "track_km",
    "pixel_area_km2", "detection_limit_MW", "transition_rate_per_MW",
    "detection_probability", "expected_retrieved_frp_MW", "regime",
    "regime_margin_MW", "aod_550", "aod_source", "cloud_flag",
    "cloud_source", "l2_granule_id", "l2_match_status",
    "equation_version", "result_version", "calculation_status",
    "quality_flags", "input_completeness", "provenance",
    "processing_utc", "execution_id",
)

EVENT_COLUMNS = (
    "event_id", "cluster_ids", "date_start", "date_end",
    "lifecycle_state", "severity", "observation_count", "n_modis",
    "n_viirs", "satellites", "peak_observed_frp_MW",
    "frp_sum_MW", "frp_mean_MW", "frp_missing",
    "operator_expected_sum_MW", "operator_coverage",
    "input_completeness", "n_alerts", "alert_severities",
    "event_analysis_status", "evidence_validation_status",
    "provenance", "processing_utc", "execution_id",
)

AGENT_COLUMNS = (
    "execution_id", "agent", "target_id", "status", "provider",
    "model", "latency_ms", "retries", "response_chars",
    "input_tokens", "generated_tokens", "fallback",
    "fallback_reason", "validation", "error_category",
    "processing_utc",
)

METRICS_COLUMNS = (
    "execution_id", "metric", "value", "unit", "processing_utc",
)


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        import math

        if math.isnan(value) or math.isinf(value):
            return ""
    if value != value:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        import json

        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _table(rows: list[dict], columns: tuple[str, ...]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row.get(col)) for col in columns])
    return buffer.getvalue()


def to_bytes(text: str) -> bytes:
    """UTF-8 with BOM (Excel-compatible, matches export view)."""
    return ("\ufeff" + text).encode("utf-8")


def observation_csv(records: list[dict], *, provenance: str,
                    processing_utc: str, execution_id: str) -> str:
    """One row per operator result record (§7 contract keys)."""
    rows = []
    for record in records:
        rows.append({
            "observation_id": record.get("observation_id"),
            "acquisition_utc": record.get("acquisition_utc"),
            "sensor": record.get("sensor"),
            "satellite": record.get("satellite"),
            "product": record.get("product"),
            "latitude": record.get("latitude"),
            "longitude": record.get("longitude"),
            "daynight_flag": record.get("daynight_flag"),
            "observed_frp_MW": record.get("observed_frp"),
            "observed_frp_units": record.get("observed_frp_units"),
            "scan_km": record.get("scan_km"),
            "track_km": record.get("track_km"),
            "pixel_area_km2": record.get("pixel_area_km2"),
            "detection_limit_MW": record.get("detection_limit_MW"),
            "transition_rate_per_MW": record.get(
                "transition_rate_per_MW"),
            "detection_probability": record.get(
                "detection_probability"),
            "expected_retrieved_frp_MW": record.get(
                "expected_retrieved_frp_MW"),
            "regime": record.get("regime"),
            "regime_margin_MW": record.get("regime_margin_MW"),
            "aod_550": record.get("aod_550"),
            "aod_source": record.get("aod_source"),
            "cloud_flag": record.get("cloud_flag"),
            "cloud_source": record.get("cloud_source"),
            "l2_granule_id": record.get("l2_granule_id"),
            "l2_match_status": record.get(
                "l2_match_status", "unmatched"),
            "equation_version": record.get("equation_version"),
            "result_version": record.get("result_version"),
            "calculation_status": record.get("calculation_status"),
            "quality_flags": record.get("quality_flags"),
            "input_completeness": record.get("inputs_present"),
            "provenance": provenance,
            "processing_utc": processing_utc,
            "execution_id": execution_id,
        })
    return _table(rows, OBSERVATION_COLUMNS)


def event_csv(events: list[dict], *, provenance: str,
              processing_utc: str, execution_id: str) -> str:
    """One row per analyzed event (rollup + agent/operator status)."""
    rows = []
    for event in events:
        rows.append({
            "event_id": event.get("event_id"),
            "cluster_ids": event.get("cluster_ids"),
            "date_start": event.get("date_start"),
            "date_end": event.get("date_end"),
            "lifecycle_state": event.get("lifecycle_state"),
            "severity": event.get("severity"),
            "observation_count": event.get("observation_count"),
            "n_modis": event.get("n_modis"),
            "n_viirs": event.get("n_viirs"),
            "satellites": event.get("satellites"),
            "peak_observed_frp_MW": event.get("peak_frp"),
            "frp_sum_MW": event.get("frp_sum"),
            "frp_mean_MW": event.get("frp_mean"),
            "frp_missing": event.get("frp_missing"),
            "operator_expected_sum_MW": event.get("operator_expected"),
            "operator_coverage": event.get("operator_coverage"),
            "input_completeness": event.get("completeness"),
            "n_alerts": event.get("n_alerts"),
            "alert_severities": event.get("alert_severities"),
            "event_analysis_status": event.get("analysis_status"),
            "evidence_validation_status": event.get(
                "validation_status"),
            "provenance": provenance,
            "processing_utc": processing_utc,
            "execution_id": execution_id,
        })
    return _table(rows, EVENT_COLUMNS)


def agent_csv(runs: list[dict], *, processing_utc: str,
              execution_id: str) -> str:
    """One row per agent execution (telemetry + validation)."""
    rows = []
    for run in runs:
        status = run.get("status", "")
        rows.append({
            "execution_id": execution_id,
            "agent": run.get("agent"),
            "target_id": run.get("target_id"),
            "status": status,
            "provider": run.get("provider") or "",
            "model": run.get("model") or "",
            "latency_ms": round(float(run.get("latency_ms") or 0.0),
                                1),
            "retries": run.get("retries", 0),
            "response_chars": run.get("response_chars", 0),
            "input_tokens": run.get("input_tokens"),
            "generated_tokens": run.get("generated_tokens"),
            "fallback": status != "ok",
            "fallback_reason": run.get("fallback_reason") or "",
            "validation": run.get("validation") or "",
            "error_category": run.get("error_category") or (
                "none" if status == "ok" else "unclassified"),
            "processing_utc": processing_utc,
        })
    return _table(rows, AGENT_COLUMNS)


def metrics_csv(metrics: list[dict], *, processing_utc: str,
                execution_id: str) -> str:
    """One row per metric (name, value, unit)."""
    rows = [{
        "execution_id": execution_id,
        "metric": item.get("metric"),
        "value": item.get("value"),
        "unit": item.get("unit") or "",
        "processing_utc": processing_utc,
    } for item in metrics]
    return _table(rows, METRICS_COLUMNS)
