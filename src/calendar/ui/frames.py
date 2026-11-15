"""Typed UI table builders + Arrow-compatibility validation.

Every DataFrame rendered by the Calendar UI goes through an explicit
schema here: pandas nullable dtypes (Float64/Int64/boolean/string),
nulls preserved, numerics never stringified into their typed column.
check_arrow_compatible() runs at construction/test boundaries (not on
hot render paths) and names the offending table and column.
"""

from __future__ import annotations

import pandas as pd


def _as_float64(values) -> pd.Series:
    """Numbers to Float64; anything else (incl. bool) becomes null."""
    cleaned = [float(v) if isinstance(v, (int, float))
               and not isinstance(v, bool) else None for v in values]
    return pd.Series(cleaned, dtype="Float64")


def _as_int64(values) -> pd.Series:
    """Integers to Int64; anything else becomes null."""
    cleaned = [int(v) if isinstance(v, int)
               and not isinstance(v, bool) else None for v in values]
    return pd.Series(cleaned, dtype="Int64")


def _as_string(values) -> pd.Series:
    """Verbatim strings; None stays null (pd.NA)."""
    cleaned = [None if v is None else str(v) for v in values]
    return pd.Series(cleaned, dtype="string")


def _as_boolean(values) -> pd.Series:
    """Booleans to boolean; anything else becomes null."""
    cleaned = [bool(v) if isinstance(v, bool) else None
               for v in values]
    return pd.Series(cleaned, dtype="boolean")


def stages_frame(stages) -> pd.DataFrame:
    """Pipeline stage table (latency numeric, rest strings)."""
    rows = [{"name": s.name, "status": s.status,
             "latency_ms": s.latency_ms, "detail": s.detail}
            for s in stages]
    return pd.DataFrame({
        "stage": _as_string([r["name"] for r in rows]),
        "status": _as_string([r["status"] for r in rows]),
        "latency_ms": _as_float64([r["latency_ms"] for r in rows]),
        "detail": _as_string([r["detail"] for r in rows]),
    })


def agent_runs_frame(runs: list[dict]) -> pd.DataFrame:
    """Agent execution table (telemetry numeric, labels strings)."""
    return pd.DataFrame({
        "agent": _as_string([r.get("agent") for r in runs]),
        "target": _as_string([r.get("target_id") for r in runs]),
        "status": _as_string([r.get("status") for r in runs]),
        "provider": _as_string([r.get("provider") or "—"
                                for r in runs]),
        "model": _as_string([r.get("model") or "—" for r in runs]),
        "latency_ms": _as_float64([r.get("latency_ms") for r in runs]),
        "retries": _as_int64([r.get("retries", 0) for r in runs]),
        "tokens_in": _as_int64([r.get("input_tokens") for r in runs]),
        "tokens_out": _as_int64([r.get("generated_tokens")
                                 for r in runs]),
        "validation": _as_string([(r.get("validation") or "")[:80]
                                  for r in runs]),
    })


def metrics_frame(metrics: list[dict]) -> pd.DataFrame:
    """Pipeline metrics table (heterogeneous by design, typed split).

    `text_value` carries every value verbatim as display text;
    `numeric_value` carries the number for numeric metrics and null
    otherwise, so provider names such as 'granite' never enter a
    numeric column and real numbers stay numeric.
    """
    values = [m.get("value") for m in metrics]
    return pd.DataFrame({
        "metric": _as_string([m.get("metric") for m in metrics]),
        "numeric_value": _as_float64(values),
        "text_value": _as_string(values),
        "unit": _as_string([m.get("unit") or "" for m in metrics]),
    })


def operator_sensor_frame(by_sensor: dict) -> pd.DataFrame:
    """Per-sensor operator aggregates (observed vs expected, MW)."""
    sensors = sorted(by_sensor)
    return pd.DataFrame({
        "sensor": _as_string(sensors),
        "n": _as_int64([by_sensor[s].get("n") for s in sensors]),
        "observed_sum_MW": _as_float64(
            [by_sensor[s].get("observed_sum_MW") for s in sensors]),
        "expected_sum_MW": _as_float64(
            [by_sensor[s].get("expected_sum_MW") for s in sensors]),
    })


def events_frame(events: list[dict]) -> pd.DataFrame:
    """Event rollup table (counts numeric, labels/dates strings)."""
    return pd.DataFrame({
        "event": _as_string([e.get("event_id") for e in events]),
        "state": _as_string([e.get("lifecycle_state") for e in events]),
        "severity": _as_string([e.get("severity_name") for e in events]),
        "first": _as_string([str(e.get("first_observed_utc") or "")[:10]
                             for e in events]),
        "last": _as_string([str(e.get("last_observed_utc") or "")[:10]
                            for e in events]),
        "obs": _as_int64([e.get("detection_count") for e in events]),
        "MODIS": _as_int64([e.get("sensor_modis_n") for e in events]),
        "VIIRS": _as_int64([e.get("sensor_viirs_n") for e in events]),
    })


def lifecycle_frame(history: list[dict]) -> pd.DataFrame:
    """Lifecycle transition table (all strings)."""
    return pd.DataFrame({
        "at": _as_string([str(h.get("at_utc") or "")[:10]
                          for h in history]),
        "transition": _as_string(
            [f"{h.get('prev_state')} → {h.get('new_state')}"
             for h in history]),
        "reason": _as_string([h.get("reason") for h in history]),
    })


def reviews_frame(reviews: list[dict]) -> pd.DataFrame:
    """Human review table (all strings)."""
    return pd.DataFrame({
        "at": _as_string([r.get("at_utc") for r in reviews]),
        "reviewer": _as_string([r.get("reviewer_label")
                                for r in reviews]),
        "decision": _as_string([r.get("decision") for r in reviews]),
        "note": _as_string([r.get("note") for r in reviews]),
    })


def alerts_list_frame(alerts: list[dict]) -> pd.DataFrame:
    """Alert list table (all strings; event falls back to pilot-wide)."""
    return pd.DataFrame({
        "alert": _as_string([a.get("alert_id") for a in alerts]),
        "type": _as_string([a.get("alert_type") for a in alerts]),
        "severity": _as_string([a.get("severity_name")
                                for a in alerts]),
        "period": _as_string([a.get("period") for a in alerts]),
        "event": _as_string([a.get("event_id") or "pilot-wide"
                             for a in alerts]),
    })


def linked_alerts_frame(alerts: list[dict]) -> pd.DataFrame:
    """Event-linked alert table (all strings)."""
    return pd.DataFrame({
        "alert": _as_string([a.get("alert_id") for a in alerts]),
        "type": _as_string([a.get("alert_type") for a in alerts]),
        "severity": _as_string([a.get("severity_name")
                                for a in alerts]),
        "status": _as_string([a.get("status") for a in alerts]),
    })


def alert_history_frame(history: list[dict]) -> pd.DataFrame:
    """Alert action history (all strings)."""
    return pd.DataFrame({
        "at": _as_string([h.get("at_utc") for h in history]),
        "action": _as_string([h.get("action") for h in history]),
    })


def notifications_frame(rows: list[dict]) -> pd.DataFrame:
    """Delivery history (labels strings, attempts numeric)."""
    return pd.DataFrame({
        "at": _as_string([str(r.get("created_at") or "")[:16]
                          for r in rows]),
        "alert": _as_string([r.get("alert_id") for r in rows]),
        "kind": _as_string([r.get("kind") for r in rows]),
        "status": _as_string([r.get("status") for r in rows]),
        "attempts": _as_int64([r.get("attempts") for r in rows]),
    })


def custom_rules_frame(rules: list[dict],
                       summaries: list[str]) -> pd.DataFrame:
    """Custom rule table (labels strings, flags booleans)."""
    return pd.DataFrame({
        "rule": _as_string([r.get("rule_id") for r in rules]),
        "name": _as_string([r.get("name") for r in rules]),
        "condition": _as_string(summaries),
        "severity": _as_string([r.get("severity_name")
                                for r in rules]),
        "enabled": _as_boolean([bool(r.get("enabled"))
                                for r in rules]),
        "notify": _as_boolean([bool(r.get("notify")) for r in rules]),
    })


def custom_evals_frame(evals: list[dict]) -> pd.DataFrame:
    """Custom evaluation table (triggered boolean, rest strings)."""
    return pd.DataFrame({
        "at": _as_string([str(e.get("at_utc") or "")[:19]
                          for e in evals]),
        "triggered": _as_boolean([bool(e.get("triggered"))
                                  for e in evals]),
        "reason": _as_string([e.get("reason") for e in evals]),
    })


def custom_hits_frame(hits: list[dict]) -> pd.DataFrame:
    """Custom hit table (all strings)."""
    return pd.DataFrame({
        "hit": _as_string([h.get("hit_id") for h in hits]),
        "rule": _as_string([h.get("rule_name") for h in hits]),
        "period": _as_string([h.get("period") for h in hits]),
        "severity": _as_string([h.get("severity_name")
                                for h in hits]),
        "status": _as_string([h.get("status") for h in hits]),
        "summary": _as_string([h.get("summary") for h in hits]),
    })


def notables_frame(rows: list[dict], fmt) -> pd.DataFrame:
    """Analytics notables (observed numeric, stats formatted strings).

    `fmt` formats baseline/pct/z values (unavailable stays a string,
    never zero); observed stays numeric for correct sorting.
    """
    return pd.DataFrame({
        "period": _as_string([r.get("eval_period") for r in rows]),
        "observed": _as_float64([r.get("observed_value")
                                 for r in rows]),
        "baseline": _as_string([fmt(r.get("baseline_value"))
                                for r in rows]),
        "pct_%": _as_string([fmt(r.get("pct_deviation"))
                             for r in rows]),
        "z": _as_string([fmt(r.get("z_score")) for r in rows]),
        "class": _as_string([r.get("classification") for r in rows]),
    })


def about_sources_frame(rows: list[dict]) -> pd.DataFrame:
    """About-page source table (all strings)."""
    return pd.DataFrame({
        "Feature": _as_string([r.get("Feature") for r in rows]),
        "Source": _as_string([r.get("Source") for r in rows]),
        "Execution": _as_string([r.get("Execution") for r in rows]),
    })


def check_arrow_compatible(df: pd.DataFrame, table: str) -> list[str]:
    """Validate Arrow conversion; [] when clean, diagnostics if not.

    On failure, converts column-by-column to name the offenders.
    Read-only on the input (no mutation, no blanket astype).
    """
    import pyarrow as pa

    try:
        pa.Table.from_pandas(df, preserve_index=False)
        return []
    except Exception as exc:  # noqa: BLE001 - diagnosed below
        root = f"{table}: Arrow conversion failed: {exc}"
    findings = [root]
    for column in df.columns:
        try:
            pa.Table.from_pandas(df[[column]], preserve_index=False)
        except Exception as exc:  # noqa: BLE001 - per-column probe
            sample = df[column].dropna().head(3).tolist()
            kinds = sorted({type(v).__name__ for v in sample})
            findings.append(
                f"{table}.{column}: dtype={df[column].dtype}, "
                f"python types={kinds}, sample={sample!r}: {exc}")
    return findings
