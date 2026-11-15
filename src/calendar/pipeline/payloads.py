"""Evidence payload builders for pipeline agent runs (read-only).

Single implementation shared by the pipeline runner and any UI entry
point. Builders return None when their source evidence is missing.
"""

from __future__ import annotations

from pathlib import Path

SCIENCE_MISSING_INPUTS = ("pixel QA flags", "cloud fraction",
                          "overpass records",
                          "per-pixel scan geometry")


def build_detection_payload(cluster_id: str) -> dict | None:
    """Agent payload for one cluster (timeline via existing helpers)."""
    from ..ui import map_view, query

    clu = query.cluster_by_id(cluster_id)
    if clu is None:
        return None
    frame = query.detections_of_cluster(cluster_id)
    timeline = map_view.cluster_timeline(frame)
    return {
        "cluster_id": cluster_id, "cluster_ids": [cluster_id],
        "n_detections": int(clu.get("n_detections", len(frame))),
        "n_modis": int(clu.get("n_modis", 0)),
        "n_viirs": int(clu.get("n_viirs", 0)),
        "first_observed_utc": clu.get("first_observation_utc"),
        "last_observed_utc": clu.get("last_observation_utc"),
        "observation_span_h": clu.get("observation_span_h"),
        "satellites": clu.get("satellites"),
        "timeline": timeline.to_dict("records"),
        "limitations": [
            "Satellite observations are not confirmed fires.",
            "Clusters are observation groups, not perimeters."],
    }


def build_science_payload(summary_path=None) -> dict | None:
    """Agent payload from a pilot summary file (read-only JSON)."""
    import json as _json

    path = Path(summary_path or "pilot_outputs/pilot_summary.json")
    if not path.is_file():
        return None
    try:
        summary = _json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    by_sensor = {}
    for sensor, vals in (summary.get("per_sensor") or {}).items():
        by_sensor[sensor] = {
            "n": int(vals.get("n", 0)),
            "observed": float((vals.get("observed_frp") or {}).get(
                "sum", 0.0)),
            "expected": float((vals.get("expected_frp") or {}).get(
                "sum", 0.0))}
    observed = (summary.get("observed_frp") or {}).get("sum")
    expected = (summary.get("expected_frp") or {}).get("sum")
    return {
        "method_version": summary.get("method_version"),
        "n": summary.get("n_rows"), "observed_sum": observed,
        "expected_sum": expected, "by_sensor": by_sensor,
        "mapping_applied": False,
        "inputs_missing": list(SCIENCE_MISSING_INPUTS),
        "limitations": [
            "Pilot input interpretation is exploratory (FIRMS FRP fed "
            "as model FRP); see the pilot report."],
    }


def build_event_payload_for(target_id: str, conn) -> tuple | None:
    """(kind, payload) for an event or alert id, or None when unknown."""
    from ..events import qwen, store

    event = store.get_event(conn, target_id)
    alert = store.get_alert(conn, target_id) if event is None else None
    if event is not None:
        linked = store.list_alerts(conn, event_id=event["event_id"],
                                   limit=50)
        return "event", qwen.build_event_payload(event, linked)
    if alert is not None:
        event = store.get_event(conn, alert["event_id"]) \
            if alert.get("event_id") else None
        return "event", qwen.build_alert_payload(alert, event)
    return None
