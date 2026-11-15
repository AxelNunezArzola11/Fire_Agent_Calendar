"""Stage 5A: deterministic event-to-observation association (assoc_v1).

Methodology: one event is seeded per Phase 2C hotspot cluster (1:1).
This reuses the project's only approved spatiotemporal association
(matcher_v1/match_params_v1 + cluster_v1/cluster_params_v1) and therefore
introduces zero new spatial radii or temporal windows:

- A cluster is never assumed to BE an event: the event is a separate
  record with its own stable ID, lifecycle state, severity, and
  provenance. The source cluster ID is preserved as lineage.
- Events are never merged for sharing a broad region; each cluster maps
  to exactly one event and each event references exactly its cluster(s).
- Every detection lands in exactly one cluster (Phase 2C invariant), so
  no observation is orphaned or double-associated. Singleton clusters
  become explicit singleton-observation events (flagged), never dropped.
- Ambiguous evidence is flagged explicitly from existing cluster fields
  (no new thresholds except the reused map-display constant for the
  very-large-group annotation).

Pure functions over cluster records; persistence lives in store.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ..ui.map_view import LARGE_CLUSTER_WARN
from .model import (ASSOC_PARAMS_VERSION, ASSOC_VERSION, MODEL_VERSION,
                    STATE_ACTIVE, check_severity)

logger = logging.getLogger(__name__)

FRP_AGGREGATION = (
    "sum-of-reported-FRP (repeated observations included; "
    "NOT total emitted energy; per-sensor rows not directly comparable)"
)

# Short suffix identifying the association generation in event IDs. A
# future assoc_v2 seeds "ev-<cluster>-a2" IDs, so generations never collide.
ASSOC_ID_SUFFIX = "a1"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def event_id_for(cluster_id: str) -> str:
    """Stable deterministic event ID for a source cluster ID."""
    if not cluster_id or not isinstance(cluster_id, str):
        raise ValueError(f"invalid cluster_id {cluster_id!r}")
    return f"ev-{cluster_id}-{ASSOC_ID_SUFFIX}"


def _as_list(value, *, what: str) -> list:
    """Parse a JSON list field from a cluster record (loud on garbage)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        raise ValueError(f"unparseable {what}: {value!r}")
    if not isinstance(parsed, list):
        raise ValueError(f"{what} is not a list: {value!r}")
    return parsed


def ambiguity_flags(cluster: dict) -> list[str]:
    """Explicit association-quality flags from existing cluster fields.

    Every flag is a structural fact of the pinned 2C record — no new
    spatial/temporal thresholds. The very-large-group cutoff reuses the
    existing map-display constant (map_view.LARGE_CLUSTER_WARN).
    """
    flags: list[str] = []
    n = int(cluster.get("n_detections", 0) or 0)
    n_modis = int(cluster.get("n_modis", 0) or 0)
    n_viirs = int(cluster.get("n_viirs", 0) or 0)
    match_groups = _as_list(cluster.get("match_groups"), what="match_groups")
    completeness_frac = cluster.get("completeness_frac")
    span_h = cluster.get("observation_span_h") or 0.0
    if n <= 1:
        flags.append("singleton-observation")
    if n_modis == 0 or n_viirs == 0:
        flags.append("single-sensor-only")
    if n > 1 and not match_groups:
        flags.append("no-cross-sensor-match")
    if isinstance(completeness_frac, (int, float)) and completeness_frac < 1.0:
        flags.append("gappy-observation-span")
    if n > LARGE_CLUSTER_WARN:
        flags.append("very-large-group")
    if n > 1 and span_h == 0.0:
        flags.append("same-overpass-group")
    return flags


def _frp_by_sensor(cluster: dict) -> dict:
    raw = cluster.get("frp_by_sensor")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        raise ValueError(f"unparseable frp_by_sensor: {raw!r}")
    return parsed if isinstance(parsed, dict) else {}


def build_event_record(cluster: dict, *, exec_id: str,
                       created_at: str | None = None,
                       coverage_status: str = "complete") -> dict:
    """Build one versioned event record from one 2C cluster record.

    New events seed into the Active state with Informational severity;
    the 5B lifecycle engine and 5C alert engine own subsequent state and
    severity. Raises ValueError on missing lineage or malformed fields —
    association never guesses.
    """
    cluster_id = cluster.get("cluster_id")
    if not cluster_id:
        raise ValueError("cluster record lacks cluster_id (lineage required)")
    first = cluster.get("first_observation_utc")
    last = cluster.get("last_observation_utc")
    if not first or not last:
        raise ValueError(f"cluster {cluster_id} lacks observation timestamps")
    n = int(cluster.get("n_detections", 0) or 0)
    if n < 1:
        raise ValueError(f"cluster {cluster_id} has no detections")
    satellites = _as_list(cluster.get("satellites"), what="satellites")
    match_groups = _as_list(cluster.get("match_groups"), what="match_groups")
    by_sensor = _frp_by_sensor(cluster)
    now = created_at or _utcnow()
    severity = "Informational"
    check_severity(severity)
    return {
        "event_id": event_id_for(cluster_id),
        "model_version": MODEL_VERSION,
        "assoc_version": ASSOC_VERSION,
        "assoc_params_version": ASSOC_PARAMS_VERSION,
        "first_observed_utc": first,
        "last_observed_utc": last,
        "last_evaluated_utc": None,
        "lifecycle_state": STATE_ACTIVE,
        "severity_name": severity,
        "severity_level": 0,
        "severity_provenance": "seed-default; 5C alert engine owns severity",
        "sensor_modis_n": int(cluster.get("n_modis", 0) or 0),
        "sensor_viirs_n": int(cluster.get("n_viirs", 0) or 0),
        "satellites": satellites,
        "cluster_ids": [cluster_id],
        "anomaly_refs": [],
        "detection_count": n,
        # Detection-level IDs resolve on demand via the pinned 2C record
        # (mega-clusters hold thousands of IDs; copying them per event
        # would duplicate analytical data without adding evidence).
        "detection_refs": None,
        "detection_resolution": {
            "via": "clusters.parquet detection_ids",
            "cluster_id": cluster_id,
            "exec_2c": cluster.get("exec_id", ""),
        },
        "activity": {
            "n_detections": n,
            "observation_span_h": cluster.get("observation_span_h"),
            "completeness_dates": cluster.get("completeness_dates"),
            "completeness_span_days": cluster.get("completeness_span_days"),
            "completeness_frac": cluster.get("completeness_frac"),
        },
        "frp": {
            "aggregation": FRP_AGGREGATION,
            "unit": "MW",
            "frp_sum": cluster.get("frp_sum"),
            "frp_mean": cluster.get("frp_mean"),
            "frp_max": cluster.get("frp_max"),
            "frp_n": cluster.get("frp_n"),
            "frp_null": cluster.get("frp_null"),
            "by_sensor": by_sensor,
        },
        "bbox": {
            "lon_min": cluster.get("lon_min"),
            "lon_max": cluster.get("lon_max"),
            "lat_min": cluster.get("lat_min"),
            "lat_max": cluster.get("lat_max"),
        },
        "centroid": {
            "lon": cluster.get("centroid_lon"),
            "lat": cluster.get("centroid_lat"),
        },
        "completeness": {
            "coverage_status": coverage_status,
            "sensors_available": sorted(
                {s for s in
                 (["MODIS"] if cluster.get("n_modis") else [])
                 + (["VIIRS"] if cluster.get("n_viirs") else [])}),
            "match_groups": match_groups,
        },
        "ambiguity": {
            "flags": ambiguity_flags(cluster),
            "note": ("Flags describe association evidence quality from the "
                     "pinned 2C record; flagged events are preserved, never "
                     "merged or dropped."),
        },
        "review_required": False,
        "review_status": "none",
        "provenance": {
            "created_by_exec": exec_id,
            "created_at": now,
            "updated_by_exec": exec_id,
            "updated_at": now,
            "assoc": ASSOC_VERSION,
            "assoc_params": ASSOC_PARAMS_VERSION,
            "cluster_algo": cluster.get("algo_version", ""),
            "cluster_params": cluster.get("params_version", ""),
            "cluster_exec": cluster.get("exec_id", ""),
        },
    }


# Lifecycle-managed fields a re-seed must never reset (5B integration:
# the 5A seeder owns observations; 5B owns state, 5C owns severity).
RESEED_PRESERVED = ("lifecycle_state", "last_evaluated_utc", "severity_name",
                    "severity_level", "severity_provenance", "anomaly_refs",
                    "review_required", "review_status")

# Observation-derived fields a re-seed refreshes when new activity arrives.
RESEED_REFRESHED = ("first_observed_utc", "last_observed_utc",
                    "sensor_modis_n", "sensor_viirs_n", "satellites",
                    "cluster_ids", "detection_count", "detection_refs",
                    "detection_resolution", "activity", "frp", "bbox",
                    "centroid", "completeness", "ambiguity")


def merge_reseed(old: dict, fresh: dict, *, exec_id: str,
                 updated_at: str | None = None) -> dict:
    """Merge a re-seeded record over an existing event (5B-safe).

    Observation-derived fields refresh from the new record; lifecycle
    state, evaluation stamp, severity, anomaly refs, and review fields
    are preserved so a re-run after 5B/5C can never silently reset them.
    Provenance keeps its creation entry and records the reseed update.
    """
    if old["event_id"] != fresh["event_id"]:
        raise ValueError("reseed merge requires identical event_id")
    merged = dict(old)
    for field in RESEED_REFRESHED:
        merged[field] = fresh[field]
    provenance = dict(old.get("provenance", {}))
    provenance["updated_by_exec"] = exec_id
    provenance["updated_at"] = updated_at or _utcnow()
    merged["provenance"] = provenance
    return merged


def seed_events(clusters: list[dict], *, exec_id: str,
                coverage_status: str = "complete") -> list[dict]:
    """Build event records for every cluster (deterministic order).

    Duplicate cluster IDs are refused (they would double-associate
    observations); callers stream clusters in cluster_id order so output
    order is stable across runs.
    """
    seen: set[str] = set()
    events = []
    for cluster in clusters:
        cid = cluster.get("cluster_id")
        if cid in seen:
            raise ValueError(f"duplicate cluster_id in association input: {cid}")
        seen.add(cid)
        events.append(build_event_record(
            cluster, exec_id=exec_id, coverage_status=coverage_status))
    logger.info("assoc %s seeded %d events", ASSOC_VERSION, len(events))
    return events
