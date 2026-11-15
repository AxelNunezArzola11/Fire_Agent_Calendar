"""Phase 2A: dataset discovery + analytical input validation.

Reads the Phase 1 catalog and DuckDB views (read-only). Never modifies
data. Produces a deterministic validation report; invalid records are
reported with example ids, never deleted.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import settings
from ..processing import catalog as catalog_mod
from ..processing import duckdb_layer

logger = logging.getLogger(__name__)

EXAMPLE_LIMIT = 20


def _resolve_local_path(raw_path: str, marker: str, root: str | Path) -> Path:
    """Resolve a recorded path, falling back to `root` if the repo was relocated."""
    p = Path(raw_path)
    if p.exists():
        return p
    if marker in raw_path:
        return settings.resolve_repo_path(Path(root) / raw_path.split(marker, 1)[1])
    return p


def discover_inputs(catalog_path: str, db_path: str) -> list[dict]:
    """Enumerate analytical input datasets with catalog + live cross-checks."""
    catalog_data = catalog_mod.load_catalog(catalog_path)
    live = {(d["product"], d["sensor"]): d["rows"]
            for d in duckdb_layer.datasets(db_path)}
    datasets = []
    for entry in catalog_data.get("datasets", []):
        latest = entry["versions"][-1] if entry.get("versions") else {}
        raw = latest.get("raw_files", [{}])[0]
        counts = latest.get("record_counts", {})
        key = None
        for (product, sensor) in live:
            if entry["dataset_id"].startswith(product):
                key = (product, sensor)
                break
        raw_path = raw.get("path", "")
        resolved_raw = (
            _resolve_local_path(raw_path, "data_raw_firms/", settings.RAW_ROOT)
            if raw_path else None
        )
        datasets.append({
            "dataset_id": entry["dataset_id"],
            "product": key[0] if key else None,
            "sensor": key[1] if key else None,
            "processing_version": latest.get("processing_version"),
            "source_file": raw.get("path"),
            "source_sha256": raw.get("sha256"),
            "source_size_bytes": raw.get("size_bytes"),
            "catalog_counts": counts,
            "live_rows": live.get(key) if key else None,
            "counts_agree": (live.get(key) == counts.get("accepted")) if key else False,
            "coverage": latest.get("coverage", {}),
            "raw_file_present": resolved_raw.is_file() if resolved_raw else False,
        })
    return datasets


def _scalar(db_path: str, sql: str, params: list | None = None):
    import duckdb

    con = duckdb.connect(db_path, read_only=True)
    try:
        return con.execute(sql, params or []).fetchall()
    finally:
        con.close()


def validate_inputs(catalog_path: str, db_path: str) -> dict:
    """Run the full input-validation battery. Returns the report dict."""
    datasets = discover_inputs(catalog_path, db_path)
    checks: list[dict] = []
    invalid_examples: list[dict] = []

    def check(name: str, status: str, detail: str):
        checks.append({"check": name, "status": status, "detail": detail})

    # Per-dataset live inspection (sensor-separated by construction).
    for ds in datasets:
        product, sensor = ds["product"], ds["sensor"]
        if product is None:
            check(f"{ds['dataset_id']}:live-view", "fail", "no live rows for dataset")
            continue
        filt, params = "product = ? AND sensor = ?", [product, sensor]
        n = _scalar(db_path, f"SELECT COUNT(*) FROM detections WHERE {filt}", params)[0][0]
        distinct_ids = _scalar(
            db_path, f"SELECT COUNT(DISTINCT detection_id) FROM detections WHERE {filt}",
            params)[0][0]
        bad_coords = _scalar(
            db_path,
            f"SELECT COUNT(*) FROM detections WHERE {filt} AND ("
            "longitude IS NULL OR latitude IS NULL OR "
            "longitude < -180 OR longitude > 180 OR latitude < -90 OR latitude > 90)",
            params)[0][0]
        bad_ts = _scalar(
            db_path,
            f"SELECT COUNT(*) FROM detections WHERE {filt} AND "
            "try_strptime(acquisition_timestamp_utc, '%Y-%m-%dT%H:%M:%S%z') IS NULL",
            params)[0][0]
        null_frp = _scalar(
            db_path, f"SELECT COUNT(*) FROM detections WHERE {filt} AND frp IS NULL",
            params)[0][0]
        empty_conf = _scalar(
            db_path, f"SELECT COUNT(*) FROM detections WHERE {filt} "
            "AND (confidence IS NULL OR confidence = '')", params)[0][0]
        sats = _scalar(
            db_path, f"SELECT DISTINCT satellite FROM detections WHERE {filt} "
            "ORDER BY satellite", params)
        instruments = _scalar(
            db_path, f"SELECT DISTINCT instrument FROM detections WHERE {filt} "
            "ORDER BY instrument", params)
        span = _scalar(
            db_path,
            f"SELECT MIN(acquisition_timestamp_utc), MAX(acquisition_timestamp_utc) "
            f"FROM detections WHERE {filt}", params)[0]
        ds.update({
            "satellites": [r[0] for r in sats],
            "instruments": [r[0] for r in instruments],
            "live_span": {"min": span[0], "max": span[1]},
            "null_frp": null_frp,
            "empty_confidence": empty_conf,
        })
        dupes = n - distinct_ids
        check(f"{ds['dataset_id']}:counts", "pass" if ds["counts_agree"] else "fail",
              f"live={n} catalog={ds['catalog_counts'].get('accepted')}")
        check(f"{ds['dataset_id']}:detection-ids",
              "pass" if dupes == 0 else "warn", f"duplicate_ids={dupes}")
        check(f"{ds['dataset_id']}:coords",
              "pass" if bad_coords == 0 else "fail", f"invalid_coords={bad_coords}")
        check(f"{ds['dataset_id']}:timestamps",
              "pass" if bad_ts == 0 else "fail", f"unparseable_ts={bad_ts}")
        check(f"{ds['dataset_id']}:frp",
              "pass" if null_frp == 0 else "warn", f"null_frp={null_frp}")
        check(f"{ds['dataset_id']}:confidence",
              "pass" if empty_conf == 0 else "warn", f"empty_confidence={empty_conf}")
        check(f"{ds['dataset_id']}:lineage",
              "pass" if ds["raw_file_present"] and ds["source_sha256"] else "fail",
              f"raw_present={ds['raw_file_present']} sha={'set' if ds['source_sha256'] else 'missing'}")
        if bad_coords or bad_ts or dupes:
            rows = _scalar(
                db_path,
                f"SELECT detection_id, satellite, acquisition_timestamp_utc, "
                f"longitude, latitude FROM detections WHERE {filt} AND ("
                "longitude IS NULL OR latitude IS NULL OR "
                "longitude < -180 OR longitude > 180 OR latitude < -90 OR latitude > 90 OR "
                "try_strptime(acquisition_timestamp_utc, '%Y-%m-%dT%H:%M:%S%z') IS NULL) "
                f"ORDER BY detection_id LIMIT {EXAMPLE_LIMIT}", params)
            for did, sat, ts, lon, lat in rows:
                invalid_examples.append({"detection_id": did, "dataset": ds["dataset_id"],
                                         "satellite": sat, "ts": ts, "lon": lon, "lat": lat})

    overall = "pass"
    if any(c["status"] == "fail" for c in checks):
        overall = "fail"
    elif any(c["status"] == "warn" for c in checks):
        overall = "warn"
    return {"datasets": datasets, "checks": checks,
            "invalid_examples": invalid_examples, "overall": overall}
