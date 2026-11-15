"""Read-only UI data access over pinned Phase 2 outputs (Phase 3B).

The calendar reads analytical results; it never recomputes them. All
functions here are framework-free (no Streamlit) so tests can call them
directly. Callers cache loaded documents (see 3E); these loaders just read.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from .. import settings

PRODUCTS = ("modis", "viirs-snpp")
SATELLITES = ("Terra", "Aqua", "N")
SATELLITE_LABELS = {"Terra": "MODIS Terra", "Aqua": "MODIS Aqua",
                    "N": "VIIRS Suomi-NPP"}
# Satellites the spec names that have no 2023 pilot data (never invented).
SATELLITES_UNAVAILABLE = ("NOAA-20", "NOAA-21")

METRICS_FILENAME = "metrics_2d.json"
INDEX_FILENAME = "index_2e.json"


def exec_chain() -> dict[str, str]:
    """Pinned Phase 2 execution ids (short provenance for the UI)."""
    return {"2A": settings.CAL_EXEC_2A, "2B": settings.CAL_EXEC_2B,
            "2C": settings.CAL_EXEC_2C, "2D": settings.CAL_EXEC_2D,
            "2E": settings.CAL_EXEC_2E}


def phase4_chain() -> dict[str, str]:
    """Pinned Phase 4 execution ids (separate from the Phase 2 chain)."""
    return {"4B": settings.CAL_EXEC_4B, "4C": settings.CAL_EXEC_4C}


def phase5_chain() -> dict[str, str]:
    """Pinned Phase 5 execution ids (empty until the 5A–5C build runs)."""
    return {"5A": settings.CAL_EXEC_5A, "5B": settings.CAL_EXEC_5B,
            "5C": settings.CAL_EXEC_5C}


def analytics_path(exec_id: str, filename: str) -> Path:
    """Resolve (and require) one file inside an execution output dir."""
    path = Path(settings.CAL_ANALYTICS_ROOT) / exec_id / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"missing analytical output: {path} "
            f"(exec {exec_id[:8]}; check CAL_EXEC_* settings)")
    return path


def load_metrics() -> dict:
    """Full Phase 2D metrics document (grains x metrics). ~1 MB, cacheable."""
    path = analytics_path(settings.CAL_EXEC_2D, METRICS_FILENAME)
    return json.loads(path.read_text(encoding="utf-8"))


def load_index() -> dict:
    """Phase 2E dual-index + reconciliation document."""
    path = analytics_path(settings.CAL_EXEC_2E, INDEX_FILENAME)
    return json.loads(path.read_text(encoding="utf-8"))


def load_anomalies() -> list[dict]:
    """Pinned Phase 4C anomaly results (~15k records, cacheable)."""
    path = analytics_path(settings.CAL_EXEC_4C, "anomalies_4c.json")
    return json.loads(path.read_text(encoding="utf-8"))


def grain_pack(metrics: dict, grain: str) -> dict:
    """One grain's three metric payloads (KeyError on unknown grain)."""
    try:
        return metrics[grain]
    except KeyError:
        raise ValueError(f"unknown grain {grain!r}; have {sorted(metrics)}")


def _row_by_period(rows: list[dict], period: str) -> dict | None:
    for row in rows:
        if row.get("period") == period:
            return row
    return None


def detection_row(metrics: dict, grain: str, period: str) -> dict | None:
    """Raw detection-activity row, or None when the period has no record."""
    pack = grain_pack(metrics, grain)
    return _row_by_period(pack["detection_activity"]["rows"], period)


def harmonized_row(metrics: dict, grain: str, period: str) -> dict | None:
    """Harmonized detection-activity row, or None when absent."""
    pack = grain_pack(metrics, grain)
    return _row_by_period(pack["detection_activity"]["harmonized_rows"], period)


def cluster_row(metrics: dict, grain: str, period: str) -> dict | None:
    """Cluster-activity row, or None when no clusters start in the period."""
    pack = grain_pack(metrics, grain)
    return _row_by_period(pack["cluster_activity"]["rows"], period)


def frp_rows(metrics: dict, grain: str, period: str) -> list[dict]:
    """Per-product FRP rows for the period ([] when absent)."""
    pack = grain_pack(metrics, grain)
    return [r for r in pack["frp_summary"]["rows"] if r.get("period") == period]


def catalog_spans(catalog: dict) -> dict[str, dict[str, str]]:
    """Per-dataset (min, max) acquisition dates from the Phase 1 catalog."""
    spans = {}
    for entry in catalog.get("datasets", []):
        versions = entry.get("versions") or []
        if not versions:
            continue
        temporal = (versions[-1].get("coverage") or {}).get("temporal") or {}
        lo, hi = temporal.get("min"), temporal.get("max")
        if lo and hi:
            spans[entry["dataset_id"]] = {"min": lo[:10], "max": hi[:10]}
    return spans


def coverage_for(day: str, spans: dict[str, dict[str, str]]) -> dict:
    """Coverage status for one UTC date (YYYY-MM-DD).

    Inside a dataset's verified span the pilot is complete (365/365 dates
    per product, Phase 1C), so a missing metric row means zero activity —
    never unavailable data. Outside every span the date is unavailable.
    Returns {"status": "complete"|"partial"|"unavailable", "products": {...}}.
    """
    products = {ds: (info["min"] <= day <= info["max"]) for ds, info in spans.items()}
    covered = sum(1 for ok in products.values() if ok)
    if not spans or covered == 0:
        status = "unavailable"
    elif covered == len(products):
        status = "complete"
    else:
        status = "partial"
    return {"status": status, "products": products}


# --- Bounded map queries (Phase 3C; read-only, parameterized, limited) ---

DETECTION_COLUMNS = ("detection_id", "satellite", "instrument", "product",
                     "sensor", "product_version", "acquisition_timestamp_utc",
                     "latitude", "longitude", "confidence", "frp", "daynight",
                     "scan", "track", "source_file", "source_sha256")
MAP_COLUMNS = ("detection_id", "satellite", "instrument", "product",
               "acquisition_timestamp_utc", "latitude", "longitude",
               "confidence", "frp", "daynight")
CLUSTER_MAP_COLUMNS = ("cluster_id", "n_detections", "n_modis", "n_viirs",
                       "first_observation_utc", "last_observation_utc",
                       "observation_span_h", "lon_min", "lon_max", "lat_min",
                       "lat_max", "centroid_lon", "centroid_lat", "frp_sum")
CLUSTER_TIMELINE_COLUMNS = ("detection_id", "acquisition_timestamp_utc",
                            "latitude", "longitude", "sensor", "satellite",
                            "product", "frp")
CLUSTER_MEMBER_ID_CAP = 10000
VIIRS_CONFIDENCE = ("l", "n", "h")


def _day_bounds(date_start: str, date_end: str) -> tuple[str, str]:
    """Inclusive date range -> [start_ts, end_exclusive_ts) ISO bounds."""
    try:
        start = date.fromisoformat(date_start)
        end = date.fromisoformat(date_end)
    except ValueError:
        raise ValueError("dates must be YYYY-MM-DD")
    if start > end:
        raise ValueError("date_start is after date_end")
    stop = end + timedelta(days=1)
    return (f"{start.isoformat()}T00:00:00+00:00",
            f"{stop.isoformat()}T00:00:00+00:00")


def month_cover(date_start: str, date_end: str) -> list[tuple[int, str]]:
    """(year, 'MM') pairs covering an inclusive date range (for pruning)."""
    start = date.fromisoformat(date_start)
    end = date.fromisoformat(date_end)
    cover = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        cover.append((year, f"{month:02d}"))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return cover


def fingerprint() -> tuple:
    """Dataset identity for cache keys (paths + sizes + mtimes + exec ids).

    Any analytical file change invalidates every cached UI query, so stale
    results after a dataset change are impossible by construction.
    """
    parts: list = [settings.CAL_DUCKDB_PATH, settings.CAL_EXEC_2B,
                   settings.CAL_EXEC_2C, settings.CAL_EXEC_2D,
                   settings.CAL_EXEC_4B, settings.CAL_EXEC_4C,
                   settings.CAL_EXEC_5A, settings.CAL_EXEC_5B,
                   settings.CAL_EXEC_5C]
    for raw in (settings.CAL_DUCKDB_PATH, settings.CAL_EVENTS_DB_PATH,
                str(Path(settings.CAL_ANALYTICS_ROOT) / settings.CAL_EXEC_2B
                    / "match_members.parquet"),
                str(Path(settings.CAL_ANALYTICS_ROOT) / settings.CAL_EXEC_2C
                    / "clusters.parquet"),
                str(Path(settings.CAL_ANALYTICS_ROOT) / settings.CAL_EXEC_2D
                    / METRICS_FILENAME),
                str(Path(settings.CAL_ANALYTICS_ROOT) / settings.CAL_EXEC_4B
                    / "baselines_4b.json"),
                str(Path(settings.CAL_ANALYTICS_ROOT) / settings.CAL_EXEC_4C
                    / "anomalies_4c.json")):
        try:
            stat = Path(raw).stat()
            parts.append((raw, stat.st_size, int(stat.st_mtime)))
        except OSError:
            parts.append((raw, None, None))
    return tuple(parts)


def _detection_where(products: tuple[str, ...], satellites: tuple[str, ...],
                     bbox: tuple[float, float, float, float] | None,
                     modis_conf_min: int | None,
                     viirs_conf: tuple[str, ...] | None,
                     date_start: str = "", date_end: str = "") -> tuple[str, list]:
    clauses = ["acquisition_timestamp_utc >= ?",
               "acquisition_timestamp_utc < ?"]
    params: list = []
    if date_start and date_end:  # hive partition pruning (verified 1:1, §3E)
        terms = []
        for year, month in month_cover(date_start, date_end):
            terms.append("(year = ? AND month = ?)")
            params.extend([year, month])
        clauses.append("(" + " OR ".join(terms) + ")")
    if products is not None:
        if len(products) == 0:
            clauses.append("FALSE")  # empty selection matches nothing
        else:
            clauses.append(f"product IN ({', '.join(['?'] * len(products))})")
            params.extend(products)
    if satellites is not None:
        if len(satellites) == 0:
            clauses.append("FALSE")
        else:
            clauses.append(f"satellite IN ({', '.join(['?'] * len(satellites))})")
            params.extend(satellites)
    if bbox:
        w, s, e, n = bbox
        clauses.append("longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?")
        params.extend([w, e, s, n])
    if modis_conf_min is not None:
        clauses.append("(product <> 'modis' OR "
                       "TRY_CAST(confidence AS INTEGER) >= ?)")
        params.append(int(modis_conf_min))
    if viirs_conf is not None:
        picked = [c for c in viirs_conf if c in VIIRS_CONFIDENCE]
        if picked:
            clauses.append(
                "(product <> 'viirs-snpp' OR "
                f"confidence IN ({', '.join(['?'] * len(picked))}))")
            params.extend(picked)
        else:  # empty selection removes VIIRS rows, keeps other products
            clauses.append("product <> 'viirs-snpp'")
    return " AND ".join(clauses), params


def query_detections(db_path: str | None = None, *, date_start: str,
                     date_end: str, products: tuple[str, ...] = PRODUCTS,
                     satellites: tuple[str, ...] = SATELLITES,
                     bbox: tuple[float, float, float, float] | None = None,
                     modis_conf_min: int | None = None,
                     viirs_conf: tuple[str, ...] | None = None,
                     limit: int = 5000,
                     columns: tuple[str, ...] = MAP_COLUMNS) -> pd.DataFrame:
    """Bounded detection rows (highest FRP first, id tiebreak: deterministic).

    Empty products/satellites means "no filter" is NOT applied: an empty
    selection matches nothing (the UI forbids it, but the query stays honest).
    """
    unknown = [c for c in columns if c not in DETECTION_COLUMNS]
    if unknown:
        raise ValueError(f"unknown detection columns: {unknown}")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    start_ts, stop_ts = _day_bounds(date_start, date_end)
    where, params = _detection_where(products, satellites, bbox,
                                     modis_conf_min, viirs_conf,
                                     date_start, date_end)
    cols = ", ".join(columns)
    con = duckdb.connect(db_path or settings.CAL_DUCKDB_PATH, read_only=True)
    try:
        return con.execute(
            f"SELECT {cols} FROM detections WHERE {where} "
            f"ORDER BY frp DESC NULLS LAST, detection_id LIMIT {int(limit)}",
            [start_ts, stop_ts, *params]).df()
    finally:
        con.close()


def count_detections(db_path: str | None = None, **kwargs) -> int:
    """Total rows matching the same filters (for cap notices)."""
    columns = kwargs.pop("columns", MAP_COLUMNS)  # noqa: F841 (same signature)
    limit = kwargs.pop("limit", 5000)  # noqa: F841
    start_ts, stop_ts = _day_bounds(kwargs["date_start"], kwargs["date_end"])
    where, params = _detection_where(
        kwargs.get("products", PRODUCTS), kwargs.get("satellites", SATELLITES),
        kwargs.get("bbox"), kwargs.get("modis_conf_min"),
        kwargs.get("viirs_conf"), kwargs["date_start"], kwargs["date_end"])
    con = duckdb.connect(db_path or settings.CAL_DUCKDB_PATH, read_only=True)
    try:
        return con.execute(
            f"SELECT COUNT(*) FROM detections WHERE {where}",
            [start_ts, stop_ts, *params]).fetchone()[0]
    finally:
        con.close()


def detection_by_id(detection_id: str,
                    db_path: str | None = None) -> dict | None:
    """One canonical detection record (None when the id is unknown)."""
    con = duckdb.connect(db_path or settings.CAL_DUCKDB_PATH, read_only=True)
    try:
        row = con.execute(
            f"SELECT {', '.join(DETECTION_COLUMNS)} FROM detections "
            f"WHERE detection_id = ? LIMIT 1", [detection_id]).fetchone()
    finally:
        con.close()
    return dict(zip(DETECTION_COLUMNS, row)) if row else None


def _parquet_literal(exec_id: str, filename: str) -> str:
    """Inline a settings-derived parquet path (quote-escaped; never user input)."""
    return str(analytics_path(exec_id, filename)).replace("'", "''")


def member_by_id(detection_id: str) -> dict | None:
    """Match-membership record for one detection (None when unknown)."""
    path = _parquet_literal(settings.CAL_EXEC_2B, "match_members.parquet")
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only, never writes analytical data
    try:
        row = con.execute(
            f"SELECT * FROM read_parquet('{path}') "
            f"WHERE detection_id = ? LIMIT 1", [detection_id]).fetchone()
        cols = [d[0] for d in con.description] if row else []
    finally:
        con.close()
    return dict(zip(cols, row)) if row else None


def members_of(detection_ids: list[str]) -> pd.DataFrame:
    """Match-membership rows for up to a few thousand ids (bounded)."""
    if not detection_ids:
        return pd.DataFrame()
    if len(detection_ids) > 10000:
        raise ValueError("members_of refuses > 10000 ids (bounded display)")
    path = _parquet_literal(settings.CAL_EXEC_2B, "match_members.parquet")
    frames = []
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only, never writes analytical data
    try:
        for lo in range(0, len(detection_ids), 500):
            chunk = detection_ids[lo:lo + 500]
            frames.append(con.execute(
                f"SELECT * FROM read_parquet('{path}') WHERE detection_id IN "
                f"({', '.join(['?'] * len(chunk))})", chunk).df())
    finally:
        con.close()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cluster_of(detection_ids: list[str]) -> pd.DataFrame:
    """Cluster-membership rows for up to a few thousand ids (bounded)."""
    if not detection_ids:
        return pd.DataFrame()
    if len(detection_ids) > 10000:
        raise ValueError("cluster_of refuses > 10000 ids (bounded display)")
    path = _parquet_literal(settings.CAL_EXEC_2C, "cluster_members.parquet")
    frames = []
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only, never writes analytical data
    try:
        for lo in range(0, len(detection_ids), 500):
            chunk = detection_ids[lo:lo + 500]
            frames.append(con.execute(
                f"SELECT * FROM read_parquet('{path}') WHERE detection_id IN "
                f"({', '.join(['?'] * len(chunk))})", chunk).df())
    finally:
        con.close()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def match_groups_of(match_ids: list[str]) -> pd.DataFrame:
    """Match-group rows for up to a few thousand ids (bounded)."""
    ids = sorted({m for m in match_ids if m})
    if not ids:
        return pd.DataFrame()
    if len(ids) > 10000:
        raise ValueError("match_groups_of refuses > 10000 ids (bounded)")
    path = _parquet_literal(settings.CAL_EXEC_2B, "match_groups.parquet")
    frames = []
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only, never writes analytical data
    try:
        for lo in range(0, len(ids), 500):
            chunk = ids[lo:lo + 500]
            frames.append(con.execute(
                f"SELECT * FROM read_parquet('{path}') WHERE match_id IN "
                f"({', '.join(['?'] * len(chunk))})", chunk).df())
    finally:
        con.close()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def match_group(match_id: str) -> dict | None:
    """One 2B match group record (None when the id is unknown)."""
    path = _parquet_literal(settings.CAL_EXEC_2B, "match_groups.parquet")
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only, never writes analytical data
    try:
        row = con.execute(
            f"SELECT * FROM read_parquet('{path}') "
            f"WHERE match_id = ? LIMIT 1", [match_id]).fetchone()
        cols = [d[0] for d in con.description] if row else []
    finally:
        con.close()
    return dict(zip(cols, row)) if row else None


def query_clusters(exec_id: str | None = None, *, date_start: str,
                   date_end: str,
                   bbox: tuple[float, float, float, float] | None = None,
                   limit: int = 5000,
                   columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Bounded clusters by first-observation date (largest first).

    First-observation attribution matches the 2D cluster metric, so map
    counts reconcile with the calendar. bbox filters the centroid.
    columns=None selects all; pass CLUSTER_MAP_COLUMNS for map rendering
    (skips the multi-KB member-id JSON per row).
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    start_ts, stop_ts = _day_bounds(date_start, date_end)
    path = _parquet_literal(exec_id or settings.CAL_EXEC_2C, "clusters.parquet")
    clauses = ["first_observation_utc >= ?", "first_observation_utc < ?"]
    params: list = [start_ts, stop_ts]
    if bbox:
        w, s, e, n = bbox
        clauses.append("centroid_lon BETWEEN ? AND ? "
                       "AND centroid_lat BETWEEN ? AND ?")
        params.extend([w, e, s, n])
    cols = ", ".join(columns) if columns else "*"
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only, never writes analytical data
    try:
        return con.execute(
            f"SELECT {cols} FROM read_parquet('{path}') "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY n_detections DESC, cluster_id LIMIT {int(limit)}",
            params).df()
    finally:
        con.close()


def count_clusters(exec_id: str | None = None, *, date_start: str,
                   date_end: str,
                   bbox: tuple[float, float, float, float] | None = None) -> int:
    """Total clusters matching the same filters (for cap notices)."""
    start_ts, stop_ts = _day_bounds(date_start, date_end)
    path = _parquet_literal(exec_id or settings.CAL_EXEC_2C, "clusters.parquet")
    clauses = ["first_observation_utc >= ?", "first_observation_utc < ?"]
    params: list = [start_ts, stop_ts]
    if bbox:
        w, s, e, n = bbox
        clauses.append("centroid_lon BETWEEN ? AND ? "
                       "AND centroid_lat BETWEEN ? AND ?")
        params.extend([w, e, s, n])
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only, never writes analytical data
    try:
        return con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{path}') "
            f"WHERE {' AND '.join(clauses)}", params).fetchone()[0]
    finally:
        con.close()


def detections_of_cluster(cluster_id: str, db_path: str | None = None,
                          columns: tuple[str, ...] = CLUSTER_TIMELINE_COLUMNS
                          ) -> pd.DataFrame:
    """Member detections of one cluster, oldest first (bounded, read-only).

    Ids come from the canonical 2C membership index; rows from the
    detections view (500-id chunks, same idiom as members_of). Unknown
    cluster -> empty frame. Refuses absurd member counts (pilot max is
    3,872; the cap only bites on corrupt inputs).
    """
    unknown = [c for c in columns if c not in DETECTION_COLUMNS]
    if unknown:
        raise ValueError(f"unknown detection columns: {unknown}")
    mem_path = _parquet_literal(settings.CAL_EXEC_2C, "cluster_members.parquet")
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only
    try:
        ids = [row[0] for row in con.execute(
            f"SELECT detection_id FROM read_parquet('{mem_path}') "
            f"WHERE cluster_id = ?", [cluster_id]).fetchall()]
    finally:
        con.close()
    if not ids:
        return pd.DataFrame(columns=list(columns))
    if len(ids) > CLUSTER_MEMBER_ID_CAP:
        raise ValueError("detections_of_cluster refuses > 10000 members "
                         "(bounded display)")
    cols = ", ".join(columns)
    frames = []
    con = duckdb.connect(db_path or settings.CAL_DUCKDB_PATH, read_only=True)
    try:
        for lo in range(0, len(ids), 500):
            chunk = ids[lo:lo + 500]
            frames.append(con.execute(
                f"SELECT {cols} FROM detections WHERE detection_id IN "
                f"({', '.join(['?'] * len(chunk))})", chunk).df())
    finally:
        con.close()
    frame = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame(columns=list(columns)))
    if len(frame):
        frame = frame.sort_values(
            ["acquisition_timestamp_utc", "detection_id"],
            kind="mergesort").reset_index(drop=True)
    return frame


def cluster_by_id(cluster_id: str, exec_id: str | None = None) -> dict | None:
    """One canonical cluster record (None when the id is unknown)."""
    path = _parquet_literal(exec_id or settings.CAL_EXEC_2C, "clusters.parquet")
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only, never writes analytical data
    try:
        row = con.execute(
            f"SELECT * FROM read_parquet('{path}') "
            f"WHERE cluster_id = ? LIMIT 1", [cluster_id]).fetchone()
        cols = [d[0] for d in con.description] if row else []
    finally:
        con.close()
    return dict(zip(cols, row)) if row else None
