"""Read-only UI data access over pinned Phase 2 outputs (Phase 3B).

The calendar reads analytical results; it never recomputes them. All
functions here are framework-free (no Streamlit) so tests can call them
directly. Callers cache loaded documents (see 3E); these loaders just read.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
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

STAGE_ENV_VARS: dict[str, str] = {
    "2A": "CAL_EXEC_2A",
    "2B": "CAL_EXEC_2B",
    "2C": "CAL_EXEC_2C",
    "2D": "CAL_EXEC_2D",
    "2E": "CAL_EXEC_2E",
    "4B": "CAL_EXEC_4B",
    "4C": "CAL_EXEC_4C",
    "5A": "CAL_EXEC_5A",
    "5B": "CAL_EXEC_5B",
    "5C": "CAL_EXEC_5C",
}


class DataUnavailableError(FileNotFoundError):
    """Raised when a required dataset, database, or analytical artifact is unavailable."""

    def __init__(self, message: str, *, status: dict | None = None) -> None:
        super().__init__(message)
        self.status: dict = status or {}


def _analytics_root() -> str:
    return settings.get_setting("CAL_ANALYTICS_ROOT", "analytics_output")


def stage_exec_id(stage: str) -> str:
    """Return configured execution ID for a pipeline stage (empty if unset)."""
    env_var = STAGE_ENV_VARS.get(stage.upper(), f"CAL_EXEC_{stage.upper()}")
    return settings.get_setting(env_var, "")


def _safe_segment(value: str, label: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise DataUnavailableError(f"Invalid {label}: empty value")
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise DataUnavailableError(f"Invalid {label}: {value!r}")
    return cleaned


def _registered_exec_ids(runs_path: str | None = None) -> set[str]:
    """Read execution_id values recorded in analytics_runs.jsonl if present."""
    raw = (runs_path if runs_path is not None
           else settings.get_setting("CAL_RUNS_PATH", "analytics_runs.jsonl"))
    if not raw:
        return set()
    p = settings.resolve_repo_path(raw)
    if not p.is_file():
        return set()
    ids: set[str] = set()
    try:
        with open(p, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                eid = rec.get("exec_id") or rec.get("execution_id")
                if isinstance(eid, str) and eid.strip():
                    ids.add(eid.strip())
    except OSError:
        return set()
    return ids


def duckdb_status(db_path: str | None = None) -> dict:
    """Validate the geospatial DuckDB store without ever creating an empty DB."""
    raw_path = (db_path if db_path is not None
                else settings.get_setting("CAL_DUCKDB_PATH", "")).strip()
    base = {
        "available": False,
        "status": "unknown",
        "path": raw_path,
        "resolved_path": "",
        "env_var": "CAL_DUCKDB_PATH",
        "reason": "",
        "remediation": (
            "Provision calendar.duckdb and normalized_parquet/ locally or set "
            "CAL_DUCKDB_PATH (and CAL_PARQUET_ROOT) via Streamlit Secrets / env."
        ),
    }
    if not raw_path:
        return {
            **base,
            "status": "unconfigured",
            "reason": "CAL_DUCKDB_PATH is not configured.",
        }
    resolved = settings.resolve_repo_path(raw_path)
    base["resolved_path"] = str(resolved)
    if not resolved.exists():
        return {
            **base,
            "status": "missing",
            "reason": f"DuckDB database does not exist at {raw_path}.",
        }
    if not resolved.is_file() or not os.access(resolved, os.R_OK):
        return {
            **base,
            "status": "invalid",
            "reason": f"DuckDB path is not a readable file: {raw_path}.",
        }
    try:
        if resolved.stat().st_size == 0:
            return {
                **base,
                "status": "invalid",
                "reason": f"DuckDB file is empty (0 bytes): {raw_path}.",
            }
    except OSError as exc:
        return {
            **base,
            "status": "invalid",
            "reason": f"Cannot stat DuckDB file {raw_path}: {exc}.",
        }

    def _probe(target: Path) -> None:
        con = duckdb.connect(str(target), read_only=True)
        try:
            con.execute("SELECT 1 FROM detections LIMIT 1").fetchall()
        finally:
            con.close()

    try:
        _probe(resolved)
    except duckdb.Error as exc:
        # Repo directory relocation / cloud mount handling: if the DuckDB detections
        # view points to stale Parquet paths while normalized_parquet exists,
        # rebuild the view cleanly from Parquet (in place if writable, else in tmp).
        parquet_raw = settings.get_setting("PARQUET_ROOT", settings.PARQUET_ROOT)
        parquet_dir = settings.resolve_repo_path(parquet_raw) if parquet_raw else Path("")
        if (
            "No files found" in str(exc)
            and parquet_dir.is_dir()
            and any(parquet_dir.rglob("*.parquet"))
        ):
            from ..processing import duckdb_layer

            candidates = []
            if os.access(resolved, os.W_OK) and os.access(resolved.parent, os.W_OK):
                candidates.append(resolved)
            tag = hashlib.sha256(str(parquet_dir.resolve()).encode("utf-8")).hexdigest()[:12]
            candidates.append(Path(tempfile.gettempdir()) / f"fireagent_calendar_{tag}.duckdb")

            last_err: Exception | None = None
            for target in candidates:
                if target != resolved and target.is_file() and target.stat().st_size > 0:
                    try:
                        _probe(target)
                        return {
                            **base,
                            "available": True,
                            "status": "ok",
                            "resolved_path": str(target),
                            "reason": "DuckDB geospatial database verified (cached view).",
                        }
                    except Exception:
                        pass
                try:
                    duckdb_layer.init_db(str(target), str(parquet_dir))
                    _probe(target)
                    return {
                        **base,
                        "available": True,
                        "status": "ok",
                        "resolved_path": str(target),
                        "reason": "DuckDB geospatial database verified (view refreshed).",
                    }
                except Exception as repair_exc:
                    last_err = repair_exc
            return {
                **base,
                "status": "invalid",
                "reason": f"DuckDB validation failed for {raw_path}: {last_err}",
            }
        return {
            **base,
            "status": "invalid",
            "reason": f"DuckDB validation failed for {raw_path}: {exc}",
        }
    except OSError as exc:
        return {
            **base,
            "status": "invalid",
            "reason": f"DuckDB file inaccessible at {raw_path}: {exc}",
        }
    return {
        **base,
        "available": True,
        "status": "ok",
        "reason": "DuckDB geospatial database verified.",
    }


def _connect_duckdb(db_path: str | None = None):
    info = duckdb_status(db_path)
    if not info["available"]:
        raise DataUnavailableError(
            f"DuckDB unavailable ({info['status']}): {info['reason']} "
            f"[{info['remediation']}]",
            status=info,
        )
    try:
        return duckdb.connect(info["resolved_path"], read_only=True)
    except duckdb.Error as exc:
        raise DataUnavailableError(
            f"Cannot open DuckDB {info['path']}: {exc}",
            status={**info, "available": False, "status": "invalid", "reason": str(exc)},
        ) from exc


def artifact_status(stage: str, filename: str, exec_id: str | None = None,
                    root: str | None = None) -> dict:
    """Validate an analytical execution output without fabricating missing data."""
    stg = stage.upper()
    env_var = STAGE_ENV_VARS.get(stg, f"CAL_EXEC_{stg}")
    eid = (stage_exec_id(stg) if exec_id is None else str(exec_id or "")).strip()
    analytics_root = (_analytics_root() if root is None else str(root or "")).strip() or "analytics_output"
    remediation = (
        f"Provide {filename} under {analytics_root}/<exec_id>/ or configure "
        f"CAL_ANALYTICS_ROOT and {env_var} via Streamlit Secrets or environment variables."
    )
    base = {
        "available": False,
        "status": "unknown",
        "stage": stg,
        "filename": filename,
        "exec_id": eid,
        "env_var": env_var,
        "root_env_var": "CAL_ANALYTICS_ROOT",
        "path": "",
        "resolved_path": "",
        "registered_in_runs": False,
        "reason": "",
        "remediation": remediation,
    }
    if not eid:
        return {
            **base,
            "status": "unconfigured",
            "reason": f"Execution ID for stage {stg} ({env_var}) is not configured.",
        }
    try:
        safe_eid = _safe_segment(eid, "execution_id")
        safe_file = _safe_segment(filename, "filename")
    except DataUnavailableError as exc:
        return {
            **base,
            "status": "invalid_exec_id",
            "reason": str(exc),
        }
    rel_path = Path(analytics_root) / safe_eid / safe_file
    resolved = settings.resolve_repo_path(rel_path)
    exec_dir = settings.resolve_repo_path(Path(analytics_root) / safe_eid)
    in_runs = safe_eid in _registered_exec_ids()
    base["path"] = str(rel_path)
    base["resolved_path"] = str(resolved)
    base["registered_in_runs"] = in_runs
    if resolved.is_file() and os.access(resolved, os.R_OK):
        try:
            if resolved.stat().st_size == 0:
                return {
                    **base,
                    "status": "missing_artifact",
                    "reason": f"Analytical artifact is empty (0 bytes): {rel_path}",
                }
        except OSError as exc:
            return {
                **base,
                "status": "missing_artifact",
                "reason": f"Cannot read analytical output {rel_path}: {exc}",
            }
        return {
            **base,
            "available": True,
            "status": "ok",
            "reason": f"Verified analytical output {rel_path}.",
        }
    if exec_dir.is_dir() or in_runs:
        return {
            **base,
            "status": "missing_artifact",
            "reason": (
                f"missing analytical output: {rel_path} "
                f"(exec {safe_eid[:8]}; check {env_var} / CAL_ANALYTICS_ROOT settings)"
            ),
        }
    return {
        **base,
        "status": "missing_execution",
        "reason": (
            f"missing analytical execution {safe_eid[:8]} and output {rel_path} "
            f"(check {env_var} / CAL_ANALYTICS_ROOT settings)"
        ),
    }


def _ensure_events_db_from_gz(resolved: Path) -> Path:
    """Decompress a sibling .sqlite.gz archive once when the raw .sqlite is absent."""
    if resolved.exists():
        return resolved
    gz_path = Path(f"{resolved}.gz")
    try:
        if not gz_path.is_file() or not os.access(gz_path, os.R_OK) or gz_path.stat().st_size == 0:
            return resolved
    except OSError:
        return resolved

    writable_parent = resolved.parent.exists() and os.access(resolved.parent, os.W_OK)
    tag = hashlib.sha256(str(gz_path.resolve()).encode("utf-8")).hexdigest()[:12]
    target = resolved if writable_parent else (Path(tempfile.gettempdir()) / f"{tag}_{resolved.name}")
    try:
        if target.is_file() and target.stat().st_size > 0:
            return target
    except OSError:
        pass

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as dst, gzip.open(gz_path, "rb") as src:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        os.replace(tmp_path, target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        return resolved
    return target


def events_db_status(db_path: str | None = None) -> dict:
    """Validate the Phase 5 SQLite events store without creating an empty DB."""
    raw_path = (db_path if db_path is not None
                else settings.get_setting("CAL_EVENTS_DB_PATH", "")).strip()
    base = {
        "available": False,
        "status": "unknown",
        "path": raw_path,
        "resolved_path": "",
        "env_var": "CAL_EVENTS_DB_PATH",
        "reason": "",
        "remediation": (
            "Generate the Phase 5 events database locally via "
            "`python -m scripts.cal_events_5 --from-analytics` or set "
            "CAL_EVENTS_DB_PATH to an existing events SQLite store."
        ),
    }
    if not raw_path:
        return {
            **base,
            "status": "unconfigured",
            "reason": "CAL_EVENTS_DB_PATH is not configured.",
        }
    resolved = _ensure_events_db_from_gz(settings.resolve_repo_path(raw_path))
    base["resolved_path"] = str(resolved)
    if not resolved.exists():
        return {
            **base,
            "status": "missing",
            "reason": f"Events database does not exist at {raw_path}.",
        }
    if not resolved.is_file() or not os.access(resolved, os.R_OK):
        return {
            **base,
            "status": "invalid",
            "reason": f"Events DB path is not a readable file: {raw_path}.",
        }
    try:
        if resolved.stat().st_size == 0:
            return {
                **base,
                "status": "invalid",
                "reason": f"Events DB file is empty (0 bytes): {raw_path}.",
            }
        uri = resolved.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            has_events = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        return {
            **base,
            "status": "invalid",
            "reason": f"Events DB validation failed for {raw_path}: {exc}",
        }
    if not has_events:
        return {
            **base,
            "status": "invalid",
            "reason": f"Events DB {raw_path} is missing the 'events' table.",
        }
    return {
        **base,
        "available": True,
        "status": "ok",
        "reason": "Events database verified.",
    }


def catalog_status(catalog_path: str | None = None) -> dict:
    """Validate catalog.json availability cleanly."""
    raw = (catalog_path if catalog_path is not None
           else settings.get_setting("CATALOG_PATH", "catalog.json")).strip()
    if not raw:
        return {"available": False, "status": "unconfigured", "path": "",
                "resolved_path": "", "env_var": "CAL_CATALOG_PATH",
                "reason": "CAL_CATALOG_PATH is not configured."}
    resolved = settings.resolve_repo_path(raw)
    if not resolved.is_file():
        return {"available": False, "status": "missing", "path": raw,
                "resolved_path": str(resolved), "env_var": "CAL_CATALOG_PATH",
                "reason": f"Catalog file does not exist at {raw}."}
    return {"available": True, "status": "ok", "path": raw,
            "resolved_path": str(resolved), "env_var": "CAL_CATALOG_PATH",
            "reason": "Phase 1 catalog verified."}


def storage_overview() -> dict:
    """Summarize verified availability of all datasets, databases, and artifacts."""
    return {
        "catalog": catalog_status(),
        "duckdb": duckdb_status(),
        "events_db": events_db_status(),
        "artifacts": {
            "2A": artifact_status("2A", "validation_2a.json"),
            "2B_members": artifact_status("2B", "match_members.parquet"),
            "2B_groups": artifact_status("2B", "match_groups.parquet"),
            "2C_clusters": artifact_status("2C", "clusters.parquet"),
            "2C_members": artifact_status("2C", "cluster_members.parquet"),
            "2D_metrics": artifact_status("2D", METRICS_FILENAME),
            "2E_index": artifact_status("2E", INDEX_FILENAME),
            "4B_baselines": artifact_status("4B", "baselines_4b.json"),
            "4C_anomalies": artifact_status("4C", "anomalies_4c.json"),
        },
    }


def exec_chain() -> dict[str, str]:
    """Pinned Phase 2 execution ids (short provenance for the UI)."""
    return {s: stage_exec_id(s) for s in ("2A", "2B", "2C", "2D", "2E")}


def phase4_chain() -> dict[str, str]:
    """Pinned Phase 4 execution ids (separate from the Phase 2 chain)."""
    return {"4B": stage_exec_id("4B"), "4C": stage_exec_id("4C")}


def phase5_chain() -> dict[str, str]:
    """Pinned Phase 5 execution ids (empty until the 5A–5C build runs)."""
    return {"5A": stage_exec_id("5A"), "5B": stage_exec_id("5B"),
            "5C": stage_exec_id("5C")}


def analytics_path(exec_id: str, filename: str) -> Path:
    """Resolve (and require) one file inside an execution output dir."""
    if exec_id is None or not str(exec_id).strip():
        raise DataUnavailableError(
            "missing analytical output: execution ID is not configured "
            "(check CAL_EXEC_* settings)")
    safe_eid = _safe_segment(str(exec_id), "execution_id")
    safe_file = _safe_segment(filename, "filename")
    rel = Path(_analytics_root()) / safe_eid / safe_file
    resolved = settings.resolve_repo_path(rel)
    if not resolved.is_file() or not os.access(resolved, os.R_OK):
        raise DataUnavailableError(
            f"missing analytical output: {rel} "
            f"(exec {safe_eid[:8]}; check CAL_EXEC_* settings)")
    return resolved


def load_metrics() -> dict:
    """Full Phase 2D metrics document (grains x metrics). ~1 MB, cacheable."""
    path = analytics_path(stage_exec_id("2D"), METRICS_FILENAME)
    return json.loads(path.read_text(encoding="utf-8"))


def load_index() -> dict:
    """Phase 2E dual-index + reconciliation document."""
    path = analytics_path(stage_exec_id("2E"), INDEX_FILENAME)
    return json.loads(path.read_text(encoding="utf-8"))


def load_anomalies() -> list[dict]:
    """Pinned Phase 4C anomaly results (~15k records, cacheable)."""
    path = analytics_path(stage_exec_id("4C"), "anomalies_4c.json")
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
    duckdb_path = settings.get_setting("CAL_DUCKDB_PATH", "")
    events_path = settings.get_setting("CAL_EVENTS_DB_PATH", "")
    aroot = _analytics_root()
    e2b = stage_exec_id("2B")
    e2c = stage_exec_id("2C")
    e2d = stage_exec_id("2D")
    e4b = stage_exec_id("4B")
    e4c = stage_exec_id("4C")
    e5a = stage_exec_id("5A")
    e5b = stage_exec_id("5B")
    e5c = stage_exec_id("5C")
    parts: list = [duckdb_path, e2b, e2c, e2d, e4b, e4c, e5a, e5b, e5c]
    for raw in (duckdb_path, events_path,
                str(Path(aroot) / e2b / "match_members.parquet") if e2b else "",
                str(Path(aroot) / e2c / "clusters.parquet") if e2c else "",
                str(Path(aroot) / e2d / METRICS_FILENAME) if e2d else "",
                str(Path(aroot) / e4b / "baselines_4b.json") if e4b else "",
                str(Path(aroot) / e4c / "anomalies_4c.json") if e4c else ""):
        if not raw:
            parts.append((raw, None, None))
            continue
        try:
            stat = settings.resolve_repo_path(raw).stat()
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
    con = _connect_duckdb(db_path)
    try:
        return con.execute(
            f"SELECT {cols} FROM detections WHERE {where} "
            f"ORDER BY frp DESC NULLS LAST, detection_id LIMIT {int(limit)}",
            [start_ts, stop_ts, *params]).df()
    except duckdb.Error as exc:
        raise DataUnavailableError(f"DuckDB query failed: {exc}") from exc
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
    con = _connect_duckdb(db_path)
    try:
        return con.execute(
            f"SELECT COUNT(*) FROM detections WHERE {where}",
            [start_ts, stop_ts, *params]).fetchone()[0]
    except duckdb.Error as exc:
        raise DataUnavailableError(f"DuckDB count query failed: {exc}") from exc
    finally:
        con.close()


def detection_by_id(detection_id: str,
                    db_path: str | None = None) -> dict | None:
    """One canonical detection record (None when the id is unknown)."""
    con = _connect_duckdb(db_path)
    try:
        row = con.execute(
            f"SELECT {', '.join(DETECTION_COLUMNS)} FROM detections "
            f"WHERE detection_id = ? LIMIT 1", [detection_id]).fetchone()
    except duckdb.Error as exc:
        raise DataUnavailableError(f"DuckDB lookup failed: {exc}") from exc
    finally:
        con.close()
    return dict(zip(DETECTION_COLUMNS, row)) if row else None


def _parquet_literal(exec_id: str, filename: str) -> str:
    """Inline a settings-derived parquet path (quote-escaped; never user input)."""
    return str(analytics_path(exec_id, filename)).replace("'", "''")


def _parquet_in_chunks(path: str, column: str, ids: list[str],
                       label: str) -> pd.DataFrame:
    """Query a Parquet file for rows whose `column` is in `ids` (500-id chunks)."""
    frames = []
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only, never writes analytical data
    try:
        for lo in range(0, len(ids), 500):
            chunk = ids[lo:lo + 500]
            frames.append(con.execute(
                f"SELECT * FROM read_parquet('{path}') WHERE {column} IN "
                f"({', '.join(['?'] * len(chunk))})", chunk).df())
    except duckdb.Error as exc:
        raise DataUnavailableError(f"{label} query failed: {exc}") from exc
    finally:
        con.close()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def member_by_id(detection_id: str) -> dict | None:
    """Match-membership record for one detection (None when unknown)."""
    path = _parquet_literal(stage_exec_id("2B"), "match_members.parquet")
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only, never writes analytical data
    try:
        row = con.execute(
            f"SELECT * FROM read_parquet('{path}') "
            f"WHERE detection_id = ? LIMIT 1", [detection_id]).fetchone()
        cols = [d[0] for d in con.description] if row else []
    except duckdb.Error as exc:
        raise DataUnavailableError(f"Match-member artifact query failed: {exc}") from exc
    finally:
        con.close()
    return dict(zip(cols, row)) if row else None


def members_of(detection_ids: list[str]) -> pd.DataFrame:
    """Match-membership rows for up to a few thousand ids (bounded)."""
    if not detection_ids:
        return pd.DataFrame()
    if len(detection_ids) > 10000:
        raise ValueError("members_of refuses > 10000 ids (bounded display)")
    path = _parquet_literal(stage_exec_id("2B"), "match_members.parquet")
    return _parquet_in_chunks(path, "detection_id", detection_ids,
                              "Match-member artifact")


def cluster_of(detection_ids: list[str]) -> pd.DataFrame:
    """Cluster-membership rows for up to a few thousand ids (bounded)."""
    if not detection_ids:
        return pd.DataFrame()
    if len(detection_ids) > 10000:
        raise ValueError("cluster_of refuses > 10000 ids (bounded display)")
    path = _parquet_literal(stage_exec_id("2C"), "cluster_members.parquet")
    return _parquet_in_chunks(path, "detection_id", detection_ids,
                              "Cluster-member artifact")


def match_groups_of(match_ids: list[str]) -> pd.DataFrame:
    """Match-group rows for up to a few thousand ids (bounded)."""
    ids = sorted({m for m in match_ids if m})
    if not ids:
        return pd.DataFrame()
    if len(ids) > 10000:
        raise ValueError("match_groups_of refuses > 10000 ids (bounded)")
    path = _parquet_literal(stage_exec_id("2B"), "match_groups.parquet")
    return _parquet_in_chunks(path, "match_id", ids, "Match-group artifact")


def match_group(match_id: str) -> dict | None:
    """One 2B match group record (None when the id is unknown)."""
    path = _parquet_literal(stage_exec_id("2B"), "match_groups.parquet")
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only, never writes analytical data
    try:
        row = con.execute(
            f"SELECT * FROM read_parquet('{path}') "
            f"WHERE match_id = ? LIMIT 1", [match_id]).fetchone()
        cols = [d[0] for d in con.description] if row else []
    except duckdb.Error as exc:
        raise DataUnavailableError(f"Match-group artifact query failed: {exc}") from exc
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
    eid = exec_id if exec_id is not None else stage_exec_id("2C")
    path = _parquet_literal(eid, "clusters.parquet")
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
    except duckdb.Error as exc:
        raise DataUnavailableError(f"Cluster artifact query failed: {exc}") from exc
    finally:
        con.close()


def count_clusters(exec_id: str | None = None, *, date_start: str,
                   date_end: str,
                   bbox: tuple[float, float, float, float] | None = None) -> int:
    """Total clusters matching the same filters (for cap notices)."""
    start_ts, stop_ts = _day_bounds(date_start, date_end)
    eid = exec_id if exec_id is not None else stage_exec_id("2C")
    path = _parquet_literal(eid, "clusters.parquet")
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
    except duckdb.Error as exc:
        raise DataUnavailableError(f"Cluster count query failed: {exc}") from exc
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
    mem_path = _parquet_literal(stage_exec_id("2C"), "cluster_members.parquet")
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only
    try:
        ids = [row[0] for row in con.execute(
            f"SELECT detection_id FROM read_parquet('{mem_path}') "
            f"WHERE cluster_id = ?", [cluster_id]).fetchall()]
    except duckdb.Error as exc:
        raise DataUnavailableError(f"Cluster-member lookup failed: {exc}") from exc
    finally:
        con.close()
    if not ids:
        return pd.DataFrame(columns=list(columns))
    if len(ids) > CLUSTER_MEMBER_ID_CAP:
        raise ValueError("detections_of_cluster refuses > 10000 members "
                         "(bounded display)")
    cols = ", ".join(columns)
    frames = []
    con = _connect_duckdb(db_path)
    try:
        for lo in range(0, len(ids), 500):
            chunk = ids[lo:lo + 500]
            frames.append(con.execute(
                f"SELECT {cols} FROM detections WHERE detection_id IN "
                f"({', '.join(['?'] * len(chunk))})", chunk).df())
    except duckdb.Error as exc:
        raise DataUnavailableError(f"Cluster detection query failed: {exc}") from exc
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
    eid = exec_id if exec_id is not None else stage_exec_id("2C")
    path = _parquet_literal(eid, "clusters.parquet")
    con = duckdb.connect()  # ephemeral in-memory; SELECT-only, never writes analytical data
    try:
        row = con.execute(
            f"SELECT * FROM read_parquet('{path}') "
            f"WHERE cluster_id = ? LIMIT 1", [cluster_id]).fetchone()
        cols = [d[0] for d in con.description] if row else []
    except duckdb.Error as exc:
        raise DataUnavailableError(f"Cluster lookup failed: {exc}") from exc
    finally:
        con.close()
    return dict(zip(cols, row)) if row else None
